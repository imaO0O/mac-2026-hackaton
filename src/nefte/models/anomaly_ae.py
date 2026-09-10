"""LSTM-автоэнкодер как альтернатива расстоянию Махаланобиса.

`models/anomaly.py` ловит нетипичное СОЧЕТАНИЕ мгновенных значений. Он линейный по
построению: эллипсоид в шести переменных. Автоэнкодер на окне видит другое —
нетипичную ДИНАМИКУ: режим, в котором каждое мгновенное сочетание допустимо, а
последовательность переходов такой в истории не встречалась.

Честно о цене. В `models/anomaly.py` записано, почему изначально выбран Махаланобис:
он детерминирован, считается на CPU за секунды и раскладывается по вкладам
переменных. Автоэнкодер всё это ухудшает: нужен torch, обучение зависит от сида,
а «вклад переменной» превращается в долю ошибки восстановления — величину куда
менее прозрачную. Поэтому он остаётся **опцией**, а не заменой: детектор по
умолчанию прежний, и демо работает на машине без видеокарты.

Нормировка и порог считаются ТОЛЬКО по обучающему периоду — иначе аномалия
2026 года измерялась бы относительно 2026 года, то есть относительно будущего.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nefte.config import ROOT

MODELS_DIR = ROOT / "models"


def _windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Скользящие окна ``[n, window, channels]`` и маска полных окон.

    Окно с меткой t содержит отсчёты (t-window, t] — только прошлое, как и везде
    в проекте.
    """
    n, c = values.shape
    out = np.full((n, window, c), np.nan, dtype="float32")
    ok = np.zeros(n, dtype=bool)
    for i in range(window - 1, n):
        chunk = values[i - window + 1:i + 1]
        if not np.isnan(chunk).any():
            out[i] = chunk
            ok[i] = True
    return out, ok


def _make_net(n_channels: int, hidden: int, latent: int):
    """Кодировщик-декодировщик на LSTM. torch импортируется лениво."""
    import torch
    from torch import nn

    class AutoEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(n_channels, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_channels)

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            z = self.to_latent(h[-1])
            seed = self.from_latent(z).unsqueeze(1).repeat(1, x.shape[1], 1)
            out, _ = self.decoder(seed)
            return self.head(out)

    return AutoEncoder().to(torch.float32)


@dataclass
class LSTMAnomalyDetector:
    """Аномальность режима как ошибка восстановления окна."""

    columns: list[str] = field(default_factory=list)
    window: int = 24
    hidden: int = 32
    latent: int = 8
    epochs: int = 200
    patience: int = 20
    lr: float = 3e-3
    batch_size: int = 256
    seed: int = 42
    quantile: float = 0.99
    threshold: float = 0.0
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    net: object | None = None
    device: str = "cpu"
    history: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def fitted(self) -> bool:
        return self.net is not None and self.threshold > 0

    def _normalize(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].to_numpy(dtype="float32")
        return (values - self.center) / self.scale

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str] | None = None,
            train: tuple[str, str] | None = None, window: int = 24,
            quantile: float = 0.99, prefer_gpu: bool = True,
            **kwargs) -> "LSTMAnomalyDetector":
        import torch

        columns = [c for c in (columns or frame.columns) if c in frame.columns]
        obj = cls(columns=columns, window=window, quantile=quantile, **kwargs)
        obj.device = "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"

        data = frame[columns]
        train_part = data.loc[train[0]:train[1]] if train else data
        clean = train_part.dropna()
        if len(clean) < window * 20 or not columns:
            return obj

        # робастная нормировка по обучающему периоду: медиана и IQR, чтобы сами
        # выбросы не задавали норму (то же правило, что у Махаланобиса)
        values = clean.to_numpy(dtype="float64")
        obj.center = np.median(values, axis=0).astype("float32")
        q75, q25 = np.percentile(values, [75, 25], axis=0)
        scale = ((q75 - q25) / 1.349).astype("float32")
        obj.scale = np.where(scale <= 0, 1.0, scale).astype("float32")

        windows, ok = _windows(obj._normalize(train_part), window)
        train_x = windows[ok]
        if len(train_x) < 100:
            return obj

        # последние 20 % обучающего периода — на раннюю остановку. Не val-период
        # проекта: он нужен для честной оценки, а не для подбора эпох.
        split = int(len(train_x) * 0.8)
        torch.manual_seed(obj.seed)
        np.random.seed(obj.seed)
        net = _make_net(len(columns), obj.hidden, obj.latent).to(obj.device)
        opt = torch.optim.Adam(net.parameters(), lr=obj.lr, weight_decay=1e-5)
        loss_fn = torch.nn.MSELoss()

        xs = torch.tensor(train_x[:split], device=obj.device)
        xv = torch.tensor(train_x[split:], device=obj.device)
        generator = torch.Generator(device="cpu").manual_seed(obj.seed)
        best_state, best_loss, bad, epoch = None, float("inf"), 0, 0

        for epoch in range(obj.epochs):
            net.train()
            order = torch.randperm(len(xs), generator=generator).to(obj.device)
            for start in range(0, len(order), obj.batch_size):
                batch = xs[order[start:start + obj.batch_size]]
                opt.zero_grad()
                loss = loss_fn(net(batch), batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(net(xv), xv)) if len(xv) else float(loss)
            if vloss < best_loss - 1e-6:
                best_loss, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= obj.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        obj.net = net
        obj.history = {"epochs": epoch + 1, "val_mse": best_loss,
                       "n_windows": int(len(train_x)), "device": obj.device}

        errors = obj.scores(train_part).dropna()
        obj.threshold = float(np.nanquantile(errors, quantile)) if len(errors) else 0.0
        return obj

    # ------------------------------------------------------------------ #
    def _reconstruct(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ошибка восстановления по каналам и маска полных окон."""
        import torch

        windows, ok = _windows(self._normalize(frame), self.window)
        per_channel = np.full((len(frame), len(self.columns)), np.nan, dtype="float32")
        if not ok.any():
            return per_channel, ok
        with torch.no_grad():
            batch = torch.tensor(windows[ok], device=self.device)
            recon = self.net(batch).cpu().numpy()
        # усредняем квадрат ошибки по времени окна, оставляя разбивку по каналам
        per_channel[ok] = ((windows[ok] - recon) ** 2).mean(axis=1)
        return per_channel, ok

    def scores(self, frame: pd.DataFrame) -> pd.Series:
        """Ошибка восстановления на каждый момент (NaN там, где окно неполное)."""
        if self.net is None:
            return pd.Series(np.nan, index=frame.index)
        per_channel, ok = self._reconstruct(frame)
        out = np.full(len(frame), np.nan, dtype="float32")
        out[ok] = per_channel[ok].mean(axis=1)
        return pd.Series(out, index=frame.index)

    def normalized(self, frame: pd.DataFrame) -> pd.Series:
        """Аномальность в тех же единицах, что у Махаланобиса: 1.0 — уровень порога."""
        if not self.fitted:
            return pd.Series(np.nan, index=frame.index)
        return self.scores(frame) / self.threshold

    def contributions_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Доля каждого канала в ошибке восстановления — то, что показываем оператору.

        Это слабее покомпонентного разложения расстояния Махаланобиса: там вклад
        имеет смысл «на сколько сигм эта переменная выбивается с учётом связей»,
        здесь — лишь «какой канал сеть восстановила хуже».
        """
        per_channel, ok = self._reconstruct(frame)
        total = np.nansum(per_channel, axis=1, keepdims=True)
        total[total == 0] = 1.0
        shares = np.full_like(per_channel, np.nan)
        shares[ok] = per_channel[ok] / total[ok]
        return pd.DataFrame(shares, index=frame.index, columns=self.columns)

    def flags(self, frame: pd.DataFrame) -> pd.Series:
        if not self.fitted:
            return pd.Series(False, index=frame.index)
        return (self.scores(frame) > self.threshold).fillna(False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def default_path() -> Path:
        return MODELS_DIR / "anomaly_ae"

    def save(self, path: Path | None = None) -> Path:
        import torch

        path = Path(path) if path else self.default_path()
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "ae.pt")
        meta = {"columns": self.columns, "window": self.window, "hidden": self.hidden,
                "latent": self.latent, "seed": self.seed, "quantile": self.quantile,
                "threshold": self.threshold, "center": self.center.tolist(),
                "scale": self.scale.tolist(), "history": self.history}
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None,
             prefer_gpu: bool = False) -> "LSTMAnomalyDetector":
        """По умолчанию на CPU: демо обязано работать без видеокарты."""
        import torch

        path = Path(path) if path else cls.default_path()
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        obj = cls(columns=meta["columns"], window=meta["window"], hidden=meta["hidden"],
                  latent=meta["latent"], seed=meta["seed"], quantile=meta["quantile"],
                  threshold=meta["threshold"], history=meta.get("history", {}))
        obj.center = np.array(meta["center"], dtype="float32")
        obj.scale = np.array(meta["scale"], dtype="float32")
        obj.device = "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"
        net = _make_net(len(obj.columns), obj.hidden, obj.latent)
        net.load_state_dict(torch.load(path / "ae.pt", map_location=obj.device))
        obj.net = net.to(obj.device).eval()
        return obj
