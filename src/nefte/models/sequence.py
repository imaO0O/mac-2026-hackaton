"""Нейросетевой виртуальный анализатор серы: окно телеметрии вместо строки признаков.

Зачем он нужен. Бустинг видит момент времени как набор из 54 чисел, часть которых —
уже посчитанные нами скользящие средние. Сеть на последовательности получает сырое
окно телеметрии и решает сама, что в нём важно: уровень, скорость или форма
переходного процесса. На гидроочистке это осмысленно — качество реагирует на режим
с запаздыванием в часы, и форма подхода к режиму значит не меньше уровня.

Честная рамка, из-за которой ожидания сдержанные:

* лабораторных анализов всего ~1450, из них 983 в обучающем периоде. Для сети это
  очень мало, поэтому модель намеренно маленькая (GRU на 48 скрытых единиц) и
  усредняется по нескольким сидам;
* сравнение идёт по тем же правилам, что у бустинга: то же разбиение по времени с
  эмбарго, те же базовые (ПАК и предыдущий анализ), та же конформная калибровка
  интервала и тот же способ выбрать порог тревоги;
* сеть — замена бустингу только если побьёт его на валидации. Проиграет —
  результат всё равно записывается: отрицательный результат тоже результат.

Контракт с системой ровно тот, что описан в docs/GPU_SETUP.md §3: ``horizon_hours``,
``alarm_threshold``, ``predict_with_sigma(state)``, ``risk_for_state(state)``.
Поэтому модель подключается к ``QualityAgent`` без единой правки в агентах, а
оптимизатор пользуется ею через кинетический суррогат.

Обучение идёт на GPU, если он есть, инференс работает и на CPU: демо обязано
запускаться на машине без видеокарты (docs/GPU_SETUP.md §6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nefte.agents.schemas import ProcessState
from nefte.config import ROOT
from nefte.models.quality_model import (
    Z90,
    conformal_sigma_scale,
    interval_metrics,
    interval_risk,
    pick_alarm_threshold,
    pick_threshold_for_budget,
)

MODELS_DIR = ROOT / "models"
# базовый сид основной конфигурации; с другим сидом — проверка устойчивости
DEFAULT_SEED_BASE = 42

# Каналы окна: сырые теги режима, производные признаки режима, поточный анализатор
# и лабораторный контекст. Скользящие средние сюда НЕ входят намеренно — их работу
# должна делать сама сеть, иначе сравнение с бустингом теряет смысл.
DEFAULT_CHANNELS = [
    # реакторный блок
    # квенч F14 и перепад Р-202 P8 — по таблице тегов 15.09 (было F15 и W10:
    # сигнал без связи с нагрузкой и массовый расход бензина)
    "ht_T5", "ht_T6", "ht_T11", "ht_P13", "ht_F26", "ht_F14", "ht_F2", "ht_P8",
    "ht_F17",
    # режим как физика
    "reg_wabt", "reg_dt_react", "reg_h2_oil", "reg_h2_partial", "reg_kinetic",
    "reg_quench_ratio", "reg_run_hours",
    # АВТ: что приходит в гидроочистку
    "avt_T55", "avt_F30", "avt_F32",
    # оперативная и лабораторная оценка качества
    "pak_sulfur", "lims_sulfur_prev", "lims_feed_sulfur_mgkg",
]

# Цель предобучения: показание поточного анализатора. Его в истории 265 тысяч
# отсчётов против 983 лабораторных анализов, и это единственный способ дать сети
# столько данных, сколько ей нужно. Сам канал из входа при этом убирается —
# иначе задача вырождается в тождество.
PRETRAIN_TARGET = "pak_sulfur"

QUANTILES = (0.1, 0.5, 0.9)


def torch_device(prefer_gpu: bool = True) -> str:
    """cuda, если карта есть и её просили; иначе cpu."""
    import torch

    return "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"


# --------------------------------------------------------------------------- #
# сборка окон
# --------------------------------------------------------------------------- #

def build_windows(features: pd.DataFrame, index: pd.DatetimeIndex,
                  channels: list[str], window: int,
                  horizon_hours: float) -> tuple[np.ndarray, np.ndarray]:
    """Окна телеметрии длиной ``window`` часов, заканчивающиеся за H часов до анализа.

    Возвращает ``(X, ok)``: тензор ``[n, window, channels]`` и маску тех анализов,
    для которых окно удалось собрать. Правило «только прошлое» соблюдается тем же
    способом, что и в табличном датасете: позиция ищется по правой границе
    ``ts - horizon``, а само окно берётся строго до неё включительно.
    """
    frame = features[channels]
    fidx = frame.index
    values = frame.to_numpy(dtype="float32")

    lag = pd.Timedelta(hours=horizon_hours)
    positions = fidx.searchsorted(pd.DatetimeIndex(index) - lag, side="right") - 1

    out = np.full((len(index), window, len(channels)), np.nan, dtype="float32")
    ok = np.zeros(len(index), dtype=bool)
    for i, pos in enumerate(positions):
        if pos < window - 1:
            continue
        out[i] = values[pos - window + 1:pos + 1]
        ok[i] = True
    return out, ok


def fill_windows(X: np.ndarray, median: np.ndarray) -> np.ndarray:
    """Пропуски внутри окна: сначала протягиваем последнее известное, потом медиана.

    Протягивание вперёд физически честное — это «последнее измеренное значение»,
    ровно то, что видит оператор. Медиана обучающего периода нужна для начала
    окна, где протягивать нечего.
    """
    filled = X.copy()
    n, w, c = filled.shape
    for t in range(1, w):
        gap = np.isnan(filled[:, t, :])
        filled[:, t, :][gap] = filled[:, t - 1, :][gap]
    for t in range(w - 2, -1, -1):        # начало окна тянем назад от первого known
        gap = np.isnan(filled[:, t, :])
        filled[:, t, :][gap] = filled[:, t + 1, :][gap]
    gap = np.isnan(filled)
    filled[gap] = np.broadcast_to(median, (n, w, c))[gap]
    return filled


# --------------------------------------------------------------------------- #
# сети
# --------------------------------------------------------------------------- #

def _make_net(arch: str, n_channels: int, hidden: int, dropout: float):
    """GRU или TCN — обе выдают три квантили. Строится лениво, torch импортируется здесь."""
    import torch
    from torch import nn

    class GRUHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(n_channels, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, len(QUANTILES))

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.drop(out[:, -1, :]))

    class TCNHead(nn.Module):
        """Причинная свёртка с расширением: рецептивное поле растёт как 2^k."""

        def __init__(self) -> None:
            super().__init__()
            layers, in_ch = [], n_channels
            for dilation in (1, 2, 4, 8):
                layers += [
                    nn.ConstantPad1d(((3 - 1) * dilation, 0), 0.0),
                    nn.Conv1d(in_ch, hidden, kernel_size=3, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_ch = hidden
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, len(QUANTILES))

        def forward(self, x):
            out = self.body(x.transpose(1, 2))
            return self.head(out[:, :, -1])

    net = {"gru": GRUHead, "tcn": TCNHead}[arch]()
    return net.to(torch.float32)


def pinball_loss(pred, target, quantiles=QUANTILES):
    """Квантильная потеря: та же, что у CatBoost с Quantile:alpha."""
    import torch

    losses = []
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        losses.append(torch.maximum(q * err, (q - 1) * err))
    return torch.stack(losses, dim=1).mean()


# --------------------------------------------------------------------------- #

@dataclass
class SulfurSequenceModel:
    """Виртуальный анализатор серы на окне телеметрии.

    Интерфейс намеренно совпадает с ``SulfurModel``: агенту качества всё равно,
    что внутри — бустинг или сеть.
    """

    horizon_hours: float = 0.0
    window: int = 24
    arch: str = "gru"
    hidden: int = 48
    dropout: float = 0.2
    epochs: int = 300
    patience: int = 30
    lr: float = 3e-3
    batch_size: int = 64
    seeds: tuple[int, ...] = (42, 43, 44)
    channels: list[str] = field(default_factory=lambda: list(DEFAULT_CHANNELS))
    limit: float = 10.0
    sigma_scale: float = 1.0
    alarm_threshold: float = 0.5
    alarm_reliable: bool = True
    alarm_threshold_fbeta: float | None = None
    device: str = "cpu"
    pretrained: bool = False
    metrics: dict = field(default_factory=dict)

    # заполняется при обучении
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    median_: np.ndarray | None = None
    nets: list = field(default_factory=list)
    feature_matrix: pd.DataFrame | None = None
    history: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def _prepare(self, features: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        X, ok = build_windows(features, index, self.channels, self.window,
                              self.horizon_hours)
        X = fill_windows(X, self.median_)
        X[~ok] = self.median_          # окна, которые не собрались, — нейтральные
        return (X - self.mean_) / self.std_

    def _train_net(self, net, X, y, val, epochs: int, lr: float, patience: int,
                   seed: int) -> tuple[float, int]:
        """Один цикл обучения с ранней остановкой. Возвращает ``(лучшая потеря, эпох)``."""
        import torch

        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        best_state, best_loss, bad, epoch = None, float("inf"), 0, 0

        for epoch in range(epochs):
            net.train()
            order = torch.randperm(len(X), generator=generator).to(self.device)
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                opt.zero_grad()
                loss = pinball_loss(net(X[idx]), y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if val is None:
                continue
            net.eval()
            with torch.no_grad():
                vloss = float(pinball_loss(net(val[0]), val[1]))
            if vloss < best_loss - 1e-5:
                best_loss, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return best_loss, epoch + 1

    def _pretrain_batch(self, features: pd.DataFrame, train_bounds: tuple[str, str]):
        """Данные предобучения: окна и показания поточного анализатора.

        Берётся только обучающий период и только те моменты, где прибор ЖИВОЙ:
        замороженные показания — не измерение, и учить на них сеть значит учить её
        воспроизводить поломку.
        """
        import torch

        if PRETRAIN_TARGET not in features.columns:
            return None
        part = features.loc[train_bounds[0]:train_bounds[1]]
        target = part[PRETRAIN_TARGET]
        alive = target.notna()
        if "pak_frozen" in part.columns:
            alive &= part["pak_frozen"].fillna(1.0) < 0.5
        index = part.index[alive.to_numpy()]
        if len(index) < 500:
            return None

        raw, ok = build_windows(features, index, self.channels, self.window, 0.0)
        raw = fill_windows(raw, self.median_)
        raw[~ok] = self.median_
        X = ((raw - self.mean_) / self.std_)[ok]
        y = target.loc[index].to_numpy(dtype="float32")[ok]
        # последние 10 % обучающего периода — на раннюю остановку предобучения.
        # Валидационный период проекта не трогаем: он нужен для честной оценки.
        split = int(len(X) * 0.9)
        return (torch.tensor(X[:split], device=self.device),
                torch.tensor(y[:split], device=self.device),
                torch.tensor(X[split:], device=self.device),
                torch.tensor(y[split:], device=self.device))

    def fit(self, features: pd.DataFrame, y_train: pd.Series,
            y_val: pd.Series | None = None, prefer_gpu: bool = True,
            pretrain_bounds: tuple[str, str] | None = None,
            pretrain_epochs: int = 40) -> "SulfurSequenceModel":
        import torch

        self.channels = [c for c in self.channels if c in features.columns]
        # предобучаемся предсказывать поточный анализатор — значит, самого его во
        # входе быть не должно, иначе сеть выучит тождество, а не физику
        if pretrain_bounds is not None and PRETRAIN_TARGET in self.channels:
            self.channels = [c for c in self.channels if c != PRETRAIN_TARGET]
        self.device = torch_device(prefer_gpu)
        self.pretrained = pretrain_bounds is not None

        raw, ok = build_windows(features, y_train.index, self.channels, self.window,
                                self.horizon_hours)
        flat = raw[ok].reshape(-1, len(self.channels))
        self.median_ = np.nanmedian(flat, axis=0).astype("float32")
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_).astype("float32")
        filled = fill_windows(raw, self.median_)
        filled[~ok] = self.median_
        self.mean_ = filled[ok].reshape(-1, len(self.channels)).mean(axis=0).astype("float32")
        std = filled[ok].reshape(-1, len(self.channels)).std(axis=0).astype("float32")
        self.std_ = np.where(std < 1e-6, 1.0, std).astype("float32")

        Xtr = ((filled - self.mean_) / self.std_)[ok]
        ytr = y_train.to_numpy(dtype="float32")[ok]
        val = None
        if y_val is not None and len(y_val):
            Xva = self._prepare(features, y_val.index)
            val = (torch.tensor(Xva, device=self.device),
                   torch.tensor(y_val.to_numpy(dtype="float32"), device=self.device))

        Xtr_t = torch.tensor(Xtr, device=self.device)
        ytr_t = torch.tensor(ytr, device=self.device)

        pre = (self._pretrain_batch(features, pretrain_bounds)
               if pretrain_bounds is not None else None)

        self.nets, self.history = [], []
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = _make_net(self.arch, len(self.channels), self.hidden,
                            self.dropout).to(self.device)

            record = {"seed": seed}
            if pre is not None:
                # фаза 1: учимся воспроизводить поточный анализатор по телеметрии.
                # Меток здесь десятки тысяч, и именно они дают сети структуру.
                loss, epochs = self._train_net(net, pre[0], pre[1], (pre[2], pre[3]),
                                               pretrain_epochs, self.lr,
                                               max(5, self.patience // 3), seed)
                record.update({"pretrain_pinball": loss, "pretrain_epochs": epochs,
                               "pretrain_n": int(len(pre[0]))})

            # фаза 2: дообучение на лабораторных анализах. Скорость обучения ниже:
            # 983 примера легко разрушают то, что выучено на десятках тысяч.
            lr = self.lr / 5 if pre is not None else self.lr
            loss, epochs = self._train_net(net, Xtr_t, ytr_t, val, self.epochs, lr,
                                           self.patience, seed)
            record.update({"val_pinball": loss, "epochs": epochs})
            self.nets.append(net)
            self.history.append(record)
        return self

    # ------------------------------------------------------------------ #
    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Средний по сидам прогноз трёх квантилей."""
        import torch

        with torch.no_grad():
            tensor = torch.tensor(X, device=self.device)
            preds = [net(tensor).cpu().numpy() for net in self.nets]
        out = np.mean(preds, axis=0)
        # квантили обязаны идти по возрастанию: усреднение по сидам этого не гарантирует
        return np.sort(out, axis=1)

    def predict_index(self, index: pd.DatetimeIndex,
                      features: pd.DataFrame | None = None) -> pd.DataFrame:
        """``q10, q50, q90, sigma`` для набора моментов."""
        features = self.feature_matrix if features is None else features
        if features is None:
            raise RuntimeError("модель не привязана к матрице признаков: "
                               "SulfurSequenceModel.attach(features)")
        raw = self._raw_predict(self._prepare(features, pd.DatetimeIndex(index)))
        out = pd.DataFrame(raw, columns=["q10", "q50", "q90"],
                           index=pd.DatetimeIndex(index))
        out["sigma"] = ((out["q90"] - out["q10"]) / (2 * Z90)).clip(lower=0.1) * self.sigma_scale
        return out

    def predict_risk(self, index: pd.DatetimeIndex,
                     features: pd.DataFrame | None = None) -> pd.Series:
        """P(сера > предела) из интервала.

        Отдельного классификатора у сети нет намеренно: на 983 анализах и 148
        превышениях второй обучаемый блок переобучится быстрее, чем принесёт
        пользу, а интервал уже откалиброван конформно.
        """
        return interval_risk(self.predict_index(index, features), self.limit)

    def calibrate(self, features: pd.DataFrame, y_val: pd.Series,
                  target: float = 0.8) -> float:
        self.sigma_scale = 1.0
        self.sigma_scale = conformal_sigma_scale(
            self.predict_index(y_val.index, features), y_val, target)
        return self.sigma_scale

    def select_alarm_threshold(self, features: pd.DataFrame, y_val: pd.Series,
                               beta: float = 1.5, min_lift: float = 1.5,
                               budget: float | None = None) -> float:
        risk = self.predict_risk(y_val.index, features).to_numpy()
        threshold, reliable = pick_alarm_threshold(
            risk, (y_val > self.limit).to_numpy(), beta=beta, min_lift=min_lift)
        self.alarm_threshold_fbeta = threshold
        self.alarm_reliable = reliable
        self.alarm_threshold = (threshold if budget is None
                                else pick_threshold_for_budget(risk, budget))
        return self.alarm_threshold

    def evaluate(self, features: pd.DataFrame, y: pd.Series,
                 limit: float | None = None) -> dict:
        limit = self.limit if limit is None else limit
        out = {"risk_source": "interval", "alarm_reliable": self.alarm_reliable}
        out.update(interval_metrics(self.predict_index(y.index, features), y,
                                    self.predict_risk(y.index, features),
                                    limit, self.alarm_threshold))
        return out

    # ------------------------------------------------------------------ #
    # контракт с QualityAgent
    # ------------------------------------------------------------------ #
    def attach(self, feature_matrix: pd.DataFrame) -> "SulfurSequenceModel":
        """Присоединяет матрицу признаков, проверяя, что модели есть чем питаться.

        Проверка здесь, а не «как получится» глубже. Сохранённая модель хранит
        СПИСОК КАНАЛОВ, с которыми обучалась, и матрица с тех пор могла уехать.
        Так и вышло: после перевода серы сырья в мг/кг колонка сменила имя
        (``lims_feed_sulfur`` → ``lims_feed_sulfur_mgkg``), и модель на горизонте
        2 ч, обученная до этого, падала внутри ``build_windows`` с
        ``KeyError: ['lims_feed_sulfur'] not in index`` — из середины подготовки
        окон, где ни модели, ни причины не видно.

        Устаревший артефакт модели — не то же, что устаревший отчёт. Отчёт даёт
        неверные числа, и это ловит контракт свежести; модель ПАДАЕТ, и падает
        так, что причину надо раскапывать. Поэтому отказ должен называть и
        модель, и недостающие каналы, и что с этим делать.
        """
        missing = [c for c in self.channels if c not in feature_matrix.columns]
        if missing:
            raise RuntimeError(
                f"модель обучена на каналах, которых в матрице больше нет: "
                f"{', '.join(missing)}. Матрица признаков изменилась с момента "
                f"обучения — переобучите: scripts/train_sequence.py "
                f"--horizon {self.horizon_hours:g} --arch {self.arch} "
                f"--window {self.window}"
                + (" --pretrain" if self.pretrained else ""))
        self.feature_matrix = feature_matrix
        return self

    def predict_with_sigma(self, state: ProcessState) -> tuple[float, float]:
        if self.feature_matrix is None:
            return float("nan"), float("inf")
        ts = pd.Timestamp(state.ts)
        idx = self.feature_matrix.index
        # окно должно заканчиваться не позже момента состояния
        if idx.searchsorted(ts, side="right") - 1 < self.window - 1:
            return float("nan"), float("inf")
        pred = self.predict_index(pd.DatetimeIndex([ts]))
        return float(pred["q50"].iloc[0]), float(pred["sigma"].iloc[0])

    def risk_for_state(self, state: ProcessState) -> float | None:
        mean, sigma = self.predict_with_sigma(state)
        if mean != mean:
            return None
        frame = pd.DataFrame({"q50": [mean], "sigma": [sigma]})
        return float(interval_risk(frame, self.limit).iloc[0])

    # ------------------------------------------------------------------ #
    @staticmethod
    def default_path(horizon_hours: float, arch: str = "gru", window: int = 24,
                     pretrained: bool = False, seed_base: int = DEFAULT_SEED_BASE) -> Path:
        """Окно, предобучение и базовый сид входят в имя: конфигурации не затирают
        друг друга.

        Сид добавлен после того, как обещание выше оказалось неправдой. Проверка
        устойчивости по сидам (``--seed-base 100``, ``200``) писала модель в тот же
        каталог, что и основная конфигурация, — имя ОТЧЁТА сид содержало, имя
        МОДЕЛИ нет. В итоге в ``sulfur_seq_tcn48_pre_h2`` лежала модель на сидах
        200–202, отчёт с тем же именем описывал сиды 42–44, а загрузчик выбрал её по
        валидации, и прогон по тестовому периоду считался на сидовом варианте.
        Суффикс тот же, что у отчёта, чтобы пара «модель — отчёт» читалась по имени.
        """
        tag = (f"{arch}{window}" + ("_pre" if pretrained else "")
               + ("" if seed_base == DEFAULT_SEED_BASE else f"_s{seed_base}"))
        return MODELS_DIR / f"sulfur_seq_{tag}_h{horizon_hours:g}"

    def save(self, path: Path | None = None) -> Path:
        import torch

        path = Path(path) if path else self.default_path(
            self.horizon_hours, self.arch, self.window, self.pretrained,
            seed_base=min(self.seeds) if self.seeds else DEFAULT_SEED_BASE)
        path.mkdir(parents=True, exist_ok=True)
        for i, net in enumerate(self.nets):
            torch.save(net.state_dict(), path / f"net{i}.pt")
        meta = {
            "horizon_hours": self.horizon_hours, "window": self.window,
            "arch": self.arch, "hidden": self.hidden, "dropout": self.dropout,
            "seeds": list(self.seeds), "channels": self.channels,
            "limit": self.limit, "sigma_scale": self.sigma_scale,
            "alarm_threshold": self.alarm_threshold,
            "alarm_reliable": self.alarm_reliable, "pretrained": self.pretrained,
            "alarm_threshold_fbeta": self.alarm_threshold_fbeta,
            "mean": self.mean_.tolist(), "std": self.std_.tolist(),
            "median": self.median_.tolist(),
            "metrics": self.metrics, "history": self.history,
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path, prefer_gpu: bool = False) -> "SulfurSequenceModel":
        """Загрузка. По умолчанию на CPU: демо обязано работать без видеокарты."""
        import torch

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        obj = cls(horizon_hours=meta["horizon_hours"], window=meta["window"],
                  arch=meta["arch"], hidden=meta["hidden"], dropout=meta["dropout"],
                  seeds=tuple(meta["seeds"]), channels=meta["channels"],
                  limit=meta["limit"], sigma_scale=meta["sigma_scale"],
                  alarm_threshold=meta["alarm_threshold"],
                  alarm_reliable=meta.get("alarm_reliable", True),
                  pretrained=meta.get("pretrained", False),
                  alarm_threshold_fbeta=meta.get("alarm_threshold_fbeta"),
                  metrics=meta.get("metrics", {}), history=meta.get("history", []))
        obj.mean_ = np.array(meta["mean"], dtype="float32")
        obj.std_ = np.array(meta["std"], dtype="float32")
        obj.median_ = np.array(meta["median"], dtype="float32")
        obj.device = torch_device(prefer_gpu)
        for file in sorted(path.glob("net*.pt")):
            net = _make_net(obj.arch, len(obj.channels), obj.hidden, obj.dropout)
            net.load_state_dict(torch.load(file, map_location=obj.device))
            net.to(obj.device).eval()
            obj.nets.append(net)
        return obj
