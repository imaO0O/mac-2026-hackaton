"""Синхронизация источников и признаки — без утечки из будущего.

Три правила, которые здесь зашиты (все три требует ТЗ):
1. Совмещение источников только по времени и только «назад» (``direction="backward"``):
   в момент t можно знать лишь то, что уже измерено.
2. Вместе со значением всегда возвращается его ВОЗРАСТ в часах — свежесть анализа
   часть решения, а не деталь реализации.
3. Разбиение train/val/test — только по времени, с эмбарго на границе.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nefte.config import load_config


def asof_features(index: pd.DatetimeIndex, series: pd.Series, name: str,
                  max_age_hours: float | None = None,
                  allow_exact_matches: bool = True) -> pd.DataFrame:
    """Последнее известное значение ``series`` на каждый момент ``index`` + его возраст.

    Parameters
    ----------
    allow_exact_matches : если False, значение, измеренное РОВНО в момент t, не
        используется. Это критично, когда ``series`` — тот же показатель, что и
        целевая переменная: иначе модель «предсказывает» анализ, зная его.

    Returns
    -------
    DataFrame с колонками ``{name}`` и ``{name}_age_h``. Если значение старше
    ``max_age_hours``, оно обнуляется в NaN, но возраст сохраняется — агент
    качества должен видеть, что данные устарели, а не думать, что их нет.
    """
    left = pd.DataFrame({"ts": pd.DatetimeIndex(index)}).sort_values("ts")
    right = (series.dropna().rename("value").reset_index()
             .rename(columns={series.index.name or "index": "ts"})
             .sort_values("ts"))
    right.columns = ["ts", "value"]

    merged = pd.merge_asof(left, right.assign(src_ts=right["ts"]), on="ts",
                           direction="backward",
                           allow_exact_matches=allow_exact_matches)
    # возраст = t - время последнего фактического измерения
    age_h = (merged["ts"] - merged["src_ts"]).dt.total_seconds() / 3600.0

    out = pd.DataFrame({name: merged["value"].to_numpy(),
                        f"{name}_age_h": age_h.to_numpy()},
                       index=pd.DatetimeIndex(merged["ts"]))
    if max_age_hours is not None:
        out.loc[out[f"{name}_age_h"] > max_age_hours, name] = np.nan
    return out.reindex(index)


def add_lags(df: pd.DataFrame, columns: list[str], lags_steps: list[int],
             prefix: str = "") -> pd.DataFrame:
    """Лаговые признаки. Шаг = шаг сетки (10 мин), 6 шагов = 1 час."""
    out = {f"{prefix}{c}_lag{l}": df[c].shift(l) for c in columns for l in lags_steps}
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def add_rollings(df: pd.DataFrame, columns: list[str], windows_steps: list[int],
                 stats: tuple[str, ...] = ("mean", "std")) -> pd.DataFrame:
    """Скользящие статистики — сглаживают шум КИП и дают «память» о режиме."""
    frames = [df]
    for w in windows_steps:
        roll = df[columns].rolling(w, min_periods=max(2, w // 3))
        for stat in stats:
            frames.append(getattr(roll, stat)().add_suffix(f"_{stat}{w}"))
    return pd.concat(frames, axis=1)


def wabt(temps: pd.DataFrame, weights: list[float] | None = None) -> pd.Series:
    """Средневзвешенная температура слоя (прокси жёсткости режима гидроочистки).

    Прямой разметки состояния катализатора в пакете нет, поэтому используем
    общепринятый прокси. Это ДОПУЩЕНИЕ, оно объявлено в docs/DATA_NOTES.md.
    """
    if weights is None:
        weights = [1.0 / temps.shape[1]] * temps.shape[1]
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    values = np.nansum(temps.to_numpy() * w, axis=1)
    return pd.Series(values, index=temps.index, name="wabt")


def time_split(index: pd.DatetimeIndex, cfg: dict | None = None) -> dict[str, pd.Series]:
    """Маски train/val/test по времени с эмбарго на границах.

    Случайное перемешивание запрещено ТЗ: любые лаги/скользящие окна протекли бы
    из будущего в прошлое.
    """
    cfg = cfg or load_config()
    sp = cfg["split"]
    emb = pd.Timedelta(hours=int(sp.get("embargo_hours", 0)))
    idx = pd.DatetimeIndex(index)

    def _mask(bounds, shrink_left: bool) -> pd.Series:
        lo, hi = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]) + pd.Timedelta(days=1)
        if shrink_left:
            lo = lo + emb
        return pd.Series((idx >= lo) & (idx < hi), index=idx)

    return {
        "train": _mask(sp["train"], False),
        "val": _mask(sp["val"], True),
        "test": _mask(sp["test"], True),
    }


def make_supervised(features: pd.DataFrame, target: pd.Series,
                    horizon_steps: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Выравнивает X и y с горизонтом прогноза.

    ``horizon_steps`` — на сколько шагов вперёд предсказываем (запаздывание
    отклика качества на изменение режима). y(t + h) объясняется X(t).
    """
    y = target.shift(-horizon_steps) if horizon_steps else target
    common = features.index.intersection(y.dropna().index)
    return features.loc[common], y.loc[common]
