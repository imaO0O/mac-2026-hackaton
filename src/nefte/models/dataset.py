"""Датасет агента качества: признаки на сетке 10 минут и обучающая таблица.

Целевая переменная — ЛАБОРАТОРНОЕ значение серы в товарном ДТ
(``Гидроочистка|2|Mg.Sulfur``): по ТЗ лабораторный результат считается контрольным
фактом, поточный анализатор — лишь оперативная оценка. Поэтому обученная модель
это, по сути, виртуальный анализатор, который приводит ПАК и телеметрию
к лабораторной шкале.

Защита от утечки:
* все признаки в момент t построены только по данным ≤ t;
* предыдущее лабораторное значение берётся строго ДО t (``allow_exact_matches=False``),
  иначе модель «предсказывала» бы анализ, зная его;
* горизонт прогноза сдвигает признаки назад, а не цель вперёд;
* разбиение train/val/test — по времени, с эмбарго.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nefte.config import cache_dir, load_config
from nefte.data.cleaning import clean_lims_sulfur, clean_telemetry, frozen_mask
from nefte.data.features import asof_features
from nefte.data.loaders import lims_series, load_lims, load_pak, load_telemetry

# Теги гидроочистки — берём все: именно там формируется сера продукта.
HT_TAGS: list[str] | None = None

# Теги АВТ: сырьё и режим колонн, влияющие на состав сырья гидроочистки.
# Отобраны по схемам и справочнику КИП (см. docs/DATA_NOTES.md §4).
AVT_TAGS = ["T55", "T1", "T20", "T33", "T48", "T66",
            "P22", "P67", "P44",
            "F7", "F8", "F9", "F30", "F31", "F32", "F57", "W70"]

# Окна скользящих статистик в отсчётах по 10 минут: 1 ч, 6 ч, 24 ч.
ROLL_WINDOWS = [6, 36, 144]

TARGET_SERIES = "Гидроочистка|2|Mg.Sulfur"
FEED_SULFUR_SERIES = "Гидроочистка|1|Mass.Sulfur"


def build_feature_matrix(freq: str = "1h", use_cache: bool = True) -> pd.DataFrame:
    """Матрица признаков на регулярной сетке.

    ``freq="1h"`` (по умолчанию) прореживает 10-минутную сетку до часовой уже ПОСЛЕ
    расчёта скользящих окон: качество меняется медленно, а матрица становится в
    шесть раз легче и спокойно обучается на CPU. Метка бакета — правая граница,
    иначе строка содержала бы данные из собственного будущего.
    """
    cache = cache_dir() / f"features_{freq}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    cfg = load_config()
    avt, _ = clean_telemetry(load_telemetry("avt", AVT_TAGS), unit="avt")
    ht, _ = clean_telemetry(load_telemetry("ht", HT_TAGS), unit="ht")

    avt = avt.add_prefix("avt_")
    ht = ht.add_prefix("ht_")
    tel = pd.concat([avt, ht], axis=1).astype("float32")

    frames = [tel]
    for w in ROLL_WINDOWS:
        roll = tel.rolling(w, min_periods=max(2, w // 3))
        frames.append(roll.mean().add_suffix(f"_mean{w}").astype("float32"))
        frames.append(roll.std().add_suffix(f"_std{w}").astype("float32"))
    # изменение режима за 6 ч — скорость, а не уровень
    frames.append((tel - tel.shift(36)).add_suffix("_d6h").astype("float32"))
    feats = pd.concat(frames, axis=1)

    # --- поточный анализатор серы: значение, динамика, достоверность ---
    pak = load_pak()["sulfur_ppm"]
    pak_frozen = frozen_mask(pak, int(cfg["telemetry"]["frozen_min_samples"]))
    pak_df = pd.DataFrame({
        "pak_sulfur": pak,
        "pak_sulfur_mean6": pak.rolling(36, min_periods=12).mean(),
        "pak_sulfur_mean24": pak.rolling(144, min_periods=48).mean(),
        "pak_sulfur_std6": pak.rolling(36, min_periods=12).std(),
        "pak_sulfur_d6h": pak - pak.shift(36),
        "pak_frozen": pak_frozen.astype("float32"),
    })
    feats = feats.join(pak_df.reindex(feats.index).astype("float32"))

    # --- лабораторные ряды как признаки: значение + возраст --------------
    lims = load_lims()
    target = clean_lims_sulfur(lims_series(TARGET_SERIES, lims))
    feed = lims_series(FEED_SULFUR_SERIES, lims)

    # предыдущий анализ продукта: строго ДО момента t
    prev = asof_features(feats.index, target, "lims_sulfur_prev",
                         allow_exact_matches=False)
    # сера сырья: контекст нагрузки на катализатор
    feed_f = asof_features(feats.index, feed, "lims_feed_sulfur")
    feats = pd.concat([feats, prev, feed_f], axis=1)

    if freq:
        # Метка бакета — ПРАВАЯ граница: строка с меткой t содержит данные (t-freq, t].
        # С меткой по левой границе строка «14:00» тянула бы данные до 14:50, то есть
        # признаки заглядывали бы в будущее относительно собственной метки времени.
        feats = feats.resample(freq, label="right", closed="right").last()

    feats.to_parquet(cache)
    return feats


def build_training_table(horizon_hours: float = 2.0,
                         features: pd.DataFrame | None = None
                         ) -> tuple[pd.DataFrame, pd.Series]:
    """Обучающая таблица: X — признаки за ``horizon_hours`` ДО анализа, y — анализ.

    Так модель отвечает на вопрос «каким будет лабораторный результат через
    H часов», а не «каким он был». Это и есть прогноз качества из ТЗ.
    """
    feats = build_feature_matrix() if features is None else features
    y = clean_lims_sulfur(lims_series(TARGET_SERIES))

    # для каждого анализа берём признаки, доступные за H часов до него
    lag = pd.Timedelta(hours=horizon_hours)
    rows, index = [], []
    fidx = feats.index
    positions = fidx.searchsorted(y.index - lag, side="right") - 1
    for pos, ts in zip(positions, y.index):
        if pos < 0:
            continue
        rows.append(fidx[pos])
        index.append(ts)

    X = feats.loc[rows].copy()
    X.index = pd.DatetimeIndex(index)          # индексируем моментом анализа
    X["feature_age_h"] = [(ts - r).total_seconds() / 3600 for ts, r in zip(index, rows)]
    y = y.loc[X.index]

    keep = X.notna().mean() > 0.5              # выкидываем почти пустые признаки
    X = X.loc[:, keep]
    return X, y


def persistence_baselines(X: pd.DataFrame, y: pd.Series) -> dict[str, pd.Series]:
    """Базовые «модели», которые обязана побить обученная.

    * ``pak`` — текущее показание поточного анализатора (то, чем пользуются сейчас);
    * ``lims_prev`` — предыдущий лабораторный результат;
    * ``const`` — медиана обучающего периода.
    """
    out = {}
    if "pak_sulfur" in X:
        out["pak"] = X["pak_sulfur"]
    if "lims_sulfur_prev" in X:
        out["lims_prev"] = X["lims_sulfur_prev"]
    out["const"] = pd.Series(np.full(len(y), np.nan), index=y.index)
    return out
