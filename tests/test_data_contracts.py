"""Тесты, которые ловят самые дорогие ошибки этого хакатона.

Они проверяют не «что код запускается», а те свойства данных и логики, из-за
нарушения которых решение проигрывает по критериям ТЗ: утечка из будущего,
потеря возраста анализа, подмена брака на физическое значение.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.data.cleaning import SENTINELS, frozen_intervals, frozen_mask, mask_sentinels
from nefte.data.features import asof_features, time_split


def _series(values, start="2024-01-01", freq="10min"):
    idx = pd.date_range(start, periods=len(values), freq=freq, name="ts")
    return pd.Series(values, index=idx)


def test_sentinel_masking_removes_307():
    df = pd.DataFrame({"T1": [100.0, 307.0, 102.0], "F2": [1.0, 2.0, 313.0]})
    out = mask_sentinels(df, SENTINELS)
    assert out["T1"].isna().sum() == 1
    assert out["F2"].isna().sum() == 1
    assert out["T1"].iloc[0] == 100.0


def test_frozen_mask_detects_stuck_analyzer():
    """Маска включается, когда порог УЖЕ набран по прошлому.

    Здесь раньше стояло `m.iloc[:20].all()` — то есть маска считалась истинной с
    первого же отсчёта полки. Утверждение выглядело разумным («вся полка
    заморожена»), но означало заглядывание вперёд: в первый момент видно одно
    одинаковое значение, и узнать, что сигнал простоит ещё три часа, нельзя.
    """
    s = _series([5.0] * 20 + [5.1, 5.2, 5.3])
    m = frozen_mask(s, min_samples=18)
    assert not m.iloc[:17].any(), "порог ещё не набран — знать неоткуда"
    assert m.iloc[17:20].all(), "с 18-го одинакового отсчёта полка видна"
    assert not m.iloc[20:].any()

    # А вот ОТЧЁТ об эпизодах смотрит на прогон целиком, и это правильно:
    # он ретроспективный и ни в какие признаки не идёт.
    intervals = frozen_intervals(s, min_samples=18)
    assert len(intervals) == 1
    assert intervals.loc[0, "value"] == 5.0
    assert intervals.loc[0, "n"] == 20


def test_asof_never_looks_into_future():
    lab = _series([1.0, 2.0], start="2024-01-01 00:00", freq="6h")
    idx = pd.date_range("2024-01-01 00:00", periods=13, freq="1h", name="ts")
    out = asof_features(idx, lab, "lab")

    # до второго анализа (06:00) должно быть только первое значение
    assert (out.loc[:"2024-01-01 05:00", "lab"] == 1.0).all()
    assert out.loc["2024-01-01 06:00", "lab"] == 2.0


def test_asof_reports_age_and_hides_stale_value():
    lab = _series([7.0], start="2024-01-01 00:00", freq="1h")
    idx = pd.date_range("2024-01-01 00:00", periods=49, freq="1h", name="ts")

    out = asof_features(idx, lab, "lab")
    assert out.loc["2024-01-02 00:00", "lab_age_h"] == pytest.approx(24.0)

    fresh = asof_features(idx, lab, "lab", max_age_hours=12)
    assert np.isnan(fresh.loc["2024-01-02 00:00", "lab"])
    assert fresh.loc["2024-01-02 00:00", "lab_age_h"] == pytest.approx(24.0)


def test_time_split_is_ordered_and_disjoint():
    idx = pd.date_range("2023-01-01", "2026-08-07", freq="1D", name="ts")
    masks = time_split(idx)

    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["val"] & masks["test"]).any()
    assert idx[masks["train"].to_numpy()].max() < idx[masks["val"].to_numpy()].min()
    assert idx[masks["val"].to_numpy()].max() < idx[masks["test"].to_numpy()].min()
