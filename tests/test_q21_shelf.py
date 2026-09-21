"""Полка Q21: прибор в неисправности с дрожанием должен считаться замёрзшим.

Q21 — оперативный анализатор серы — уходит в неисправность на 24.88 ± 0.04 мг/кг
сотни часов подряд. Прежний детектор сравнивал значения на точное равенство и не
видел такую полку вовсе: 0 % её точек. Тесты держат три свойства нового: ловит
полку с дрожанием, не трогает живой прибор и не заглядывает вперёд.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nefte.config import load_config
from nefte.data.cleaning import flat_mask, frozen_mask

N = 18                       # frozen_min_samples: три часа на 10-минутной сетке


def _series(values) -> pd.Series:
    index = pd.date_range("2026-06-18", periods=len(values), freq="10min")
    return pd.Series(values, index=index, dtype=float)


def _shelf(n: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(24.88 + rng.uniform(-0.02, 0.02, n))


def test_jittery_shelf_is_caught_where_exact_equality_sees_nothing():
    shelf = _shelf()
    assert not frozen_mask(shelf, N).any()           # прежний детектор слеп
    caught = flat_mask(shelf, N, 0.05)
    assert caught.iloc[N:].all()                     # после N отсчётов — всё


def test_live_analyzer_is_not_switched_off():
    """Живой прибор гуляет на десятые доли и больше — его не трогаем."""
    rng = np.random.default_rng(1)
    live = _series(8.5 + np.cumsum(rng.normal(0.0, 0.15, 200)))
    assert flat_mask(live, N, 0.05).mean() < 0.01


def test_no_look_ahead_the_first_points_of_a_shelf_are_not_flagged():
    """Узнать, что прибор простоит ещё три часа, в первый момент нельзя."""
    rng = np.random.default_rng(2)
    before = 8.5 + np.cumsum(rng.normal(0.0, 0.2, 40))
    series = _series(np.concatenate([before, 24.88 + rng.uniform(-0.02, 0.02, 40)]))
    mask = flat_mask(series, N, 0.05)
    assert not mask.iloc[:40 + N - 1].any()
    assert mask.iloc[40 + N:].all()


def test_config_enables_the_tolerance_detector_for_q21():
    tolerance = load_config()["quality"]["q21_frozen_tolerance_mgkg"]
    assert tolerance is not None and 0 < float(tolerance) <= 0.2
