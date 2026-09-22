"""Пороги правила о полках стоят там, где их записали ДО счёта.

Правило (`docs/PLAN.md`, коммит a113c1b): неисправность — шум тега упал ниже 5 %
своего обычного И активность процесса не ниже 50 %; ровный режим — активность ниже
20 %; между — неопределённо. Три пометки (шкала, нагрузка, синхронность) найдены уже
после счёта и добавлены РЯДОМ с вердиктом. Тесты держат ровно это: пороги не
сдвинулись, а пометки вердикта не трогают — иначе правило оказалось бы подогнанным
под то, что нашлось в данных (`docs/SHELVES.md`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.check_shelves import (
    ACTIVE,
    LOW_LOAD,
    NOISE_DEAD,
    QUIET,
    load_flag,
    scale_flag,
    verdict,
)


def _series(values) -> pd.Series:
    index = pd.date_range("2024-04-20", periods=len(values), freq="10min")
    return pd.Series(values, index=index, dtype=float)


def test_thresholds_are_the_ones_written_before_counting():
    assert (NOISE_DEAD, ACTIVE, QUIET) == (0.05, 0.50, 0.20)


def test_dead_instrument_on_a_living_unit():
    assert verdict(0.01, 1.0) == "неисправность"


def test_quiet_process_is_not_an_instrument_fault():
    assert verdict(0.01, 0.1) == "ровный режим"


def test_middle_band_stays_undecided():
    """Между 20 и 50 % активности правило молчит — такие полки никуда не идут."""
    assert verdict(0.01, 0.3) == "неопределённо"
    assert verdict(0.5, 1.0) == "неопределённо"


def test_missing_measurements_are_not_a_verdict():
    assert verdict(float("nan"), 1.0) == "нет данных"


def test_scale_flag_sees_zero_reading_that_the_noise_rule_misses():
    """Случай `ht:P3`: прибор дрожит, но на нуле при рабочих 3.66 МПа."""
    rng = np.random.default_rng(0)
    live = _series(np.r_[3.66 + rng.normal(0, 0.05, 400), 0.0023 + rng.normal(0, 0.0005, 200)])
    assert scale_flag(live, 0.0023, 0.01).startswith("на нуле")
    assert verdict(1.32, 1.10) == "неопределённо"      # вердикт правила при этом прежний


def test_scale_flag_sees_the_top_of_the_range():
    """Случай `ht:Q20`: упор в потолок диапазона."""
    rng = np.random.default_rng(1)
    live = _series(np.r_[7914 + rng.normal(0, 800, 400), np.full(50, 15040.0)])
    assert scale_flag(live, 15040.0, 5.0).startswith("на верхней границе")


def test_scale_flag_is_silent_in_the_middle_of_the_range():
    rng = np.random.default_rng(2)
    live = _series(3.66 + rng.normal(0, 0.05, 400))
    assert scale_flag(live, 3.66, 0.01) is None


def test_load_flag_marks_the_startup_hours():
    """19.04.2024: полка у 20 тегов сразу, а установка шла на 160 т/ч вместо 256."""
    feed = _series(np.r_[np.full(300, 256.0), np.full(100, 160.0)])
    lo, hi = feed.index[300], feed.index[-1]
    assert load_flag(feed, 256.0, lo, hi) < LOW_LOAD          # 0.62 обычной — пуск
    assert load_flag(feed, 256.0, feed.index[0], feed.index[299]) == pytest.approx(1.0)


def test_load_flag_has_no_opinion_without_data():
    assert load_flag(_series([np.nan] * 10), 256.0,
                     pd.Timestamp("2024-04-20"), pd.Timestamp("2024-04-21")) is None
