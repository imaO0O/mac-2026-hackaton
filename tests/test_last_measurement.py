"""Пустая проба — не измерение.

`StateBuilder._last` берёт последнее значение до момента среза. Если в ряду есть
строка без значения (NaN), без отсева она выглядела бы свежим анализом с возрастом
ноль: флаг устаревания молчал бы, а решения считались бы по несуществующему числу.
На выданных данных таких строк нет, поэтому числа не меняются — но правило должно
стоять в коде, а не держаться на том, что данные оказались чистыми.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.pipeline import StateBuilder

TS = pd.Timestamp("2026-03-01 12:00")


def _series(values: list[float | None]) -> pd.Series:
    index = pd.date_range("2026-02-28 12:00", periods=len(values), freq="12h", name="date")
    return pd.Series([np.nan if v is None else v for v in values], index=index, dtype=float)


def test_empty_last_sample_does_not_pass_as_a_fresh_one():
    """Последняя строка пустая: берём предыдущий настоящий анализ и его возраст."""
    value, age = StateBuilder._last(_series([7.5, 8.1, None]), TS)
    assert value == pytest.approx(8.1)
    assert age == pytest.approx(12.0)


def test_value_and_age_agree_when_the_last_sample_has_a_value():
    value, age = StateBuilder._last(_series([7.5, 8.1, 9.0]), TS)
    assert value == pytest.approx(9.0) and age == pytest.approx(0.0)


def test_all_samples_empty_means_no_measurement():
    assert StateBuilder._last(_series([None, None, None]), TS) == (None, None)


def test_lims_age_counts_from_sampling_for_the_last_real_analysis():
    """Возраст лабораторной пробы всё так же считается от отбора, а не от публикации."""
    builder = StateBuilder.__new__(StateBuilder)
    builder.lims_delay_hours = 4.0
    value, age = builder._last_lims(_series([7.5, 8.1, None]), TS)
    assert value == pytest.approx(8.1)
    assert age == pytest.approx(16.0)          # 12 ч до пробы + 4 ч публикации
