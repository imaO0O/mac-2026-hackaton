"""Тесты отбора эпизодов «режим менялся не в ответ на серу».

Главное, что держат: знак реакции у каждой переменной свой, ступенька против тренда
серы не может считаться реакцией, а пуски и смены катализатора в ступеньки режима
не попадают — там отклик смешан с другим.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nefte.data.episodes import (
    NoiseBand,
    classify,
    detect_steps,
    regime_step_episodes,
    step_series,
    sulfur_trend,
)

BAND = NoiseBand(flat=0.3, clear=0.6)


def test_rising_sulfur_then_hotter_reactor_is_a_reaction():
    assert classify("wabt", +2.5, +1.0, 8.0, 8.0, 10.0, BAND) == "похоже на реакцию"


def test_rising_sulfur_then_lower_feed_is_a_reaction_too():
    """У нагрузки знак реакции обратный: на рост серы её снижают."""
    assert classify("feed", -20.0, +1.0, 8.0, 8.0, 10.0, BAND) == "похоже на реакцию"
    assert classify("feed", +20.0, +1.0, 8.0, 8.0, 10.0, BAND) == "против тренда серы"


def test_cooling_while_sulfur_rises_cannot_be_a_reaction():
    assert classify("wabt", -2.5, +1.0, 8.0, 8.0, 10.0, BAND) == "против тренда серы"


def test_flat_and_undetermined_bands_are_named_separately():
    assert classify("wabt", +2.5, 0.2, 8.0, 8.0, 10.0, BAND) == "сера ровная"
    assert classify("wabt", +2.5, 0.5, 8.0, 8.0, 10.0, BAND) == "неопределённо"


def test_sulfur_above_spec_is_never_called_external():
    """Выше предела вмешиваются при любом наклоне — ровная сера тут не довод."""
    assert classify("wabt", +2.5, 0.0, 12.0, 8.0, 10.0, BAND) == "сера выше предела"
    assert classify("wabt", +2.5, 0.0, 8.0, 11.0, 10.0, BAND) == "сера выше предела"


def test_missing_sulfur_context_is_not_guessed():
    assert classify("wabt", +2.5, None, None, 8.0, 10.0, BAND) == "нет данных о сере"


def test_sulfur_trend_has_the_sign_of_the_ramp():
    index = pd.date_range("2025-01-01", periods=48, freq="h")
    ramp = pd.Series(np.linspace(5.0, 9.0, 48), index=index)
    assert sulfur_trend(ramp).dropna().gt(0).all()


def _hourly(steps_at: dict[str, tuple[int, float]], hours: int = 24 * 20) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=hours, freq="h")
    frame = pd.DataFrame({"wabt": 360.0, "feed": 250.0, "pressure": 3.9}, index=index)
    for column, (hour, size) in steps_at.items():
        frame.iloc[hour:, frame.columns.get_loc(column)] += size
    return frame


def test_one_step_is_counted_once():
    hourly = _hourly({"wabt": (200, 3.0)})
    found = detect_steps(step_series(hourly)["wabt"], 2.0)
    assert len(found) == 1
    assert abs((found.index[0] - hourly.index[200]).total_seconds()) <= 3600


def _context(hourly: pd.DataFrame, trend_up: bool = False):
    pak = pd.Series(8.0, index=hourly.index)
    if trend_up:
        pak = pd.Series(np.linspace(6.0, 9.0, len(hourly)), index=hourly.index)
    pak = pak + np.random.default_rng(0).normal(0, 0.3, len(pak))
    published = pd.Series(8.0, index=hourly.index[::24])
    return pak, published


def test_steps_inside_startup_and_catalyst_windows_are_left_out():
    hourly = _hourly({"wabt": (100, 3.0), "feed": (300, 25.0)})
    pak, published = _context(hourly)
    outage_end = hourly.index[90]          # ступенька температуры — в окне пуска
    change = hourly.index[296]              # ступенька нагрузки — у смены катализатора
    found = regime_step_episodes(hourly, pak, published, 10.0, [outage_end], [change])
    assert found.empty


def test_concurrent_change_disqualifies_a_clean_episode():
    """Температуру подняли вместе с нагрузкой: отклик на одну не выделить."""
    hourly = _hourly({"wabt": (200, 3.0), "feed": (200, 25.0)})
    pak, published = _context(hourly)
    found = regime_step_episodes(hourly, pak, published, 10.0, [], [])
    assert not found.empty
    assert not found["чистый"].any()
    assert all(found["одновременно заметно"].map(len) > 0)


def test_concurrent_change_below_threshold_still_spoils_the_episode():
    """Нагрузку сдвинули на 10 м3/ч при пороге 13 — ступенькой она не считается.

    Но для оценки отклика на температуру этого достаточно, чтобы всё испортить:
    4 % нагрузки меняют серу сравнимо с несколькими градусами.
    """
    hourly = _hourly({"wabt": (200, 3.0), "feed": (200, 10.0)})
    pak, published = _context(hourly)
    found = regime_step_episodes(hourly, pak, published, 10.0, [], [])
    wabt = found[found["переменная"] == "wabt"]
    assert len(wabt) == 1
    assert wabt["одновременно заметно"].iloc[0] == ["feed"]
    assert not wabt["чистый"].iloc[0]


def test_small_concurrent_wiggle_does_not_spoil_the_episode():
    hourly = _hourly({"wabt": (200, 3.0), "feed": (200, 3.0)})
    pak, published = _context(hourly)
    found = regime_step_episodes(hourly, pak, published, 10.0, [], [])
    wabt = found[found["переменная"] == "wabt"]
    assert len(wabt) == 1
    assert wabt["одновременно заметно"].iloc[0] == []
