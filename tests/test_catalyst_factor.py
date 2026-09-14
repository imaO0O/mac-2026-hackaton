"""Выключатели износа катализатора в агенте надёжности.

Держат три вещи. По умолчанию поведение прежнее — выключатель не имеет права
незаметно поменять severity. Журнал замен сбрасывает наработку только на замене, а
не на ремонте. И фактор активности считается строго по прошлому и ведёт себя так,
как обещает: падает после замены катализатора и не падает после ремонта.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from nefte.agents.reliability import ReliabilityAgent
from nefte.config import load_config
from nefte.models.catalyst import activity_level_series, hours_since_marks
from nefte.models.regime import hours_since_outage

TRAIN = ("2024-01-01", "2024-12-31")


def _history(days: int = 330, outages: list[tuple[int, int, float]] | None = None,
             seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Синтетическая история: (сутки начала, длительность в сутках, сдвиг WABT после).

    Сдвиг WABT после останова моделирует смену катализатора (свежий держит ту же
    серу холоднее) или ремонт без смены (уровень прежний).
    """
    index = pd.date_range("2024-01-01", periods=days * 144, freq="10min", name="date")
    rng = np.random.default_rng(seed)
    wabt = 360.0 + np.linspace(0.0, 8.0, len(index)) + rng.normal(0, 1.0, len(index))
    feed = 250.0 + rng.normal(0, 5.0, len(index))
    for start_day, length_days, shift in outages or []:
        a, b = start_day * 144, (start_day + length_days) * 144
        feed[a:b] = 1.0
        wabt[a:b] = 40.0
        wabt[b:] += shift
    ht = pd.DataFrame({"T5": wabt - 1.0, "T6": wabt, "T11": wabt + 1.0,
                       "W10": 1.2 + rng.normal(0, 0.02, len(index)), "F26": feed,
                       "F2": 30000.0 + rng.normal(0, 300, len(index)),
                       "P13": 3.9 + rng.normal(0, 0.01, len(index))}, index=index)
    avt = pd.DataFrame({"T55": 380.0 + rng.normal(0, 1.0, len(index))}, index=index)
    return avt, ht


def _cfg(**reliability) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["split"] = {**cfg["split"], "train": list(TRAIN)}
    cfg["reliability"] = {**(cfg.get("reliability") or {}), **reliability}
    return cfg


def test_default_switches_keep_the_old_age_factor():
    avt, ht = _history(outages=[(120, 3, 0.0)])
    agent = ReliabilityAgent.from_history(avt, ht, _cfg(catalyst_factor="age",
                                                        catalyst_reset="outage_48h"))
    assert agent.catalyst_series is None
    expected = hours_since_outage(ht["F26"], min_outage_hours=48.0, steps_per_hour=6)
    assert agent.run_hours.equals(expected)


def test_unknown_switch_value_is_an_error_not_a_silent_default():
    avt, ht = _history(days=200)
    with pytest.raises(ValueError):
        ReliabilityAgent.from_history(avt, ht, _cfg(catalyst_factor="возраст"))


def test_log_reset_ignores_a_long_repair_and_resets_on_a_listed_change():
    index = pd.date_range("2024-01-01", periods=24 * 100, freq="h")
    marks = ["2024-02-15 12:00"]
    hours = hours_since_marks(index, marks)
    assert hours.loc["2024-02-15 11:00"] > 1000          # до замены — от начала данных
    assert hours.loc["2024-02-15 12:00"] == pytest.approx(0.0)
    assert hours.loc["2024-02-16 12:00"] == pytest.approx(24.0)


def test_activity_factor_uses_only_the_past():
    """Значение на момент t не меняется, если отрезать всё после t.

    Шкала берётся с обучающего периода, поэтому момент — после его конца, как в
    настоящем прогоне по валидации и тесту.
    """
    _, ht = _history(outages=[(150, 3, -20.0)])
    train = ("2024-01-01", "2024-06-30")
    full = activity_level_series(ht, ht["F26"], train)
    cut = pd.Timestamp("2024-09-01 12:00")
    past = activity_level_series(ht.loc[:cut], ht["F26"].loc[:cut], train)
    assert full is not None and past is not None
    assert full.loc[cut] == pytest.approx(past.loc[cut])


def _around(series: pd.Series, start_day: int, length_days: int) -> tuple[float, float]:
    origin = series.index[0]
    stop = origin + pd.Timedelta(days=start_day)
    restart = origin + pd.Timedelta(days=start_day + length_days)
    before = series.loc[stop - pd.Timedelta(days=30):stop].median()
    after = series.loc[restart + pd.Timedelta(days=30):restart + pd.Timedelta(days=40)].median()
    return float(before), float(after)


def test_activity_factor_drops_after_a_catalyst_change_but_not_after_a_repair():
    change = _history(days=330, outages=[(200, 3, -20.0)], seed=1)[1]
    repair = _history(days=330, outages=[(200, 3, 0.0)], seed=1)[1]
    before, after = _around(activity_level_series(change, change["F26"], TRAIN), 200, 3)
    assert after <= before - 0.2
    before, after = _around(activity_level_series(repair, repair["F26"], TRAIN), 200, 3)
    assert after > before - 0.1


def test_activity_switch_puts_the_series_into_severity():
    avt, ht = _history(outages=[(200, 3, -20.0)], seed=2)
    agent = ReliabilityAgent.from_history(avt, ht, _cfg(catalyst_factor="activity"))
    assert agent.catalyst_series is not None
    frame = pd.DataFrame({"wabt": ht[["T5", "T6", "T11"]].mean(axis=1), "W10": ht["W10"],
                          "T55": avt["T55"]})
    _, factors = agent.severity_series(frame, with_factors=True)
    joined = pd.concat([factors["catalyst"], agent.catalyst_series.clip(0, 1.5)],
                       axis=1).dropna()
    assert len(joined) > 1000
    assert np.allclose(joined.iloc[:, 0], joined.iloc[:, 1])
