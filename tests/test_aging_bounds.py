"""Поправка верхней границы температур на старение катализатора — за выключателем.

Диапазоны уставок сняты с обучающего периода, где катализатору было 14.4 месяца, а
он стареет на 0.85 °C/мес. На старшем катализаторе устаревшая граница запрещает
подъём температуры там, где он нужен (`docs/PLAN.md`). Тесты держат четыре вещи:
по умолчанию поправки нет вовсе; включённая — двигает только ВЕРХ и только
температуры; на катализаторе моложе опорного её нет; шаг цикла она не обходит.
"""
from __future__ import annotations

import copy
from datetime import datetime

import pandas as pd
import pytest

from nefte.agents.optimizer import OptimizerAgent
from nefte.config import load_config
from tests.test_agents import make_state

BOUNDS = {"T5": (350.0, 384.0), "T11": (350.0, 380.0), "P13": (3.5, 4.2), "F26": (200.0, 300.0)}
AGED = "2026-03-01"        # старый катализатор: замена была 23.04.2026
FRESH = "2026-06-01"       # после замены — моложе опорного


def _agent(enabled: bool, rate: float | None = None, cap: float | None = None) -> OptimizerAgent:
    cfg = copy.deepcopy(load_config())
    cfg["optimization"] = {**cfg["optimization"], "aging_bounds": enabled}
    agent = OptimizerAgent(bounds=dict(BOUNDS), surrogate=lambda state, moves: {}, cfg=cfg)
    if rate is not None:
        agent._aging_cache = (rate, cap)
    return agent


def _bounds(agent: OptimizerAgent, ts: str) -> dict:
    state = make_state()
    state.ts = datetime.fromisoformat(ts)
    state.telemetry_ht.update({"T5": 370.0, "T11": 366.0, "P13": 3.9, "F26": 250.0})
    from nefte.agents.schemas import ReliabilityAssessment
    reliability = ReliabilityAssessment(ts=state.ts, severity_index=0.4, risk_class="low")
    return agent._effective_bounds(state, reliability)


def test_switch_is_off_by_default_and_in_config():
    assert load_config()["optimization"]["aging_bounds"] is False
    assert _agent(False).aging_shift(pd.Timestamp(AGED)) == 0.0


def test_shift_grows_with_catalyst_age():
    """Восемь месяцев разницы возраста при 0.85 °C/мес — около семи градусов."""
    shift = _agent(True).aging_shift(pd.Timestamp(AGED))
    assert 5.0 < shift < 9.0


def test_no_shift_for_catalyst_younger_than_the_reference():
    assert _agent(True).aging_shift(pd.Timestamp(FRESH)) == 0.0


def test_shift_is_capped_by_the_cycle_span():
    """Устаревшая опора не должна уводить границу без предела."""
    assert _agent(True, rate=100.0, cap=25.3).aging_shift(pd.Timestamp(AGED)) == pytest.approx(25.3)


def test_shift_moves_only_the_upper_bound_of_temperatures():
    off, on = _bounds(_agent(False), AGED), _bounds(_agent(True), AGED)
    # шаг цикла по температуре меньше поправки, поэтому верх упирается в шаг;
    # сравниваем сам диапазон модели через тег, стоящий у верхней границы
    assert off["P13"] == on["P13"] and off["F26"] == on["F26"]
    assert on["T5"][0] == off["T5"][0], "нижняя граница не двигается"


def test_shift_does_not_bypass_the_step_limit():
    cfg = load_config()
    step = float(cfg["limits"]["max_step_per_cycle"]["temperature_c"])
    on = _bounds(_agent(True, rate=100.0, cap=25.3), AGED)
    assert on["T5"][1] <= 370.0 + step + 1e-9


def test_aged_catalyst_gets_more_room_at_the_top():
    """Тег у верхней границы: со старением у оптимизатора появляется ход вверх."""
    state = make_state()
    state.ts = datetime.fromisoformat(AGED)
    state.telemetry_ht.update({"T5": 383.5, "T11": 366.0, "P13": 3.9, "F26": 250.0})
    from nefte.agents.schemas import ReliabilityAssessment
    r = ReliabilityAssessment(ts=state.ts, severity_index=0.4, risk_class="low")
    off = OptimizerAgent(bounds=dict(BOUNDS), surrogate=lambda s, m: {},
                         cfg={**load_config(), "optimization": {
                             **load_config()["optimization"], "aging_bounds": False}})
    on = _agent(True)
    lo_off, hi_off = off._effective_bounds(state, r)["T5"]
    lo_on, hi_on = on._effective_bounds(state, r)["T5"]
    assert hi_off == pytest.approx(384.0)          # упёрся в устаревший p95
    assert hi_on > hi_off + 1.0                    # старение дало ход вверх
    assert lo_on == pytest.approx(lo_off)
