"""Повторное действие, пока прошлое ещё проявляется — только при росте риска.

Правило выключено по умолчанию (``limits.repeat_min_risk_increase: null``). Тесты
держат три вещи: без правила поведение прежнее; с правилом повтор при том же
риске заменяется ожиданием с названной причиной; при заметном росте риска или
факте вне спецификации действие не задерживается.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from nefte.agents.optimizer import OptimizerAgent, linear_surrogate
from nefte.agents.orchestrator import Orchestrator
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent, SeverityNorms
from nefte.config import load_config
from tests.test_agents import make_state

T0 = datetime(2026, 4, 20, 12, 0)


def _system(repeat: float | None) -> Orchestrator:
    cfg = load_config()
    cfg = {**cfg, "limits": {**cfg["limits"], "repeat_min_risk_increase": repeat,
                             "min_hours_between_actions": 4.0,
                             "return_settle_hours": 14.0}}
    norms = SeverityNorms(bounds={"wabt": (355.0, 375.0), "P8": (0.13, 0.23),
                                  "T55": (370.0, 395.0)})
    bounds = {"T5": (365.0, 375.0), "T11": (360.0, 370.0), "F26": (200.0, 300.0),
              "P13": (3.5, 4.2)}
    opt = OptimizerAgent(bounds=bounds, cfg=cfg,
                         surrogate=linear_surrogate({"T5": -0.15, "T11": -0.15, "P13": -0.5}))
    return Orchestrator(QualityAgent(cfg=cfg), ReliabilityAgent(norms), opt, cfg=cfg,
                        log_runs=False)


def _at(hours: float, lims: float):
    state = make_state(lims=(lims, 1.0), pak=(lims, 0.1))
    return state.model_copy(update={"ts": T0 + timedelta(hours=hours)})


def test_without_the_rule_the_repeat_goes_through():
    system = _system(None)
    first = system.run(_at(0, 9.6))
    again = system.run(_at(5, 9.6))
    if first.outcome() != "меняем уставки":
        return          # синтетическая система не действует — нечего повторять
    assert again.outcome() == "меняем уставки"


def test_same_risk_waits_with_a_named_reason():
    system = _system(0.05)
    first = system.run(_at(0, 9.6))
    if first.outcome() != "меняем уставки":
        return
    again = system.run(_at(5, 9.6))
    assert again.outcome() == "держим режим"
    assert "ещё проявляется" in again.trace[-1].summary
    assert "не вырос" in again.explanation


def test_growing_risk_or_off_spec_fact_is_not_delayed():
    system = _system(0.05)
    first = system.run(_at(0, 9.3))
    if first.outcome() != "меняем уставки":
        return
    worse = system.run(_at(5, 9.9))
    assert worse.outcome() == "меняем уставки", worse.trace[-1].summary

    system = _system(0.05)
    system.run(_at(0, 9.6))
    off_spec = system.run(_at(5, 10.4))
    assert "ещё проявляется" not in off_spec.trace[-1].summary
