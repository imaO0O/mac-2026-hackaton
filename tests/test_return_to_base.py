# -*- coding: utf-8 -*-
"""Возврат к базовому режиму: откуда берётся сдвиг и когда возврат не предлагается.

Возврат реализован и измерен (scripts/check_return_to_base.py), но выключен по
умолчанию: на окне с остановом установки прибавилось превышение серы, а часть
возвратов отменялась качелями. Тесты держат обещания, которые важны при любом
значении флага:

* сдвигом считается только ПРИМЕНЁННОЕ изменение — выданная рекомендация, которую
  никто не применил (разомкнутый прогон по истории), не в счёт;
* выключенный возврат не предлагается никогда;
* качели в сводке имитации считаются так, как описаны.
"""
import pandas as pd

from nefte.agents.orchestrator import Orchestrator
from nefte.config import load_config
from nefte.sim import SimStep, _swings_after_return


class _Stub:
    t95_fn = None
    alarm_threshold = 0.18


def _orchestrator(enabled: bool | None = None) -> Orchestrator:
    cfg = load_config()
    if enabled is not None:
        cfg = {**cfg, "limits": {**cfg["limits"], "return_to_base": enabled}}
    return Orchestrator(_Stub(), _Stub(), _Stub(), cfg=cfg, log_runs=False)


def test_return_is_off_by_default():
    assert _orchestrator().return_to_base is False


def test_only_applied_changes_accumulate():
    orch = _orchestrator(True)
    assert orch._applied == {}
    orch.record_applied({"F26": -5.0, "T11": 0.5})
    orch.record_applied({"F26": -2.0})
    assert orch._applied == {"F26": -7.0, "T11": 0.5}
    orch.record_applied({"T11": -0.5})
    assert "T11" not in orch._applied, "полностью возвращённый тег не должен висеть нулём"


def test_disabled_or_unapplied_never_returns():
    off = _orchestrator(False)
    off.record_applied({"F26": -10.0})
    assert off._return_to_base(None, None, None, object(), []) is None
    on = _orchestrator(True)
    assert on._return_to_base(None, None, None, object(), []) is None, \
        "без применённых сдвигов возвращать нечего"


def _step(hours: float, kind: str = "", moved: dict | None = None) -> SimStep:
    return SimStep(ts=pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hours),
                   outcome="меняем уставки" if moved else "держим режим",
                   sulfur_sim=5.0, sulfur_hist=None, offsets={}, moved=moved or {},
                   kind=kind)


def test_swing_is_a_return_undone_within_a_day():
    steps = [_step(0, moved={"F26": -5.0}),
             _step(20, "возврат", {"F26": +5.0}),
             _step(30, moved={"F26": -3.0}),          # отменили возврат через 10 ч
             _step(60, "возврат", {"F26": +3.0}),
             _step(90, moved={"F26": -2.0})]          # через 30 ч — уже не качели
    assert _swings_after_return(steps) == 1
