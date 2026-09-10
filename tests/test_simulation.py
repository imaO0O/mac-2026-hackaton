"""Тесты имитационной среды и сравнения архитектур.

Оба модуля появились ради двух пунктов ТЗ — имитационная среда и сравнение
архитектурных подходов — и оба сразу нашли дефекты в самой системе. Значит, их
собственную механику надо стеречь тестами: если сломается счётчик воздействий или
накопление уставок, «замкнутый контур устойчив» превратится в ничем не
подкреплённое утверждение.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source
from nefte.sim import MAX_DRIFT, ClosedLoopSimulator, SimStep, summarize


class _Builder:
    """Минимальный StateBuilder: постоянный режим, чтобы видеть только наши действия."""

    def __init__(self, sulfur: float = 9.0):
        self.sulfur = sulfur

    def build(self, ts) -> ProcessState:
        return ProcessState(
            ts=pd.Timestamp(ts).to_pydatetime(),
            telemetry_avt={"T55": 380.0},
            telemetry_ht={"T5": 360.0, "T6": 358.0, "T11": 362.0,
                          "F26": 250.0, "P13": 3.9, "W10": 2.5},
            quality={"pak_sulfur_ppm": Measurement(
                value=self.sulfur, unit="мг/кг", source=Source.PAK, age_hours=0.1)},
            data_quality=DataQuality(missing_share=0.0, usable=True),
        )


class _System:
    """Оркестратор-заглушка: всегда просит одно и то же изменение."""

    def __init__(self, delta: dict[str, float]):
        self.delta = delta
        self.calls = 0
        self._last_action_ts = None

    def run(self, state):
        from nefte.agents.schemas import Candidate, Recommendation

        self.calls += 1
        moves = {t: float(state.telemetry_ht.get(t, 0.0) + d)
                 for t, d in self.delta.items()}
        action = Candidate(id="test", moves=moves, deltas=dict(self.delta),
                           predicted_quality={"product_sulfur_mgkg": 8.0})
        return Recommendation(ts=state.ts, problem="тест", action=action,
                              confidence=0.7)


def _surrogate(level: float = 8.0):
    def _fn(_state, _moves):
        return {"product_sulfur_mgkg": level}
    return _fn


# --------------------------------------------------------------------------- #
# накопление уставок
# --------------------------------------------------------------------------- #

def test_offsets_accumulate_and_reach_the_state():
    """Смысл замкнутого контура: следующий шаг видит последствия предыдущего."""
    sb, system = _Builder(), _System({"T5": 1.0})
    sim = ClosedLoopSimulator(sb, system, _surrogate())
    stamps = pd.date_range("2026-01-01", periods=3, freq="4h")
    sim.run(stamps)

    assert sim.offsets["T5"] == pytest.approx(3.0)
    shifted = sim._apply_offsets(sb.build(stamps[-1]))
    assert shifted.telemetry_ht["T5"] == pytest.approx(363.0)


def test_drift_is_capped():
    """Без потолка ошибка в кинетике увела бы режим куда угодно."""
    sb, system = _Builder(), _System({"T5": 5.0})
    sim = ClosedLoopSimulator(sb, system, _surrogate())
    sim.run(pd.date_range("2026-01-01", periods=10, freq="4h"))
    assert sim.offsets["T5"] == pytest.approx(MAX_DRIFT["T5"])


def test_simulated_quality_replaces_the_measurement():
    """Иначе система читала бы анализ, снятый при ИСТОРИЧЕСКОМ режиме."""
    sb = _Builder(sulfur=9.0)
    sim = ClosedLoopSimulator(sb, _System({"T5": 0.5}), _surrogate(level=4.0))
    state = sim._with_simulated_quality(sb.build("2026-01-01"), 4.0)
    assert state.quality["pak_sulfur_ppm"].value == pytest.approx(4.0)
    assert "имитационной среды" in state.quality["pak_sulfur_ppm"].comment


def test_response_is_gradual_not_instant():
    """Качество идёт к новому уровню с постоянной времени, а не прыжком."""
    sb = _Builder()
    sim = ClosedLoopSimulator(sb, _System({}), _surrogate(level=4.0), tau_hours=4.0)
    steps = sim.run(pd.date_range("2026-01-01", periods=2, freq="1h"))
    # первый шаг задаёт уровень, второй сдвигается к цели лишь частично
    assert steps[0].sulfur_sim == pytest.approx(4.0)
    assert steps[1].sulfur_sim == pytest.approx(4.0, abs=1e-6)

    sim2 = ClosedLoopSimulator(sb, _System({}), _surrogate(level=4.0), tau_hours=4.0)
    sim2.sulfur = 9.0
    steps2 = sim2.run(pd.date_range("2026-01-01", periods=2, freq="1h"))
    assert 4.0 < steps2[1].sulfur_sim < 9.0


# --------------------------------------------------------------------------- #
# сводка
# --------------------------------------------------------------------------- #

def test_single_move_is_zero_direction_changes():
    """Было: счётчик вычитал единицу и выдавал «−1 смена направления»."""
    steps = [SimStep(ts=pd.Timestamp("2026-01-01"), outcome="меняем уставки",
                     sulfur_sim=8.0, sulfur_hist=None, offsets={"T5": 1.0},
                     moved={"T5": 1.0})]
    report = summarize(steps, limit=10.0)
    assert report["уставки"]["T5"]["смен направления"] == 0


def test_direction_changes_are_counted():
    """Туда-обратно-туда — две смены направления."""
    moves = [{"T5": 1.0}, {"T5": -1.0}, {"T5": 1.0}]
    steps = [SimStep(ts=pd.Timestamp("2026-01-01") + pd.Timedelta(hours=4 * i),
                     outcome="меняем уставки", sulfur_sim=8.0, sulfur_hist=None,
                     offsets={"T5": 0.0}, moved=m) for i, m in enumerate(moves)]
    report = summarize(steps, limit=10.0)
    assert report["уставки"]["T5"]["смен направления"] == 2
    assert report["уставки"]["T5"]["суммарно, ед."] == pytest.approx(3.0)


def test_summary_counts_time_above_limit():
    steps = [SimStep(ts=pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                     outcome="держим режим", sulfur_sim=value, sulfur_hist=None,
                     offsets={}) for i, value in enumerate([8.0, 12.0, 9.0, 11.0])]
    report = summarize(steps, limit=10.0)
    assert report["сера_сим"]["доля выше предела"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# сравнение архитектур
# --------------------------------------------------------------------------- #

def test_architecture_metrics_count_misses_and_effort():
    from scripts.compare_architectures import evaluate

    rows = [
        {"исход": "держим режим", "усилие": 0.0, "превышение за 24 ч": True},
        {"исход": "меняем уставки", "усилие": 2.0, "превышение за 24 ч": True},
        {"исход": "меняем уставки", "усилие": 3.0, "превышение за 24 ч": False},
        {"исход": "отказ", "усилие": 0.0, "превышение за 24 ч": False},
    ]
    out = evaluate(rows, limit=10.0)
    assert out["пропущено"] == 1 and out["доля пропусков"] == pytest.approx(0.5)
    assert out["ложных тревог"] == 1
    assert out["суммарно °C"] == pytest.approx(5.0)
    assert out["отказов"] == 1


def test_single_agent_rule_is_threshold_only():
    """Одноагентная конфигурация обязана быть именно тривиальной, иначе сравнение
    перестаёт быть сравнением с базой."""
    from nefte.agents.schemas import QualityAssessment
    from scripts.compare_architectures import single_agent_decision

    quiet = QualityAssessment(ts=pd.Timestamp("2026-01-01").to_pydatetime(),
                              spec_risk={"product_sulfur_mgkg": 0.05})
    loud = QualityAssessment(ts=pd.Timestamp("2026-01-01").to_pydatetime(),
                             spec_risk={"product_sulfur_mgkg": 0.9})
    assert single_agent_decision(quiet, 0.2) == "держим режим"
    assert single_agent_decision(loud, 0.2) == "меняем уставки"
    assert np.isfinite(0.0)
