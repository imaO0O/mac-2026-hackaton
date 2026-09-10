"""Тесты оптимизатора и разрешения конфликта целей (участник 3)."""
from __future__ import annotations

import pytest

from nefte.agents.optimizer import (
    OptimizerAgent,
    default_energy_proxy,
    default_throughput,
    linear_surrogate,
)
from nefte.agents.orchestrator import Orchestrator
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent, SeverityNorms
from nefte.agents.schemas import QualityAssessment, ReliabilityAssessment
from tests.test_agents import make_state

BOUNDS = {"T5": (365.0, 375.0), "T11": (360.0, 370.0), "F26": (200.0, 300.0)}


def _agent(sensitivities=None) -> OptimizerAgent:
    return OptimizerAgent(
        bounds=BOUNDS,
        surrogate=linear_surrogate(sensitivities or {"T5": -0.5, "T11": -0.5, "F26": 0.01}),
    )


def _assessments(sigma: float = 1.0, severity: float = 0.3, admissible: bool = True):
    state = make_state()
    quality = QualityAssessment(
        ts=state.ts, predictions={"product_sulfur_mgkg": 8.0},
        intervals={"product_sulfur_mgkg": (8.0 - 1.96 * sigma, 8.0 + 1.96 * sigma)},
        spec_risk={"product_sulfur_mgkg": 0.2}, confidence=0.8)
    reliability = ReliabilityAssessment(
        ts=state.ts, severity_index=severity,
        risk_class="low" if severity < 0.5 else "high", admissible=admissible)
    return state, quality, reliability


def test_hold_candidate_always_generated():
    """Без варианта «ничего не делать» система обязана что-то менять — это неверно."""
    state, _, reliability = _assessments()
    cands = _agent().generate(state, reliability)
    hold = [c for c in cands if c.id == "hold"]
    assert len(hold) == 1
    assert all(abs(d) < 1e-9 for d in hold[0].deltas.values())


def test_single_move_candidates_exist():
    """Покоординатная сетка: рекомендация «поменяй один параметр» понятнее всего."""
    state, _, reliability = _assessments()
    cands = _agent().generate(state, reliability)
    singles = [c for c in cands
               if sum(1 for d in c.deltas.values() if abs(d) > 1e-6) == 1]
    assert singles, "нет ни одного варианта с изменением ровно одной уставки"


def test_bounds_and_step_limit_are_respected():
    state, _, reliability = _assessments()
    agent = _agent()
    for c in agent.generate(state, reliability):
        for tag, value in c.moves.items():
            lo, hi = BOUNDS[tag]
            assert lo - 1e-6 <= value <= hi + 1e-6
            current = state.telemetry_ht[tag]
            if tag.startswith("T"):
                assert abs(value - current) <= 2.0 + 1e-6      # max_step_per_cycle


def test_reliability_constraints_shrink_search_space():
    """Ограничение надёжности сужает поиск — кроме «ничего не делать»:
    бездействие всегда физически возможно, даже если текущее значение вне рамок."""
    state, quality, reliability = _assessments()
    reliability.constraints = {"T5": (368.0, 369.0)}
    for c in _agent().generate(state, reliability):
        if c.id == "hold":
            continue
        assert 368.0 - 1e-6 <= c.moves["T5"] <= 369.0 + 1e-6


def test_inadmissible_regime_rejects_everything():
    """Запрет агента надёжности нельзя перевесить хорошим прогнозом качества."""
    state, quality, reliability = _assessments(admissible=False)
    agent = _agent()
    assert agent.propose(state, quality, reliability) == []
    assert "недопустим" in agent.rejection_summary()


def test_guaranteed_flag_separates_margin_from_improvement():
    """Вариант у предела допустим, только если он лучше бездействия — и без гарантии."""
    state, quality, reliability = _assessments(sigma=1.0)
    # текущая сера 6.2 (ПАК), уводим прогноз к пределу большой чувствительностью
    agent = OptimizerAgent(bounds=BOUNDS,
                           surrogate=linear_surrogate({"T5": 2.0, "T11": 0.0, "F26": 0.0}))
    ranked = agent.propose(state, quality, reliability)
    assert ranked
    # гарантированные варианты стоят выше негарантированных
    flags = [c.guaranteed for c in ranked]
    assert flags == sorted(flags, reverse=True)


def test_ranking_prefers_quality_margin_until_it_is_enough():
    """Запас по качеству ценен ДО требуемого, дальше решают выпуск и нагрузка.

    Раньше критерий качества всегда тянул «ещё чище», и в замкнутом контуре
    система шаг за шагом уводила уставки в упор допустимого дрейфа: поштучно
    каждый шаг разумен, а последовательность — нет. Теперь запас насыщается.
    """
    state, quality, reliability = _assessments()
    agent = _agent()
    ranked = agent.propose(state, quality, reliability)
    limit = agent.cfg["spec"]["product_sulfur_mgkg"]["max"]
    required = agent._required_margin

    # выбранный вариант обязан иметь достаточный запас…
    best = ranked[0].predicted_quality["product_sulfur_mgkg"]
    assert best <= limit - required + 1e-9
    # …но не обязан быть самым глубоким: за пределом требуемого запаса
    # решают выпуск, энергия и нагрузка на оборудование
    deepest = min(c.predicted_quality["product_sulfur_mgkg"] for c in ranked)
    assert deepest <= best


def test_deeper_is_better_while_the_margin_is_not_reached():
    """Пока запаса не хватает, глубже — лучше: насыщение не должно это ломать."""
    state, quality, reliability = _assessments()
    agent = _agent()
    ranked = agent.propose(state, quality, reliability)
    limit = agent.cfg["spec"]["product_sulfur_mgkg"]["max"]
    required = agent._required_margin
    short = [c for c in ranked if c.predicted_quality["product_sulfur_mgkg"]
             > limit - required]
    if len(short) > 1:
        values = [c.predicted_quality["product_sulfur_mgkg"] for c in short]
        assert values == sorted(values)


def test_pareto_front_is_subset_and_nonempty():
    state, quality, reliability = _assessments()
    agent = _agent()
    ranked = agent.propose(state, quality, reliability)
    front = agent.pareto_front(ranked)
    assert front and len(front) <= len(ranked)
    assert all(c.pareto_rank == 0 for c in front)


def test_alternatives_are_actually_different():
    state, quality, reliability = _assessments()
    agent = _agent()
    alts = agent.diverse_alternatives(agent.propose(state, quality, reliability), 3)
    assert len(alts) == 3
    for i, a in enumerate(alts):
        for b in alts[i + 1:]:
            assert any(abs(a.deltas[t] - b.deltas[t]) > 1e-3 for t in a.deltas)


def test_throughput_scales_with_feed():
    state, _, _ = _assessments()
    state.telemetry_ht.update({"F17": 250.0, "F26": 250.0})
    fn = default_throughput()
    assert fn(state, {"F26": 250.0}) == pytest.approx(250.0)
    assert fn(state, {"F26": 275.0}) == pytest.approx(275.0)


def test_energy_proxy_grows_with_temperature():
    state, _, _ = _assessments()
    fn = default_energy_proxy()
    assert fn(state, {"T5": 375.0}) > fn(state, {"T5": 365.0})


def test_action_frequency_limit_blocks_immediate_second_move():
    """После воздействия нужно дождаться отклика, а не давить снова."""
    norms = SeverityNorms(bounds={"wabt": (350.0, 375.0)})
    agent = OptimizerAgent(bounds=BOUNDS,
                           surrogate=linear_surrogate({"T5": -0.5, "T11": -0.5, "F26": 0.0}))
    system = Orchestrator(QualityAgent(), ReliabilityAgent(norms), agent,
                          log_runs=False, act_risk_threshold=0.0,
                          min_hours_between_actions=4.0)

    first = system.run(make_state(lims=(9.5, 1.0), pak=(9.6, 0.1)))
    assert first.action is not None

    state2 = make_state(lims=(9.5, 1.0), pak=(9.6, 0.1))
    state2.ts = state2.ts.replace(hour=state2.ts.hour + 1)
    second = system.run(state2)
    assert second.action is not None
    assert all(abs(d) < 1e-9 for d in second.action.deltas.values())
    assert "ждём отклика" in second.explanation
