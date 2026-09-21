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
from nefte.config import load_config
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
    """Запрет агента надёжности нельзя перевесить хорошим прогнозом качества.

    Так ведёт себя режим ``severity_veto: all``. С 21.09 по умолчанию стоит
    ``raise_only``: вычёркивается то, что греет или утяжеляет режим, а шаг к
    безопасности остаётся (test_raise_only_never_proposes_heating_on_inadmissible_regime).
    """
    state, quality, reliability = _assessments(admissible=False)
    agent = OptimizerAgent(
        bounds=BOUNDS, cfg=_cfg("reliability", "severity_veto", "all"),
        surrogate=linear_surrogate({"T5": -0.5, "T11": -0.5, "F26": 0.01}))
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


def test_action_interval_comes_from_config_not_from_a_default_argument():
    """Интервал между воздействиями — параметр безопасности, а не деталь вызова.

    Вместе с limits.max_step_per_cycle он задаёт предельную СКОРОСТЬ изменения
    режима, а её организаторы в пакете не задавали — значит, выбор наш, и он
    обязан быть виден в конфиге. Пока значение жило в аргументе по умолчанию,
    оно менялось у каждого, кто иначе создал оркестратор.
    """
    cfg = load_config()
    norms = SeverityNorms(bounds={"wabt": (350.0, 375.0)})
    agent = OptimizerAgent(bounds=BOUNDS, surrogate=linear_surrogate({"T5": -0.5}))
    system = Orchestrator(QualityAgent(), ReliabilityAgent(norms), agent, log_runs=False)
    assert system.min_hours_between_actions == cfg["limits"]["min_hours_between_actions"]


def test_ramp_rate_is_bounded_by_step_and_interval_together():
    """Предельная скорость изменения режима = шаг за цикл / интервал воздействий.

    Ни одно из двух чисел по отдельности скорость не ограничивает: шаг без
    интервала позволяет двигать уставку каждый цикл, интервал без шага — двигать
    редко, но сразу далеко. Тест закрепляет именно произведение, потому что
    защищает нас оно, а поменять могут любое из двух.

    0.5 °C/ч не взято с потолка: постоянная времени канала серы измерена и равна
    4.6 ч (scripts/find_delays.py). Двигать быстрее, чем процесс отвечает, значит
    гоняться за собственным ещё не проявившимся воздействием.
    """
    limits = load_config()["limits"]
    rate = limits["max_step_per_cycle"]["temperature_c"] / limits["min_hours_between_actions"]
    assert rate == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# переключатели решений 21.09: вето тяжести и консервативная Т95
# --------------------------------------------------------------------------- #

def _cfg(section: str, key: str, value: str) -> dict:
    cfg = load_config()
    return {**cfg, section: {**cfg[section], key: value}}


def _move(deltas: dict[str, float], severity: float):
    from nefte.agents.schemas import Candidate

    return Candidate(id="v", moves={t: 370.0 + d for t, d in deltas.items()},
                     deltas=deltas, severity_index=severity)


def test_severity_veto_all_strikes_every_variant():
    """Прежнее поведение: высокий класс тяжести вычёркивает всё, даже шаг назад."""
    _, _, reliability = _assessments(severity=0.8, admissible=False)
    agent = OptimizerAgent(bounds=BOUNDS, surrogate=linear_surrogate({"T5": -0.5}),
                           cfg=_cfg("reliability", "severity_veto", "all"))
    assert agent._severity_vetoes(_move({"T5": -1.0}, 0.7), reliability)
    assert agent._severity_vetoes(_move({"F26": -10.0}, 0.75), reliability)


def test_severity_veto_raise_only_keeps_steps_toward_safety():
    """Кандидат 21.09: вычёркивается только то, что тяжесть повышает или греет."""
    _, _, reliability = _assessments(severity=0.8, admissible=False)
    agent = OptimizerAgent(bounds=BOUNDS, surrogate=linear_surrogate({"T5": -0.5}),
                           cfg=_cfg("reliability", "severity_veto", "raise_only"))
    assert not agent._severity_vetoes(_move({"T5": -1.0}, 0.7), reliability)
    assert not agent._severity_vetoes(_move({"F26": -10.0}, 0.8), reliability)
    assert agent._severity_vetoes(_move({"F26": 10.0}, 0.85), reliability)
    # подъём температуры запрещён, даже если тяжесть по расчёту не выросла
    assert agent._severity_vetoes(_move({"T5": 0.5}, 0.7), reliability)


def test_raise_only_never_proposes_heating_on_inadmissible_regime():
    """Условие 4 правила приёмки — на уровне оптимизатора: при недопустимом режиме
    ни один предложенный вариант не поднимает температуру."""
    state, quality, reliability = _assessments(severity=0.8, admissible=False)
    agent = OptimizerAgent(bounds=BOUNDS,
                           surrogate=linear_surrogate({"T5": -0.5, "T11": -0.5, "F26": 0.01}),
                           cfg=_cfg("reliability", "severity_veto", "raise_only"))
    proposed = agent.propose(state, quality, reliability)
    assert proposed, "шаг к безопасности должен остаться"
    for c in proposed:
        assert all(d <= 1e-6 for t, d in c.deltas.items() if t.startswith("T")), c.deltas


def _t95_state(age_hours: float):
    from nefte.agents.schemas import Measurement, Source

    state = make_state()
    state.quality["lims_t95_c"] = Measurement(value=352.0, unit="°C", source=Source.LIMS,
                                              age_hours=age_hours)
    return state


def test_t95_shift_is_zero_for_last_and_sigma_for_upper_bound():
    from nefte.agents.quality import t95_sigma

    state = _t95_state(age_hours=40.0)
    last = OptimizerAgent(bounds=BOUNDS, surrogate=linear_surrogate({"T5": -0.5}),
                          cfg=_cfg("quality", "t95_estimate", "last"))
    upper = OptimizerAgent(bounds=BOUNDS, surrogate=linear_surrogate({"T5": -0.5}),
                           cfg=_cfg("quality", "t95_estimate", "last_plus_sigma"))
    assert last._t95_shift(state) == 0.0
    assert upper._t95_shift(state) == pytest.approx(t95_sigma(40.0))
    assert upper._t95_shift(state) > 0


def test_upper_bound_t95_strikes_heating_that_the_last_analysis_allows():
    """Т95 варианта ниже предела, но ближе σ: по последнему анализу вариант
    допустим, по верхней границе доверия — нет. Сдвиг только в проверке:
    прогноз Т95 в варианте остаётся прежним."""
    from nefte.agents.quality import t95_sigma

    state = _t95_state(age_hours=40.0)
    _, quality, reliability = _assessments()
    limit = load_config()["spec"]["t95_c"]["max"]
    sigma = t95_sigma(40.0)

    def t95_fn(_state, moves):          # текущий режим далеко, подъём T5 — вплотную
        return limit - 0.5 * sigma if moves.get("T5", 370.0) > 370.0 + 1e-6 \
            else limit - 3.0 * sigma

    def violated(mode: str) -> bool:
        agent = OptimizerAgent(bounds=BOUNDS, surrogate=linear_surrogate({"T5": -0.5}),
                               cfg=_cfg("quality", "t95_estimate", mode), t95_fn=t95_fn)
        heat = _move({"T5": 1.0}, 0.3)
        agent.evaluate(state, [heat], quality, reliability)
        assert heat.predicted_quality["product_t95_c"] == pytest.approx(limit - 0.5 * sigma)
        return any("Т95" in v for v in heat.violations)

    assert not violated("last")
    assert violated("last_plus_sigma")
