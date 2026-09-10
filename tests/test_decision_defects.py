"""Тесты на дефекты, найденные обзором пути принятия решения.

Каждый тест здесь закрывает конкретную ошибку, которая уже была в рабочем коде и
которую легко вернуть обратно неосторожной правкой. Поэтому в каждом написано,
что именно ломалось.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.agents.optimizer import PARETO_EPS, OptimizerAgent, linear_surrogate
from nefte.agents.quality import (
    SOURCE_CONFIDENCE,
    QualityAgent,
    confidence_parts,
)
from nefte.agents.reliability import ReliabilityAgent, SeverityNorms
from nefte.agents.schemas import Source
from nefte.models.dataset import cache_key
from tests.test_agents import build_system, make_state

NORMS = SeverityNorms(bounds={"wabt": (355.0, 375.0), "W10": (1.0, 4.0),
                              "T55": (370.0, 395.0)})
BOUNDS = {"T5": (365.0, 375.0), "T11": (360.0, 370.0),
          "F26": (200.0, 300.0), "P13": (3.5, 4.2)}


def _optimizer(agent: ReliabilityAgent | None) -> OptimizerAgent:
    return OptimizerAgent(bounds=BOUNDS, reliability_agent=agent,
                          surrogate=linear_surrogate({"T5": -0.15, "T11": -0.15,
                                                      "P13": -0.5}))


# --------------------------------------------------------------------------- #
# 1. тяжесть режима у каждого варианта своя
# --------------------------------------------------------------------------- #

def test_severity_differs_between_candidates():
    """Было: у всех вариантов стоял severity текущего режима.

    Нормировка одинаковых чисел даёт константу, поэтому критерий severity в
    свёртке и на фронте Парето не работал вовсе — оптимизация была не по четырём
    критериям, а по трём.
    """
    agent = ReliabilityAgent(NORMS)
    state = make_state()
    optimizer = _optimizer(agent)
    quality = QualityAgent().assess(state)
    cands = optimizer.propose(state, quality, agent.assess(state))

    values = {round(c.severity_index, 6) for c in cands if c.severity_index is not None}
    assert len(values) > 1, "тяжесть режима обязана различаться между вариантами"


def test_raising_reactor_temperature_raises_severity():
    """Физика: глубже режим — тяжелее оборудованию. Знак обязан быть таким."""
    agent = ReliabilityAgent(NORMS)
    state = make_state()
    base = agent.severity_for(state, {})
    hotter = agent.severity_for(state, {t: state.telemetry_ht[t] + 3.0
                                        for t in ("T5", "T6", "T11")})
    cooler = agent.severity_for(state, {t: state.telemetry_ht[t] - 3.0
                                        for t in ("T5", "T6", "T11")})
    assert hotter > base > cooler


def test_severity_falls_back_to_current_without_agent():
    """Без агента надёжности оптимизатор работает, просто критерий не различает."""
    state = make_state()
    reliability = ReliabilityAgent(NORMS).assess(state)
    cands = _optimizer(None).propose(state, QualityAgent().assess(state), reliability)
    assert all(c.severity_index == pytest.approx(reliability.severity_index)
               for c in cands)


# --------------------------------------------------------------------------- #
# 2. риск варианта — вероятность, а не флаг
# --------------------------------------------------------------------------- #

def test_candidate_risk_is_a_probability_not_a_flag():
    """Было: 0 или 1. В карточке оператора все альтернативы выглядели одинаково."""
    agent = ReliabilityAgent(NORMS)
    state = make_state(lims=(9.4, 1.0), pak=(9.5, 0.1))
    cands = _optimizer(agent).propose(state, QualityAgent().assess(state),
                                      agent.assess(state))
    risks = [c.spec_risk["product_sulfur_mgkg"] for c in cands]
    assert all(0.0 <= r <= 1.0 for r in risks)
    assert len({round(r, 4) for r in risks}) > 2


def test_lower_predicted_sulfur_means_lower_risk():
    agent = ReliabilityAgent(NORMS)
    state = make_state(lims=(9.4, 1.0), pak=(9.5, 0.1))
    cands = _optimizer(agent).propose(state, QualityAgent().assess(state),
                                      agent.assess(state))
    pairs = sorted((c.predicted_quality["product_sulfur_mgkg"],
                    c.spec_risk["product_sulfur_mgkg"]) for c in cands)
    risks = [r for _, r in pairs]
    assert risks == sorted(risks), "риск обязан расти вместе с прогнозом серы"


# --------------------------------------------------------------------------- #
# 3. фронт Парето остаётся читаемым
# --------------------------------------------------------------------------- #

def test_pareto_front_is_not_everything():
    """При четырёх непрерывных критериях строгий фронт вырождается в «все»."""
    agent = ReliabilityAgent(NORMS)
    state = make_state(lims=(9.4, 1.0), pak=(9.5, 0.1))
    optimizer = _optimizer(agent)
    cands = optimizer.propose(state, QualityAgent().assess(state), agent.assess(state))
    front = optimizer.pareto_front(cands)
    assert 0 < len(front) < len(cands), "фронт обязан быть подмножеством, а не всем"
    assert PARETO_EPS > 0


# --------------------------------------------------------------------------- #
# 4. уверенность что-то значит
# --------------------------------------------------------------------------- #

def test_confidence_drops_with_stale_analysis():
    """Было: уверенность считалась только по σ и у обученной модели была всегда 0.95.

    σ ничего не знает ни про устаревший анализ, ни про молчащий прибор.
    """
    fresh = confidence_parts(1.7, Source.LIMS, 2.0, 24.0, True)
    stale = confidence_parts(1.7, Source.LIMS, 120.0, 24.0, True)
    assert np.prod(list(stale.values())) < np.prod(list(fresh.values()))


def test_confidence_depends_on_source():
    lims = confidence_parts(1.7, Source.LIMS, 1.0, 24.0, True)["источник"]
    pak = confidence_parts(1.7, Source.PAK, 1.0, 24.0, True)["источник"]
    vak = confidence_parts(1.7, Source.VAK, 1.0, 24.0, True)["источник"]
    assert lims > pak > vak
    assert SOURCE_CONFIDENCE[Source.NONE] == 0.0


def test_confidence_names_its_weakest_link():
    """Оператору важно не число, а почему оно такое."""
    state = make_state(lims=(6.0, 120.0), pak=None)
    out = QualityAgent().assess(state)
    assert out.confidence < 0.9
    assert any("снижает" in note for note in out.notes)


# --------------------------------------------------------------------------- #
# 5. кэш признаков привязан к настройкам
# --------------------------------------------------------------------------- #

def test_feature_cache_key_reacts_to_settings():
    """Было: кэш лежал в features_1h.parquet и молча переживал смену набора тегов."""
    import nefte.models.dataset as dataset

    before = cache_key("1h")
    original = list(dataset.AVT_TAGS)
    try:
        dataset.AVT_TAGS = original + ["T99"]
        assert cache_key("1h") != before
    finally:
        dataset.AVT_TAGS = original
    assert cache_key("1h") == before
    assert cache_key("10min") != before


# --------------------------------------------------------------------------- #
# 6. в карточке сказано, что НЕ проверено
# --------------------------------------------------------------------------- #

def test_operator_card_admits_unchecked_properties():
    """Прогнозируется только сера; молчать об этом нельзя."""
    from nefte.agents.blending import BlendingAgent
    from tests.test_orchestrator_blending import GODT
    from tests.test_orchestrator_blending import build_system as blend_system

    rec = blend_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert any("НЕ прогнозируются" in item for item in rec.checked_constraints)
    assert isinstance(BlendingAgent(), BlendingAgent) and GODT.sulfur_mgkg > 0


def test_plain_system_still_lists_hard_limit():
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert any("сера" in item for item in rec.checked_constraints)
    assert isinstance(pd.Timestamp(rec.ts), pd.Timestamp)
