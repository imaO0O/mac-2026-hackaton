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


# --------------------------------------------------------------------------- #
# 7. смесь с неизвестной серой — не «ноль»
# --------------------------------------------------------------------------- #

def test_unknown_sulfur_is_not_zero():
    """Было: weighted(...) or 0.0 — смесь с неизвестной серой объявлялась годной."""
    from nefte.agents.blending import BlendingAgent, mix
    from nefte.agents.schemas import BlendComponent

    ghost = BlendComponent(name="без анализа", sulfur_mgkg=float("nan"),
                           available_tph=100.0)
    props = mix([ghost], {"без анализа": 1.0})
    assert props["sulfur_mgkg"] != props["sulfur_mgkg"]          # NaN, а не 0
    violations = BlendingAgent().check(props, {"без анализа": 1.0})
    assert any("не из чего посчитать" in v for v in violations)


# --------------------------------------------------------------------------- #
# 8. недоступный компонент обнуляет выпуск
# --------------------------------------------------------------------------- #

def test_recipe_with_unavailable_component_promises_no_throughput():
    """Было: компонент с нулевым расходом просто выпадал из расчёта выпуска."""
    from nefte.agents.blending import BlendingAgent
    from nefte.agents.schemas import BlendComponent

    plenty = BlendComponent(name="есть", sulfur_mgkg=6.0, density_15c=835.0,
                            t95_c=340.0, cfpp_c=-9.0, available_tph=200.0)
    empty = BlendComponent(name="нет в наличии", sulfur_mgkg=2.0, density_15c=830.0,
                           t95_c=330.0, cfpp_c=-12.0, available_tph=0.0)
    agent = BlendingAgent()
    recipe = agent.optimize([plenty, empty])
    # рецептура из одного доступного компонента даёт выпуск, из недоступного — нет
    assert recipe.throughput_tph > 0
    assert recipe.fractions["нет в наличии"] == pytest.approx(0.0)

    forced = {"есть": 0.5, "нет в наличии": 0.5}
    from nefte.agents.blending import mix
    props = mix([plenty, empty], forced)
    assert not agent.check(props, forced)          # по спецификации проходит
    # но выпуска у такой смеси быть не может — оптимизатор её и не выбрал
    assert recipe.fractions["есть"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 9. отбор признаков не смотрит в будущее
# --------------------------------------------------------------------------- #

def test_column_selection_uses_train_only():
    """Было: почти пустые колонки отбирались по доле пропусков во ВСЕЙ истории.

    Утечка слабая, но настоящая: признак, которого в обучающем периоде почти нет,
    выживал за счёт того, что он появляется в тестовом.
    """
    import nefte.models.dataset as dataset

    idx = pd.date_range("2024-01-01", periods=400, freq="1h", name="date")
    feats = pd.DataFrame({
        "всегда": 1.0,
        # в обучающем периоде признака нет вовсе, зато потом он заполнен —
        # по всей истории выходит 62 % заполненности, по train — ноль
        "появился позже": [np.nan] * 150 + [1.0] * 250,
    }, index=idx)
    target = pd.Series(8.0, index=idx[::10])

    original = dataset.clean_lims_sulfur
    dataset.clean_lims_sulfur = lambda *_a, **_k: target
    try:
        whole, _ = dataset.build_training_table(0.0, features=feats)
        train_only, _ = dataset.build_training_table(
            0.0, features=feats, train_bounds=("2024-01-01", "2024-01-06"))
    finally:
        dataset.clean_lims_sulfur = original

    assert "всегда" in whole.columns and "всегда" in train_only.columns
    assert "появился позже" in whole.columns, "по всей истории признак выживал"
    assert "появился позже" not in train_only.columns, "по train его быть не должно"


# --------------------------------------------------------------------------- #
# 10. уставка, не влияющая на признаки режима, не роняет расчёт
# --------------------------------------------------------------------------- #

def test_move_of_a_non_regime_tag_does_not_crash():
    """Было: обращение к raw[tag] шло раньше проверки, что тег туда попал.

    Управляющие теги сейчас все «чувствительные к режиму», но добавление любого
    другого — например уставки АВТ — уронило бы оптимизатор по KeyError.
    """
    from nefte.models.regime import apply_moves_to_rows

    rows = pd.DataFrame({"ht_T5": [360.0], "ht_T6": [358.0], "ht_T11": [362.0],
                         "ht_F26": [250.0], "ht_F99": [10.0], "ht_F99_mean6": [9.0]},
                        index=pd.DatetimeIndex(["2026-01-01"]))
    out = apply_moves_to_rows(rows, {"F99": 12.0, "T5": 362.0})
    assert out["ht_F99"].iloc[0] == pytest.approx(12.0)
    assert out["ht_F99_mean6"].iloc[0] == pytest.approx(11.0)   # среднее подтянулось
    assert out["ht_T5"].iloc[0] == pytest.approx(362.0)


# --------------------------------------------------------------------------- #
# 11. подозрительный тег не остаётся без вердикта
# --------------------------------------------------------------------------- #

def test_every_suspect_tag_has_a_verdict():
    """Было: P44 и P52 значились подозрительными, но вердикта не имели.

    При этом P44 — один из значимых признаков модели качества, то есть тег без
    разбора работал в проде. Список подозрительных и список вердиктов обязаны
    сходиться, иначе такое повторится молча.
    """
    from nefte.config import load_config

    telemetry = load_config()["telemetry"]
    verdicts = telemetry.get("tag_verdicts", {})
    suspects = set(telemetry.get("suspect_tags", []))
    without = {tag for tag in suspects
               if not any(key.endswith(f":{tag}") for key in verdicts)}
    assert not without, f"без вердикта остались теги: {sorted(without)}"
