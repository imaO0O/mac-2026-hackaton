"""Тесты на дефекты, найденные обзором пути принятия решения.

Каждый тест здесь закрывает конкретную ошибку, которая уже была в рабочем коде и
которую легко вернуть обратно неосторожной правкой. Поэтому в каждом написано,
что именно ломалось.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from nefte.agents.optimizer import PARETO_EPS, OptimizerAgent, linear_surrogate
from nefte.agents.orchestrator import Orchestrator
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


def test_severity_accounts_for_the_reactor_temperature_it_cannot_set():
    """WABT считалась по СТАРОЙ T6 и занижала тяжесть подъёма температуры.

    Оптимизатор двигает T5 и T11, а WABT — среднее по T5, T6 и T11. T6 никто не
    задаёт уставкой: это температура Р-202, следствие того, что сделали с T5.
    Пока связь не учитывалась, шаг +2 °C по T5 поднимал WABT на 0.67 °C вместо
    1.15 — то есть критерий надёжности систематически недооценивал цену подъёма
    температуры примерно на 40 %. Работал он при этом не в ту сторону, ради
    которой в системе и нужен.

    Связь измерена: ΔT6 = 0.72·ΔT5 + 0.18·ΔT11 (models/regime.py).
    """
    from nefte.models.regime import T6_RESPONSE

    agent = ReliabilityAgent(NORMS)
    state = make_state()
    t5 = state.telemetry_ht["T5"]
    grew = agent.severity_for(state, {"T5": t5 + 2.0}) - agent.severity_for(state, {})

    # то же самое, но с искусственно замороженной связью — так вело себя до правки
    frozen = dict(T6_RESPONSE)
    T6_RESPONSE.clear()
    try:
        naive = agent.severity_for(state, {"T5": t5 + 2.0}) - agent.severity_for(state, {})
    finally:
        T6_RESPONSE.update(frozen)

    assert grew > naive > 0, "учёт отклика T6 обязан УСИЛИВАТЬ реакцию severity"
    assert grew == pytest.approx(naive * (1 + T6_RESPONSE["T5"]), rel=0.05)


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


# --------------------------------------------------------------------------- #
# 12. факт вне спецификации выносится в карточку
# --------------------------------------------------------------------------- #

def test_card_says_when_the_measurement_is_already_off_spec():
    """Было: карточка говорила «риск 29 %», хотя лаборатория показала 10.2 мг/кг.

    Нашлось на дашборде: самый тревожный факт из доступных не попадал в карточку
    вовсе, и она читалась как разговор про будущий риск.
    """
    rec = build_system().run(make_state(lims=(10.2, 3.0), pak=(10.1, 0.1)))
    assert "ФАКТ ВНЕ СПЕЦИФИКАЦИИ" in rec.problem
    assert "10.20" in rec.problem
    assert rec.state_summary["sulfur_measured"] == pytest.approx(10.2)


def test_card_stays_quiet_when_the_measurement_is_within_spec():
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert "ФАКТ ВНЕ СПЕЦИФИКАЦИИ" not in rec.problem
    assert rec.state_summary["sulfur_measured"] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# 13. альтернативы действительно различаются
# --------------------------------------------------------------------------- #

def test_alternatives_differ_by_more_than_noise():
    """Было: порог различия 1e-3, и три «разные» альтернативы отличались в третьем
    знаке. На дашборде график их разброса показывал шум вместо выбора."""
    agent = ReliabilityAgent(NORMS)
    state = make_state(lims=(9.4, 1.0), pak=(9.5, 0.1))
    optimizer = _optimizer(agent)
    cands = optimizer.propose(state, QualityAgent().assess(state), agent.assess(state))

    threshold = optimizer.min_alternative_distance()
    assert threshold > 0.1, "порог обязан быть соизмерим с шагом уставки"

    picked = optimizer.diverse_alternatives(cands, 3)
    assert len(picked) <= 3
    spreads = [sum(abs(a.deltas.get(t, 0.0) - b.deltas.get(t, 0.0))
                   for t in set(a.deltas) | set(b.deltas))
               for i, a in enumerate(picked) for b in picked[i + 1:]]
    assert spreads and max(spreads) > threshold


# --------------------------------------------------------------------------- #
# 14. запрещённый вердиктом тег не проходит ни одним путём
# --------------------------------------------------------------------------- #

def test_banned_tags_do_not_enter_through_vak_formulas():
    """Вердикт do_not_use соблюдался по договорённости, а не проверкой.

    Список признаков AVT_TAGS тест уже стерёг, но формулы ВАК считаются по СЫРЫМ
    тегам той же установки и могли протащить запрещённый тег в модель мимо него.
    Проверять надо с учётом установки: avt:F26 — это пар в К-6 и он запрещён, а
    ht:F26 — расход сырья, один из управляющих тегов.
    """
    import re

    from nefte.config import load_config
    from nefte.models.vak import compile_formulas

    banned: dict[str, set[str]] = {}
    for key, verdict in load_config()["telemetry"]["tag_verdicts"].items():
        if verdict["status"] == "do_not_use":
            unit, tag = key.split(":")
            banned.setdefault(unit, set()).add(tag)

    usable, _ = compile_formulas()
    violations = []
    for item in usable:
        used = set(re.findall(r"\b([A-Z]{1,4}[0-9]{1,3})\b", str(item["expr"])))
        bad = used & banned.get(item["unit"], set())
        if bad:
            violations.append(f"{item['target']} ({item['unit']}): {sorted(bad)}")
    assert not violations, "формулы ВАК используют запрещённые теги: " + "; ".join(violations)


# --------------------------------------------------------------------------- #
# 15. решения об отборе признаков — только по обучающему периоду
# --------------------------------------------------------------------------- #

def _flow_dictionary(monkeypatch) -> None:
    """Справочник, в котором F7 — расход: детектор знака решает по описанию КИП.

    Без подстановки тест зависел от выданного справочника и на машине без данных
    проверял не правило, а отсутствие файла.
    """
    frame = pd.DataFrame({"unit": ["avt", "avt"], "code": ["F7", "T1"],
                          "description": ["Расход обессоленной нефти", "Температура верха К1"]})
    monkeypatch.setattr("nefte.data.validity.load_tag_dictionary", lambda: frame)


def test_negative_share_is_measured_on_train_only(monkeypatch):
    """Было: доля отрицательных считалась по всей истории вместе с тестом.

    Это такое же выведенное из данных правило, как нормировка severity, и
    выводить его по будущему нельзя.
    """
    from nefte.data.validity import SignalValidity

    _flow_dictionary(monkeypatch)

    idx = pd.date_range("2024-01-01", periods=400, freq="1h", name="date")
    values = np.r_[np.full(200, 5.0), np.full(200, -5.0)]      # брак только «после»
    raw = pd.DataFrame({"F7": values}, index=idx)
    cfg = {"telemetry": {"sentinel_values": [307.0], "frozen_min_samples": 10_000,
                         "dead_tags": []},
           "split": {"train": ["2024-01-01", "2024-01-08"]}}

    built = SignalValidity.build(raw, unit="avt", cfg=cfg)
    # в обучающем периоде отрицательных нет — тег считается неотрицательным,
    # и поздний брак маскируется, а не легализуется задним числом
    assert built.negative["F7"].iloc[300]
    assert not built.negative["F7"].iloc[10]


# --------------------------------------------------------------------------- #
# 16. один и тот же момент даёт один и тот же ответ
# --------------------------------------------------------------------------- #

def test_same_moment_gives_the_same_recommendation():
    """Было: генератор случайных чисел жил в агенте и продвигался от вызова к вызову.

    Повторный прогон того же среза давал ДРУГУЮ рекомендацию. Это ломало
    обещание воспроизводимости и портило обе проверки устойчивости: базовый и
    возмущённый прогоны считались при разном состоянии генератора, так что в
    «чувствительность к весам» попадал ещё и случайный разброс.
    """
    agent = ReliabilityAgent(NORMS)
    state = make_state(lims=(9.4, 1.0), pak=(9.5, 0.1))
    optimizer = _optimizer(agent)
    quality = QualityAgent().assess(state)

    first = optimizer.propose(state, quality, agent.assess(state))[0]
    second = optimizer.propose(state, quality, agent.assess(state))[0]
    assert first.id == second.id
    assert first.deltas == pytest.approx(second.deltas)


def test_different_moments_give_different_candidate_sets():
    """Привязка к моменту не должна выродиться в один и тот же набор на всю историю."""
    agent = ReliabilityAgent(NORMS)
    optimizer = _optimizer(agent)
    early = make_state()
    late = make_state()
    late.ts = pd.Timestamp(early.ts) + pd.Timedelta(hours=6)

    a = optimizer.generate(early, agent.assess(early))
    b = optimizer.generate(late, agent.assess(late))
    mixes_a = [c.moves for c in a if c.id.startswith("mix_")]
    mixes_b = [c.moves for c in b if c.id.startswith("mix_")]
    assert mixes_a and mixes_b and mixes_a[0] != mixes_b[0]


# --------------------------------------------------------------------------- #
# 17. объяснение не утверждает того, чего не было
# --------------------------------------------------------------------------- #

def test_no_vak_note_when_the_model_returned_nothing():
    """Было: «значение получено виртуальным анализатором» писалось и при NaN.

    Момент раньше начала истории — модель возвращает NaN, а примечание утверждало,
    что значение получено. Объяснение обязано быть верным даже в углу.
    """
    class SilentModel:
        horizon_hours = 0.0
        alarm_threshold = 0.2

        def predict_with_sigma(self, state):
            return float("nan"), float("inf")

    out = QualityAgent(model=SilentModel()).assess(make_state(lims=None, pak=None))
    assert out.source is Source.NONE and out.confidence == 0.0
    assert not any("ВАК" in note for note in out.notes)
    assert any("прогноз недоступен" in note for note in out.notes)


# --------------------------------------------------------------------------- #
# 18. происхождение эффекта названо оператору
# --------------------------------------------------------------------------- #

def test_kinetic_effect_is_declared_as_physics_not_measurement():
    """«Сера к бездействию −3.4 мг/кг» читается как факт, а это допущение.

    Уровень серы берёт модель, а приращение от изменения уставок считает кинетика
    с принятой энергией активации. Оператору это надо сказать прямо.
    """
    from nefte.models.kinetics import make_kinetic_surrogate

    class LevelModel:
        horizon_hours = 0.0
        alarm_threshold = 0.2

        def predict_with_sigma(self, _state):
            return 9.5, 1.7

    surrogate = make_kinetic_surrogate(LevelModel())
    assert getattr(surrogate, "kind", None) == "kinetic"

    agent = ReliabilityAgent(NORMS)
    optimizer = OptimizerAgent(bounds=BOUNDS, surrogate=surrogate,
                               reliability_agent=agent)
    system = Orchestrator(QualityAgent(model=LevelModel()), agent, optimizer,
                          log_runs=False)
    rec = system.run(make_state(lims=(9.5, 1.0), pak=(9.6, 0.1)))
    if not rec.abstained:
        assert "кинетике" in rec.explanation and "допущение" in rec.explanation


# --------------------------------------------------------------------------- #
# 19. модель и оператор видят ОДНИ И ТЕ ЖЕ данные
# --------------------------------------------------------------------------- #

def test_training_and_serving_clean_data_the_same_way(monkeypatch):
    """Было: два пути очистки. Модель училась на одних данных, оператор видел другие.

    В обучающем пути отрицательные расходы не маскировались, а вердикты по тегам
    не соблюдались вовсе. Расхождение доходило до 0.9 % значений АВТ и задевало
    ht:Q21 — самый значимый признак модели.
    """
    import nefte.models.dataset as dataset
    from nefte.data.validity import SignalValidity

    _flow_dictionary(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=300, freq="10min", name="date")
    raw = pd.DataFrame({"F7": np.r_[np.full(150, 5.0), np.full(150, -5.0)],
                        "T1": np.linspace(300, 320, 300)}, index=idx)
    cfg = {"telemetry": {"sentinel_values": [307.0], "frozen_min_samples": 10_000,
                         "dead_tags": [],
                         "tag_verdicts": {"avt:T1": {"status": "do_not_use",
                                                     "reason": "проверка"}}},
           "split": {"train": ["2024-01-01", "2024-01-01 12:00"]}}

    clean = SignalValidity.build(raw, unit="avt", cfg=cfg).clean
    # вердикт соблюдён кодом, а не договорённостью
    assert "T1" not in clean.columns
    # отрицательные расходы замаскированы
    assert clean["F7"].isna().iloc[200]

    # и матрица признаков строится ровно этой же функцией
    source = pathlib.Path(dataset.__file__).read_text(encoding="utf-8")
    assert "SignalValidity.build" in source
    assert "clean_telemetry(" not in source


# --------------------------------------------------------------------------- #
# 20. лабораторный анализ используется не раньше публикации
# --------------------------------------------------------------------------- #

def test_lab_result_is_not_used_before_it_could_be_known():
    """Метка ЛИМС — момент ОТБОРА пробы, результат появляется до 4 часов позже.

    Подтверждено организаторами. Без поправки система использует анализ раньше,
    чем оператор мог его увидеть, — то есть заглядывает в будущее.
    """
    from nefte.data.features import known_from

    idx = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 12:00"], name="date")
    sampled = pd.Series([8.0, 9.0], index=idx)
    published = known_from(sampled, 4.0)

    assert published.index[0] == pd.Timestamp("2026-01-01 04:00")
    # через час после отбора значение ещё недоступно
    assert published.loc[:"2026-01-01 01:00"].empty
    # через пять — уже да
    assert float(published.loc[:"2026-01-01 05:00"].iloc[-1]) == pytest.approx(8.0)


def test_zero_delay_changes_nothing():
    """Поправка отключаема: без задержки ряд обязан остаться прежним."""
    from nefte.data.features import known_from

    idx = pd.DatetimeIndex(["2026-01-01", "2026-01-02"], name="date")
    series = pd.Series([1.0, 2.0], index=idx)
    assert known_from(series, 0.0).index.equals(idx)


def test_fact_series_keeps_the_sampling_time():
    """Факт, с которым сверяются прогоны, сдвигать НЕЛЬЗЯ.

    Превышение спецификации случилось в момент отбора пробы, а не публикации:
    сдвинув факт, мы сдвинули бы и оценку собственных пропусков.
    """
    import inspect

    from nefte.pipeline import StateBuilder

    source = inspect.getsource(StateBuilder.__init__)
    assert "self.lims_sulfur = clean_lims_sulfur(" in source
    assert "self.lims_sulfur_known = known_from(" in source


# --------------------------------------------------------------------------- #
# правило исхода: одно на всю систему
# --------------------------------------------------------------------------- #

def test_outcome_rule_exists_in_exactly_one_place():
    """«Отказ / меняем уставки / держим режим» считается ровно в одном месте.

    Это правило уже разъезжалось дважды. Оно жило пятью независимыми копиями в
    прогонах по тесту, в сравнении архитектур, в двух проверках устойчивости и в
    имитации; копии успели разойтись в мелочах, и сравнивать прогоны с разным
    определением исхода стало бессмысленно. Свели в `Recommendation.outcome()` —
    и через один заход появились новые копии, потому что написать три строки
    быстрее, чем вспомнить про метод.

    Поэтому проверка не логики, а исходников. Ищем именно ПОДПИСЬ дублирования:
    вывод исхода из дельт кандидата, то есть сравнение `abs(d)` с порогом рядом со
    словом исхода. Сравнивать готовую строку с «меняем уставки» никто не
    запрещает — так считают доли и ложные тревоги, и это не копия правила.
    Отдельно разрешён `single_agent_decision`: одноагентная система решает по
    порогу риска, у неё правило ДРУГОЕ, и в этом весь смысл сравнения архитектур.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    contract = root / "src" / "nefte" / "agents" / "schemas.py"
    # abs(...) с порядком 1e-6 и слово исхода в пределах пяти строк друг от друга
    near = re.compile(r"abs\(\w+\)\s*[<>]\s*1e-6")

    def duplicates_rule(text: str) -> bool:
        """Есть ли в файле вывод исхода из дельт — в пределах пяти строк."""
        lines = text.splitlines()
        marks = [i for i, line in enumerate(lines) if near.search(line)]
        words = [i for i, line in enumerate(lines) if "меняем уставки" in line]
        return any(abs(a - b) <= 5 for a in marks for b in words)

    offenders = []
    for path in [*(root / "scripts").glob("*.py"),
                 *(root / "src").rglob("*.py")]:
        if path == contract:
            continue
        if duplicates_rule(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(root)))

    assert not offenders, (
        "правило исхода продублировано в " + ", ".join(offenders)
        + "; используйте Recommendation.outcome()")
