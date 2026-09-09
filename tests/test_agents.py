"""Тесты контракта агентов: система обязана уметь отказываться от рекомендации."""
from __future__ import annotations

from datetime import datetime

import pytest

from nefte.agents.optimizer import OptimizerAgent, linear_surrogate
from nefte.agents.orchestrator import Orchestrator
from nefte.agents.quality import QualityAgent, fuse_sulfur, spec_risk_normal
from nefte.agents.reliability import ReliabilityAgent, SeverityNorms
from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source


def make_state(*, lims=(6.0, 2.0), pak=(6.2, 0.1), frozen=False, usable=True) -> ProcessState:
    quality = {}
    if lims:
        quality["lims_sulfur_mgkg"] = Measurement(
            value=lims[0], unit="мг/кг", source=Source.LIMS, age_hours=lims[1],
            is_stale=lims[1] > 24)
    if pak:
        quality["pak_sulfur_ppm"] = Measurement(
            value=pak[0], unit="мг/кг", source=Source.PAK, age_hours=pak[1], is_frozen=frozen)
    return ProcessState(
        ts=datetime(2026, 4, 20, 12, 0),
        telemetry_avt={"T55": 380.0},
        telemetry_ht={"T5": 370.0, "T11": 365.0, "F26": 250.0, "P13": 3.9, "W10": 2.8, "T6": 362.0},
        quality=quality,
        data_quality=DataQuality(missing_share=0.0, usable=usable),
    )


def build_system(cfg=None) -> Orchestrator:
    norms = SeverityNorms(bounds={"wabt": (355.0, 375.0), "W10": (1.0, 4.0), "T55": (370.0, 395.0)})
    bounds = {"T5": (365.0, 375.0), "T11": (360.0, 370.0),
              "F26": (200.0, 300.0), "P13": (3.5, 4.2)}
    opt = OptimizerAgent(bounds=bounds,
                         surrogate=linear_surrogate({"T5": -0.15, "T11": -0.15, "P13": -0.5}))
    return Orchestrator(QualityAgent(), ReliabilityAgent(norms), opt, log_runs=False)


def test_lims_wins_over_pak_when_fresh():
    m = fuse_sulfur(make_state(lims=(6.0, 2.0), pak=(9.5, 0.1)))
    assert m.source is Source.LIMS and m.value == 6.0


def test_frozen_pak_is_not_used_as_fact():
    """Случай 2026-04-15…27: ПАК завис на 18.4, лаборатория показывает норму."""
    m = fuse_sulfur(make_state(lims=(3.4, 10.0), pak=(18.4, 0.1), frozen=True))
    assert m.source is Source.LIMS
    assert m.value == pytest.approx(3.4)


def test_stale_lims_with_broken_pak_is_flagged():
    m = fuse_sulfur(make_state(lims=(3.4, 100.0), pak=(18.4, 0.1), frozen=True))
    assert m.is_stale and m.source is Source.LIMS


def test_spec_risk_grows_with_prediction():
    assert spec_risk_normal(6.0, 1.7, 10.0) < 0.05
    assert spec_risk_normal(10.0, 1.7, 10.0) == pytest.approx(0.5, abs=0.01)
    assert spec_risk_normal(14.0, 1.7, 10.0) > 0.95


def test_system_abstains_on_unusable_data():
    rec = build_system().run(make_state(usable=False))
    assert rec.abstained and rec.abstain_reason


def test_system_holds_when_process_is_stable():
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert not rec.abstained
    assert rec.action is not None and rec.action.id == "hold"


def test_recommendation_is_explainable():
    rec = build_system().run(make_state(lims=(9.6, 1.0), pak=(9.8, 0.1)))
    text = rec.to_operator_text()
    assert "Уверенность" in text or rec.abstained
    if not rec.abstained:
        assert rec.checked_constraints
        assert rec.explanation


def test_source_of_decision_is_reported_not_first_measurement_in_slice():
    """В срезе есть и сера продукта, и сера сырья — показывать надо ту, по которой решили."""
    state = make_state(lims=(5.0, 1.0), pak=(5.2, 0.1))
    state.quality["lims_feed_sulfur_mgkg"] = Measurement(
        value=9500.0, unit="мг/кг", source=Source.LIMS, age_hours=200.0,
        comment="сера сырья гидроочистки")
    rec = build_system().run(state)
    assert rec.state_summary["sulfur_source"] == "lims"
    assert QualityAgent().assess(state).source is Source.LIMS


def test_model_without_lims_and_pak_is_declared_as_vak():
    """Модель без обоих измерений — виртуальный анализатор, третий приоритет ТЗ.

    Работать по нему можно, выдавать его за измерение — нельзя.
    """
    class FakeModel:
        horizon_hours = 0.0
        alarm_threshold = 0.2

        def predict_with_sigma(self, state):
            return 7.0, 1.7

    state = make_state(lims=None, pak=None)
    with_sources = QualityAgent(model=FakeModel()).assess(
        make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    without = QualityAgent(model=FakeModel()).assess(state)

    assert without.source is Source.VAK
    assert without.confidence < with_sources.confidence
    assert any("ВАК" in note for note in without.notes)


def test_without_model_and_without_sources_there_is_no_forecast():
    out = QualityAgent().assess(make_state(lims=None, pak=None))
    assert out.source is Source.NONE and out.confidence == 0.0
    assert not out.predictions
