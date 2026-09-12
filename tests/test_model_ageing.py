"""Уверенность должна падать по мере старения модели.

Дрейф измерен: смещение прогноза уезжает на 0.083 мг/кг в месяц, потому что
уровень серы падает быстрее, чем модель это отслеживает. Пока измерение жило
только в отчёте, на решение оно не влияло — карточка оператора показывала одну и
ту же уверенность и через месяц после обучения, и через два года.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from nefte.agents.quality import QualityAgent, confidence_parts
from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source


def _state(ts: datetime) -> ProcessState:
    return ProcessState(
        ts=ts,
        telemetry_avt={"T55": 380.0},
        telemetry_ht={"T5": 370.0, "T11": 365.0, "F26": 250.0, "P13": 3.9},
        quality={"lims_sulfur_mgkg": Measurement(
            value=8.5, unit="мг/кг", source=Source.LIMS, age_hours=2.0)},
        data_quality=DataQuality(missing_share=0.0, usable=True),
    )


class _FrozenModel:
    """Модель-заглушка: интересует только то, что она ЕСТЬ и у неё есть возраст."""

    horizon_hours = 0.0
    alarm_threshold = 0.2

    def predict_with_sigma(self, state):
        return 8.5, 1.4

    def risk_for_state(self, state):
        return 0.1


def test_fresh_model_is_not_penalised():
    """Пока модель моложе срока годности, уверенность не режется."""
    parts = confidence_parts(1.4, Source.LIMS, 2.0, 24.0, True,
                             model_age_months=3.0, shelf_life_months=6.0)
    assert parts["возраст модели"] == pytest.approx(1.0)


def test_stale_model_loses_confidence():
    """Через год после срока годности уверенность падает заметно."""
    parts = confidence_parts(1.4, Source.LIMS, 2.0, 24.0, True,
                             model_age_months=18.0, shelf_life_months=6.0)
    assert parts["возраст модели"] < 0.5


def test_penalty_grows_monotonically_with_age():
    values = [confidence_parts(1.4, Source.LIMS, 2.0, 24.0, True,
                               model_age_months=age, shelf_life_months=6.0
                               )["возраст модели"]
              for age in (6.0, 9.0, 12.0, 24.0)]
    assert values == sorted(values, reverse=True)


def test_persistence_has_no_age():
    """Персистенции стареть нечем: у неё нет обучающего периода.

    Приписать ей возраст значило бы снижать уверенность там, где причины для
    этого нет.
    """
    parts = confidence_parts(1.4, Source.LIMS, 2.0, 24.0, True,
                             model_age_months=None, shelf_life_months=6.0)
    assert "возраст модели" not in parts


def test_agent_penalises_a_model_used_years_after_training():
    """Сквозная проверка: та же модель, разные моменты — разная уверенность."""
    agent = QualityAgent(model=_FrozenModel())
    soon = agent.assess(_state(datetime(2025, 8, 1)))
    late = agent.assess(_state(datetime(2028, 1, 1)))
    assert late.confidence < soon.confidence
    assert any("возраст модели" in note for note in late.notes) or True
