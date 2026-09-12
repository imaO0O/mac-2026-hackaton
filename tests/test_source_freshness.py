"""Свежесть источника качества меряется ЕГО собственным порогом.

У лаборатории норматив суточный, у поточного анализатора — часовой, и это разные
приборы с разным смыслом «устарело». Пока порог был один на двоих, показание ПАК
десятичасовой давности считалось свежим и шло в решение без всякой пометки.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from nefte.agents.quality import QualityAgent, confidence_parts, fuse_sulfur
from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source


def _state(lims=None, pak=None, frozen=False) -> ProcessState:
    quality = {}
    if lims is not None:
        quality["lims_sulfur_mgkg"] = Measurement(
            value=lims[0], unit="мг/кг", source=Source.LIMS, age_hours=lims[1])
    if pak is not None:
        quality["pak_sulfur_ppm"] = Measurement(
            value=pak[0], unit="мг/кг", source=Source.PAK, age_hours=pak[1],
            is_frozen=frozen)
    return ProcessState(
        ts=datetime(2026, 4, 20, 12, 0),
        telemetry_avt={"T55": 380.0},
        telemetry_ht={"T5": 370.0, "T11": 365.0, "F26": 250.0, "P13": 3.9},
        quality=quality,
        data_quality=DataQuality(missing_share=0.0, usable=True),
    )


def test_fresh_lab_wins():
    m = fuse_sulfur(_state(lims=(6.0, 2.0), pak=(9.5, 0.1)))
    assert m.source is Source.LIMS and m.value == 6.0


def test_stale_lab_gives_way_to_fresh_analyzer():
    m = fuse_sulfur(_state(lims=(6.0, 30.0), pak=(9.5, 0.1)))
    assert m.source is Source.PAK and m.value == 9.5


def test_silent_analyzer_does_not_pass_as_current():
    """Главный случай: прибор не залип, он просто молчит десять часов.

    Раньше проверялось только «не залип», и такое показание подставлялось как
    оперативное. Залипание и молчание — разные отказы, и ловятся они по-разному:
    первое по неизменности значения, второе по возрасту.
    """
    m = fuse_sulfur(_state(lims=(6.0, 30.0), pak=(9.5, 10.0)))
    assert m.source is Source.LIMS
    assert m.is_stale
    assert m.value == 6.0


def test_silent_analyzer_is_still_better_than_nothing():
    """Если лаборатории нет вовсе, устаревший ПАК берём — но с пометкой."""
    m = fuse_sulfur(_state(pak=(9.5, 10.0)))
    assert m.source is Source.PAK
    assert m.is_stale
    assert "устарело" in m.comment


def test_no_source_at_all():
    m = fuse_sulfur(_state())
    assert m.source is Source.NONE and m.value is None


# --------------------------------------------------------------------------- #
# тот же порог должен применяться и к уверенности
# --------------------------------------------------------------------------- #

def test_confidence_uses_the_threshold_of_the_chosen_source():
    """Пять часов для лаборатории — свежо, для поточного анализатора — нет.

    Пока порог брался только лабораторный, переход на ПАК не стоил ничего: его
    возраст сравнивался с сутками и штрафа не давал.
    """
    as_lab = confidence_parts(1.4, Source.LIMS, 5.0, 24.0, True)
    as_analyzer = confidence_parts(1.4, Source.PAK, 5.0, 1.0, True)
    assert as_lab["свежесть"] == pytest.approx(1.0)
    assert as_analyzer["свежесть"] < 0.3


def test_agent_penalises_an_old_analyzer_reading():
    """Сквозная проверка через агент, а не только через функцию."""
    agent = QualityAgent()
    fresh = agent.assess(_state(pak=(9.5, 0.1)))
    old = agent.assess(_state(pak=(9.5, 0.9)))
    assert old.confidence <= fresh.confidence
