# -*- coding: utf-8 -*-
"""Поправка к вероятности нарушения Т95: когда применяется и что сохраняет.

Сырая вероятность из нормального приближения завышена в 2–4 раза на всех трёх
периодах и проигрывает по Brier константе (scripts/check_t95_calibration.py).
Поправка подобрана на обучении и принята по правилу, записанному заранее.

Проверяется три обещания:

* без отчёта или без принятой поправки вероятность прежняя, сырая;
* с принятой — поправленная, и порядок моментов не меняется;
* заметка в карточке появляется В ТЕХ ЖЕ точках, что и до поправки: иначе честное
  число стоило бы оператору предупреждения, которое было полезным (из 46 отмеченных
  анализов превышение было в 7 — втрое чаще обычного).
"""
import pytest

from nefte.agents import quality
from nefte.agents.quality import (
    T95_NOTE_RAW_RISK,
    spec_risk_normal,
    t95_note_threshold,
    t95_sigma,
    t95_violation_risk,
)

LIMIT = 360.0
ESTIMATES = [330.0, 345.0, 352.0, 355.0, 357.0, 358.5, 360.0, 363.0, 370.0]


@pytest.fixture
def calibration(monkeypatch):
    original = quality._t95_risk_calibration

    def use(value):
        monkeypatch.setattr(quality, "_t95_risk_calibration", lambda: value)
    yield use
    # кэш настоящей функции мог запомнить отчёт до подмены — сбрасываем
    original.cache_clear()


def test_without_accepted_calibration_risk_is_raw(calibration):
    calibration(None)
    for t in ESTIMATES:
        assert t95_violation_risk(t, 24.0, LIMIT) == spec_risk_normal(t, t95_sigma(24.0), LIMIT)
    assert t95_note_threshold() == T95_NOTE_RAW_RISK


def test_accepted_calibration_lowers_but_keeps_order(calibration):
    calibration((-2.356, 0.428, 0.028))
    raw = [spec_risk_normal(t, t95_sigma(24.0), LIMIT) for t in ESTIMATES]
    fixed = [t95_violation_risk(t, 24.0, LIMIT) for t in ESTIMATES]
    assert fixed == sorted(fixed), "поправка обязана сохранять порядок моментов"
    near_limit = [f < r for f, r in zip(fixed, raw) if 0.05 < r < 0.95]
    assert near_limit and all(near_limit), "в рабочем диапазоне поправка снижает завышение"


def test_note_fires_at_the_same_points_as_before(calibration):
    """Порог заметки переводится через ту же поправку — набор точек не меняется."""
    calibration(None)
    before = [t95_violation_risk(t, age, LIMIT) > t95_note_threshold()
              for t in ESTIMATES for age in (5.0, 40.0, 150.0)]
    calibration((-2.356, 0.428, 0.028))
    after = [t95_violation_risk(t, age, LIMIT) > t95_note_threshold()
             for t in ESTIMATES for age in (5.0, 40.0, 150.0)]
    assert before == after
    assert any(after) and not all(after), "синтетика не та: нужны точки по обе стороны"
