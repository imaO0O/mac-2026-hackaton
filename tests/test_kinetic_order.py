"""Порядок реакции в кинетическом суррогате.

Порядок добавлен ради одной проверки — «что, если сера отвечает на температуру
слабее, чем думает система» (scripts/check_kinetic_order.py). Рабочий контур от
него меняться не должен, и тесты держат именно это: первый порядок совпадает с
прежней формулой, без изменения уставок любой порядок возвращает уровень модели,
а более высокий порядок даёт более слабый отклик того же знака.
"""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source
from nefte.models.kinetics import arrhenius_factor, make_kinetic_surrogate

FEED_MGKG = 9000.0
LEVEL = 8.0


class _Model:
    def predict_with_sigma(self, state):
        return LEVEL, 1.4


def _state() -> ProcessState:
    return ProcessState(
        ts=datetime(2026, 6, 1, 12, 0),
        telemetry_avt={},
        telemetry_ht={"T5": 370.0, "T6": 365.0, "T11": 366.0,
                      "F26": 250.0, "P13": 3.9, "F25": 13600.0},
        quality={"lims_feed_sulfur_mgkg": Measurement(
            value=FEED_MGKG, unit="мг/кг", source=Source.LIMS, age_hours=5.0)},
        data_quality=DataQuality(missing_share=0.0, usable=True),
    )


def _hotter(state: ProcessState, by: float) -> dict[str, float]:
    return {t: state.telemetry_ht[t] + by for t in ("T5", "T6", "T11")}


def test_first_order_is_the_old_formula():
    """Рабочий порядок обязан давать ровно то, что давала формула до параметра."""
    state = _state()
    got = make_kinetic_surrogate(_Model())(state, _hotter(state, 2.0))["product_sulfur_mgkg"]
    wabt = (370.0 + 365.0 + 366.0) / 3
    tau = math.log(FEED_MGKG / LEVEL) * arrhenius_factor(wabt, wabt + 2.0)
    assert got == pytest.approx(FEED_MGKG * math.exp(-tau), rel=1e-12)


@pytest.mark.parametrize("order", [1.0, 1.5, 2.0])
def test_no_move_returns_the_model_level(order):
    """Без изменения уставок физика не должна сдвигать уровень ни при каком порядке."""
    state = _state()
    fn = make_kinetic_surrogate(_Model(), order=order)
    assert fn(state, {})["product_sulfur_mgkg"] == pytest.approx(LEVEL, rel=1e-9)


def test_higher_order_gives_weaker_response_of_the_same_sign():
    state = _state()
    deltas = []
    for order in (1.0, 1.5, 2.0):
        fn = make_kinetic_surrogate(_Model(), order=order)
        deltas.append(fn(state, _hotter(state, 1.0))["product_sulfur_mgkg"] - LEVEL)
    assert all(d < 0 for d in deltas), "выше температура — ниже сера при любом порядке"
    assert abs(deltas[0]) > abs(deltas[1]) > abs(deltas[2])
    # порядок величины, названный в документации: около −19 %, −6 % и −3 % на градус
    rel = [d / LEVEL for d in deltas]
    assert rel[0] == pytest.approx(-0.19, abs=0.03)
    assert rel[1] == pytest.approx(-0.06, abs=0.015)
    assert rel[2] == pytest.approx(-0.03, abs=0.01)
