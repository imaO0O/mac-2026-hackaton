"""Кэш уровня в кинетическом суррогате: быстрее, но ровно так же.

Оптимизатор зовёт суррогат для каждого из двухсот вариантов с одним и тем же
срезом. Уровень серы берётся у модели и от уставок не зависит по построению —
значит, двести раз считалось одно и то же число тремя бустингами. Кэш убирает
эту работу; тесты следят, чтобы он не убрал заодно и смысл.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source
from nefte.models.kinetics import make_kinetic_surrogate


class _CountingModel:
    """Модель, которая считает, сколько раз её спросили."""

    def __init__(self, level: float = 9.0):
        self.level = level
        self.calls = 0

    def predict_with_sigma(self, state):
        self.calls += 1
        return self.level, 1.4


def _state(ts: datetime) -> ProcessState:
    return ProcessState(
        ts=ts,
        telemetry_avt={},
        telemetry_ht={"T5": 370.0, "T6": 365.0, "T11": 366.0,
                      "F26": 250.0, "P13": 3.9, "F25": 13600.0},
        quality={"lims_feed_sulfur_mgkg": Measurement(
            value=9000.0, unit="мг/кг", source=Source.LIMS, age_hours=5.0)},
        data_quality=DataQuality(missing_share=0.0, usable=True),
    )


def test_model_is_asked_once_per_moment_not_once_per_candidate():
    """Ради чего кэш и заведён: двести вариантов — один запрос к модели."""
    model = _CountingModel()
    surrogate = make_kinetic_surrogate(model)
    state = _state(datetime(2026, 6, 1, 12, 0))
    for delta in range(20):
        surrogate(state, {"T5": 370.0 + delta * 0.1})
    assert model.calls == 1


def test_new_moment_invalidates_the_cache():
    """Ключ — сам момент среза, поэтому устареть кэш не может."""
    model = _CountingModel()
    surrogate = make_kinetic_surrogate(model)
    surrogate(_state(datetime(2026, 6, 1, 12, 0)), {})
    surrogate(_state(datetime(2026, 6, 1, 16, 0)), {})
    assert model.calls == 2


def test_candidates_still_differ_from_each_other():
    """Главная опасность кэша — заморозить выход целиком.

    Уровень общий, а приращение обязано считаться для каждого варианта: иначе
    оптимизатору стало бы не из чего выбирать, и это тот самый дефект, ради
    которого кинетику вообще добавляли.
    """
    surrogate = make_kinetic_surrogate(_CountingModel())
    state = _state(datetime(2026, 6, 1, 12, 0))
    cold = surrogate(state, {"T5": 368.0})["product_sulfur_mgkg"]
    hold = surrogate(state, {})["product_sulfur_mgkg"]
    hot = surrogate(state, {"T5": 374.0})["product_sulfur_mgkg"]
    assert hot < hold < cold, "выше температура — ниже сера"


def test_cached_and_uncached_agree():
    """Кэш обязан быть чистой оптимизацией: те же входы — те же числа."""
    state = _state(datetime(2026, 6, 1, 12, 0))
    first = make_kinetic_surrogate(_CountingModel())
    second = make_kinetic_surrogate(_CountingModel())
    moves = {"T5": 372.0, "F26": 240.0}
    # у второго суррогата кэш прогрет другим вариантом — результат не должен зависеть
    second(state, {"T5": 360.0})
    assert (first(state, moves)["product_sulfur_mgkg"]
            == pytest.approx(second(state, moves)["product_sulfur_mgkg"]))
