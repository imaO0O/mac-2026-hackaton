"""Т95 как жёсткое ограничение оптимизатора.

Организаторы назвали Т95 обязательным показателем наравне с серой. Это не
формальность: по исправленной формуле виртуального анализатора температура Р-202
входит в Т95 с коэффициентом 0.50, то есть каждое повышение температуры ради серы
тянет Т95 вверх. Запас до предела 360 °C на выданных данных бывает в пять
градусов. Без проверки система чинила бы одно за счёт другого и не знала об этом.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from nefte.agents.optimizer import (
    OptimizerAgent,
    default_t95_estimator,
    linear_surrogate,
)
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent, SeverityNorms
from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source


def make_state(t95: float | None = 355.0, t6: float = 362.0) -> ProcessState:
    quality = {
        "lims_sulfur_mgkg": Measurement(value=9.6, unit="мг/кг", source=Source.LIMS,
                                        age_hours=2.0),
        "pak_sulfur_ppm": Measurement(value=9.7, unit="мг/кг", source=Source.PAK,
                                      age_hours=0.1),
    }
    if t95 is not None:
        quality["lims_t95_c"] = Measurement(value=t95, unit="°C", source=Source.LIMS,
                                            age_hours=3.0)
    return ProcessState(
        ts=datetime(2026, 4, 20, 12, 0),
        telemetry_avt={"T55": 380.0},
        telemetry_ht={"T5": 370.0, "T11": 365.0, "F26": 250.0, "P13": 3.9,
                      "W10": 2.8, "T6": t6, "F9": 120.0, "F2": 50000.0},
        quality=quality,
        data_quality=DataQuality(missing_share=0.0, usable=True),
    )


def build_optimizer() -> OptimizerAgent:
    bounds = {"T5": (365.0, 375.0), "T6": (355.0, 385.0), "T11": (360.0, 370.0)}
    return OptimizerAgent(
        bounds=bounds,
        surrogate=linear_surrogate({"T5": -0.15, "T6": -0.15, "T11": -0.15}))


# --------------------------------------------------------------------------- #
# сама оценка
# --------------------------------------------------------------------------- #

def test_current_t95_is_the_lab_value_not_the_formula():
    """Уровень берём из лаборатории. У формулы MAE 5.5 °C — на предел не годится."""
    estimator = default_t95_estimator()
    assert estimator(make_state(t95=355.0), {}) == pytest.approx(355.0)


def test_raising_reactor_temperature_raises_t95():
    """Коэффициент 0.50: +2 °C по Р-202 дают +1 °C к Т95."""
    estimator = default_t95_estimator()
    state = make_state(t95=355.0, t6=362.0)
    assert estimator(state, {"T6": 364.0}) == pytest.approx(356.0, abs=0.05)
    assert estimator(state, {"T6": 372.0}) == pytest.approx(360.0, abs=0.05)


def test_no_lab_analysis_means_unknown_not_zero():
    """Ноль прошёл бы проверку предела, и вариант объявили бы годным."""
    estimator = default_t95_estimator()
    assert estimator(make_state(t95=None), {"T6": 380.0}) is None


# --------------------------------------------------------------------------- #
# ограничение в оптимизаторе
# --------------------------------------------------------------------------- #

def _evaluate(state, optimizer):
    quality = QualityAgent().assess(state)
    norms = SeverityNorms(bounds={"wabt": (355.0, 375.0), "W10": (1.0, 4.0),
                                  "T55": (370.0, 395.0)})
    reliability = ReliabilityAgent(norms).assess(state)
    return optimizer.evaluate(state, optimizer.generate(state, reliability),
                              quality, reliability)


def test_candidate_that_breaks_t95_is_rejected():
    """Тонкий запас по Т95 делает подъём температуры недопустимым.

    Порог здесь не случайный. За один цикл температуре разрешено уйти на 2 °C
    (limits.max_step_per_cycle), а Т95 отвечает на это половиной градуса на
    градус — то есть один цикл добавляет к Т95 не больше 1 °C. Значит, ограничение
    вообще способно сработать только при запасе меньше градуса; на большем запасе
    опасность накапливается ЦИКЛАМИ, и ловится она в имитации замкнутого контура
    (scripts/run_simulation.py), а не здесь.
    """
    optimizer = build_optimizer()
    evaluated = _evaluate(make_state(t95=359.8, t6=362.0), optimizer)
    over = [c for c in evaluated
            if c.predicted_quality.get("product_t95_c", 0.0) > 360.0]
    assert over, "нужен хотя бы один вариант, выводящий Т95 за предел"
    assert all(not c.feasible for c in over)
    assert all(any("Т95" in v for v in c.violations) for c in over)


def test_t95_is_reported_for_every_candidate():
    """Показатель обязан быть виден оператору, а не только влиять на отсев."""
    evaluated = _evaluate(make_state(t95=355.0), build_optimizer())
    assert all("product_t95_c" in c.predicted_quality for c in evaluated)


def test_wide_t95_margin_blocks_nothing():
    """Обратная сторона: ограничение не должно срабатывать без причины.

    Если запас велик, ни один вариант не отсеивается по Т95 — иначе проверка
    превратилась бы в тормоз, который просто запрещает трогать температуру.
    """
    evaluated = _evaluate(make_state(t95=330.0), build_optimizer())
    assert not any(any("Т95" in v for v in c.violations) for c in evaluated)


def test_already_over_limit_does_not_forbid_improving_moves():
    """Т95 уже за пределом — запрещать снижение температуры бессмысленно.

    Иначе система, единожды выйдя за предел, отказывалась бы от любого действия,
    в том числе от того, которое возвращает продукт в норму.
    """
    optimizer = build_optimizer()
    evaluated = _evaluate(make_state(t95=365.0, t6=370.0), optimizer)
    lowering = [c for c in evaluated
                if c.moves.get("T6", 370.0) < 369.0 and "T6" in c.deltas]
    assert lowering, "нужен вариант со снижением температуры Р-202"
    assert all(not any("Т95" in v for v in c.violations) for c in lowering)
