"""Нормативы по маркам ДТ — ответ организаторов 15.09.

* ДТ с гидроочистки: плотность 820–845, цетановое число не нормируется;
* ДТ летнее товарное: плотность 820–845, ЦЧ не ниже 51;
* ДТ зимнее товарное: плотность 800–845, ЦЧ не ниже 49.

До ответа норматив ЦЧ 51 стоял как допущение по EN 590 и применялся к ГО ДТ, у
которого его нет. Здесь закреплено, что проверка смеси берёт норматив марки.
"""
from __future__ import annotations

import pytest

from nefte.agents.blending import BlendComponent, BlendingAgent, grade_spec, mix
from nefte.config import load_config

LOW_CETANE = BlendComponent(name="ГО ДТ", sulfur_mgkg=8.0, density_15c=810.0,
                            t95_c=345.0, cfpp_c=-6.0, cetane_number=50.0,
                            available_tph=250.0)


def _violations(grade: str) -> list[str]:
    agent = BlendingAgent(grade=grade)
    fractions = {"ГО ДТ": 1.0}
    return agent.check(mix([LOW_CETANE], fractions), fractions)


def test_top_level_spec_is_the_blend_grade():
    """Значения верхнего уровня обязаны совпадать с маркой смешения — иначе два источника."""
    spec = load_config()["spec"]
    grade = spec["grades"][spec["blend_grade"]]
    assert [spec["density_15c_kgm3"]["min"], spec["density_15c_kgm3"]["max"]] == \
        grade["density_15c_kgm3"]
    assert spec["cetane_number"]["min"] == grade["cetane_number_min"]
    assert spec["cetane_number"]["assumption"] is False


def test_grades_as_answered_by_organizers():
    grades = load_config()["spec"]["grades"]
    assert grades["hydrotreated"]["cetane_number_min"] is None
    assert grades["summer"]["cetane_number_min"] == 51.0
    assert grades["winter"]["cetane_number_min"] == 49.0
    assert grades["winter"]["density_15c_kgm3"] == [800.0, 845.0]
    assert grades["summer"]["density_15c_kgm3"] == [820.0, 845.0]
    assert grades["hydrotreated"]["density_15c_kgm3"] == [820.0, 845.0]


def test_summer_grade_rejects_cetane_50_and_density_810():
    violations = _violations("summer")
    assert any("цетановое число 50.0 ниже 51.0" in v for v in violations)
    assert any("плотность 810.0" in v for v in violations)
    assert all("допущение" not in v for v in violations if "цетан" in v or "плотн" in v)


def test_winter_grade_accepts_the_same_blend():
    assert _violations("winter") == []


def test_hydrotreated_product_has_no_cetane_norm():
    agent = BlendingAgent(grade="hydrotreated")
    fractions = {"ГО ДТ": 1.0}
    props = mix([LOW_CETANE.model_copy(update={"density_15c": 830.0})], fractions)
    assert agent.check(props, fractions) == []
    # и отсутствие анализа ЦЧ не делает рецептуру неподтверждённой
    no_cetane = {k: v for k, v in props.items() if k != "cetane_number"}
    assert "цетановое число" not in agent.uncertified(no_cetane)
    # и присадку ради ЦЧ назначать не за что
    assert agent._min_improver(props) == 0.0


def test_unknown_grade_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        grade_spec(load_config()["spec"], "arctic")
