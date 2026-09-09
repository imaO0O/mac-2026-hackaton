"""Тесты блока смешения: жёсткие проверки ТЗ и физика разбавления."""
from __future__ import annotations

import pytest

from nefte.agents.blending import (
    ADDITIVE_CFPP_EFFECT_C,
    BlendComponent,
    BlendingAgent,
    mix,
)

GODT = BlendComponent(name="ГО ДТ", sulfur_mgkg=8.0, density_15c=836.0,
                      t95_c=345.0, cfpp_c=-6.0, available_tph=250.0)
STRAIGHT = BlendComponent(name="прямогонка", sulfur_mgkg=9000.0, density_15c=848.0,
                          t95_c=350.0, cfpp_c=-8.0, available_tph=130.0)
CLEAN = BlendComponent(name="чистый компонент", sulfur_mgkg=2.0, density_15c=830.0,
                       t95_c=330.0, cfpp_c=-15.0, available_tph=60.0)


def test_sulfur_mixes_linearly_by_mass():
    props = mix([GODT, STRAIGHT], {"ГО ДТ": 0.5, "прямогонка": 0.5})
    assert props["sulfur_mgkg"] == pytest.approx((8.0 + 9000.0) / 2)


def test_fractions_must_sum_to_one():
    """Требование ТЗ: доли компонентов при блендинге дают ровно 100 %."""
    agent = BlendingAgent()
    props = mix([GODT, CLEAN], {"ГО ДТ": 0.5, "чистый компонент": 0.3})
    violations = agent.check(props, {"ГО ДТ": 0.5, "чистый компонент": 0.3})
    assert any("сумма долей" in v for v in violations)


def test_recipe_from_optimizer_always_sums_to_one():
    recipe = BlendingAgent().optimize([GODT, CLEAN])
    assert recipe.fractions_sum() == pytest.approx(1.0, abs=1e-9)


def test_high_sulfur_component_is_excluded_and_explained():
    """Прямогонку в Евро-5 подмешать нельзя — и система обязана это объяснить."""
    recipe = BlendingAgent().optimize([GODT, STRAIGHT])
    assert recipe.fractions["прямогонка"] == pytest.approx(0.0)
    assert any("прямогонка" in note for note in recipe.notes)


def test_max_share_of_high_sulfur_component_is_tiny():
    agent = BlendingAgent()
    share = agent.max_share_of(GODT, STRAIGHT)
    # (10 - 8) / (9000 - 8) ≈ 0.02 %
    assert share == pytest.approx((10.0 - 8.0) / (9000.0 - 8.0), rel=1e-6)
    assert share < 0.001


def test_cleaner_component_allows_more_dilution():
    agent = BlendingAgent()
    dirty = agent.max_share_of(GODT, STRAIGHT)
    lighter = agent.max_share_of(GODT, BlendComponent(name="x", sulfur_mgkg=20.0))
    assert lighter > dirty


def test_spec_violation_makes_recipe_infeasible():
    bad = BlendComponent(name="грязный", sulfur_mgkg=50.0, density_15c=835.0,
                         t95_c=340.0, cfpp_c=-7.0, available_tph=100.0)
    recipe = BlendingAgent().optimize([bad])
    assert not recipe.feasible
    assert any("сера" in v for v in recipe.violations)
    assert recipe.notes


def test_additive_improves_cfpp_within_declared_limit():
    base = mix([GODT], {"ГО ДТ": 1.0})
    dosed = mix([GODT], {"ГО ДТ": 1.0}, additive_ppm=500.0)
    assert dosed["cfpp_c"] == pytest.approx(base["cfpp_c"] + ADDITIVE_CFPP_EFFECT_C)

    half = mix([GODT], {"ГО ДТ": 1.0}, additive_ppm=250.0)
    assert half["cfpp_c"] == pytest.approx(base["cfpp_c"] + ADDITIVE_CFPP_EFFECT_C / 2)


def test_optimizer_prefers_more_throughput():
    """Из двух допустимых рецептур выбирается та, что даёт больший выпуск."""
    recipe = BlendingAgent().optimize([GODT, CLEAN])
    assert recipe.throughput_tph > 0
    assert recipe.feasible
