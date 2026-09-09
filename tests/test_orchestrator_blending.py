"""Блок смешения внутри цикла оркестратора.

Проверяем главное: рецептура считается для ТОГО режима, который рекомендован,
сумма долей попадает в проверенные ограничения, а смешение не выдаётся за способ
вытянуть продукт, уже вышедший за спецификацию.
"""
from __future__ import annotations

import pytest

from nefte.agents.blending import BlendingAgent
from nefte.agents.optimizer import OptimizerAgent, linear_surrogate
from nefte.agents.orchestrator import Orchestrator
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent, SeverityNorms
from nefte.agents.schemas import BlendComponent
from tests.test_agents import make_state

GODT = BlendComponent(name="ГО ДТ", sulfur_mgkg=8.6, density_15c=836.0,
                      t95_c=345.0, cfpp_c=-9.0, available_tph=250.0)
STRAIGHT = BlendComponent(name="Прямогонная фр. 290-350", sulfur_mgkg=8700.0,
                          density_15c=840.0, t95_c=352.0, cfpp_c=-8.0,
                          available_tph=130.0)


def build_system(components=None) -> Orchestrator:
    norms = SeverityNorms(bounds={"wabt": (355.0, 375.0), "W10": (1.0, 4.0),
                                  "T55": (370.0, 395.0)})
    bounds = {"T5": (365.0, 375.0), "T11": (360.0, 370.0),
              "F26": (200.0, 300.0), "P13": (3.5, 4.2)}
    opt = OptimizerAgent(bounds=bounds,
                         surrogate=linear_surrogate({"T5": -0.15, "T11": -0.15, "P13": -0.5}))
    return Orchestrator(QualityAgent(), ReliabilityAgent(norms), opt, log_runs=False,
                        blending=BlendingAgent(),
                        components_fn=lambda ts: list(components
                                                      if components is not None
                                                      else [GODT, STRAIGHT]))


def test_recipe_is_attached_and_sums_to_one():
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert rec.blend is not None
    assert rec.blend.fractions_sum() == pytest.approx(1.0, abs=1e-9)
    assert any("доли компонентов смешения" in c for c in rec.checked_constraints)


def test_recipe_is_built_on_the_forecast_not_on_the_last_analysis():
    """Рецептура относится к рекомендуемому режиму, а не к прошедшему."""
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    predicted = rec.action.predicted_quality["product_sulfur_mgkg"]
    assert rec.blend.basis == "forecast"
    assert rec.blend.basis_sulfur_mgkg == pytest.approx(predicted)
    # прогноз, а не 8.6 мг/кг из последнего анализа компонента
    assert rec.blend.basis_sulfur_mgkg != pytest.approx(GODT.sulfur_mgkg)


def test_straight_run_never_enters_euro5_blend():
    """Ключевой вывод docs/BLENDING.md: разбавлять прямогонкой нельзя."""
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert rec.blend.fractions[STRAIGHT.name] == pytest.approx(0.0)
    assert any(STRAIGHT.name in note for note in rec.blend.notes)


def test_blending_does_not_rescue_off_spec_product():
    """Продукт за пределом — допустимой рецептуры нет, и это сказано вслух."""
    rec = build_system().run(make_state(lims=(12.0, 1.0), pak=(12.2, 0.1)))
    assert rec.blend is not None and not rec.blend.feasible
    assert rec.blend.violations
    assert "Смешением это не компенсируется" in rec.explanation


def test_no_recipe_when_data_is_not_usable():
    """Нет достоверного качества — нет и рецептуры: считать её не на чем."""
    rec = build_system().run(make_state(usable=False))
    assert rec.abstained and rec.blend is None


def test_recipe_appears_in_operator_text():
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    text = rec.to_operator_text()
    assert "Смешение" in text and "сумма долей 100.0 %" in text


def test_system_without_blending_agent_keeps_working():
    """Контракт остаётся совместимым: без блока смешения цикл прежний."""
    norms = SeverityNorms(bounds={"wabt": (355.0, 375.0)})
    opt = OptimizerAgent(bounds={"T5": (365.0, 375.0)},
                         surrogate=linear_surrogate({"T5": -0.15}))
    system = Orchestrator(QualityAgent(), ReliabilityAgent(norms), opt, log_runs=False)
    rec = system.run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert rec.blend is None and not rec.abstained
