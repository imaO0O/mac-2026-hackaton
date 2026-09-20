"""Гашение превышения в резервуаре — практика установки, а не наша выдумка.

Технолог на сессии 11.09 (`docs/transcripts/qa_2026-09-11.txt`, 30:26) описал, что
делают с партией, у которой сера вышла за 10 мг/кг: её сливают в резервуар и гасят
топливом, у которого есть запас. Партия, признанная некондицией, идёт на повторную
переработку и стоит «в 50–100 раз дороже, чем просто получить запас по качеству»
(15.09, 07:26).

До этого система на превышение отвечала только «смешением это не компенсируется» —
верно для прямогонки, но не для того, что делают на установке. Тесты держат три
вещи: расчёт доли линеен и совпадает с ручной проверкой, подсказка появляется
ровно тогда, когда продукт за пределом, и невозможность гасить названа прямо.
"""
from __future__ import annotations

import pytest

from nefte.agents.blending import tank_rescue
from nefte.config import load_config


def test_share_is_the_linear_mix_that_reaches_the_target():
    """Партия 11, резервуар 7.5, цель 9 — доля партии 3/7."""
    out = tank_rescue(11.0, 10.0, 7.5, 1.0)
    assert out["возможно"] is True
    assert out["доля партии"] == pytest.approx((9.0 - 7.5) / (11.0 - 7.5), abs=1e-4)
    # смесь в этой пропорции действительно приходит в цель
    share = out["доля партии"]
    # доля округлена до сотых долей процента — этого хватает и для карточки
    assert share * 11.0 + (1 - share) * 7.5 == pytest.approx(9.0, abs=1e-3)


def test_dirtier_batch_needs_more_tank():
    """Чем грязнее партия, тем меньше её можно принять — это и есть цена ошибки."""
    mild = tank_rescue(11.0, 10.0, 7.5, 1.0)["на тонну партии нужно тонн резервуара"]
    harsh = tank_rescue(18.4, 10.0, 7.5, 1.0)["на тонну партии нужно тонн резервуара"]
    assert harsh > mild * 3


def test_tank_without_margin_cannot_rescue():
    out = tank_rescue(11.0, 10.0, 9.5, 1.0)
    assert out["возможно"] is False
    assert "нет запаса" in out["почему"]
    assert out["доля партии"] == 0.0


def test_batch_within_spec_needs_no_rescue():
    out = tank_rescue(8.0, 10.0, 7.5, 1.0)
    assert out["доля партии"] == 1.0
    assert "гасить нечего" in out["почему"]


def test_config_keeps_the_assumption_visible():
    """Сера резервуара — допущение из примера технолога, и это должно быть видно."""
    cfg = load_config()["blending"]
    assert cfg["tank_reserve_assumption"] is True
    assert 7.0 <= cfg["tank_reserve_sulfur_mgkg"] <= 8.0
    # запас берём из названного технологического: 1–2 мг/кг
    assert 1.0 <= cfg["tank_rescue_margin_mgkg"] <= 2.0
