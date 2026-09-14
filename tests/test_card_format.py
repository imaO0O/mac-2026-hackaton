"""Карточка оператора читается как текст, а не как отладочный вывод.

Было: «Эффект: {'сера, мг/кг': 8.12, 'тяжесть режима': 0.73, 'Т95, °C': 345.0,
'сера к бездействию': -1.45, …}». Данные контракта не менялись — меняется только
то, как они показаны.
"""
from __future__ import annotations

from nefte.agents.schemas import effect_text
from tests.test_agents import build_system, make_state

EFFECT = {"сера, мг/кг": 8.12, "тяжесть режима": 0.73, "Т95, °C": 345.0,
          "сера к бездействию": -1.45, "Т95 к бездействию": -0.01, "выпуск, %": -2.37,
          "энергия, %": -0.33, "выпуск смеси, т/ч": 231.9}


def test_effect_is_a_sentence_with_both_sides_of_the_trade_off():
    assert effect_text(EFFECT) == (
        "сера 8.12 мг/кг (-1.45 к бездействию), Т95 345.0 °C (-0.01), выпуск -2.37 %, "
        "энергия -0.33 %, тяжесть режима 0.73, выпуск смеси 231.9 т/ч")


def test_unknown_fields_are_not_dropped():
    assert effect_text({"сера, мг/кг": "н/д", "новое поле": 1}) == "сера н/д мг/кг, новое поле 1"
    assert effect_text({}) == "н/д"


def test_card_has_no_dict_repr():
    rec = build_system().run(make_state(lims=(9.6, 1.0), pak=(9.8, 0.1)))
    text = rec.to_operator_text()
    assert "Эффект:" in text or rec.abstained
    assert "{'" not in text, text
