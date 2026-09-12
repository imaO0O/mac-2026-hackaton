# -*- coding: utf-8 -*-
"""Второй обязательный показатель не должен нарушаться молча.

Оптимизатор запрещает варианты, ухудшающие Т95. Но когда Т95 УЖЕ за пределом,
запрещать по нему бессмысленно: бездействие не «хуже себя» и проходит как
допустимое. Логика верная — а следствие было такое: карточка писала «Режим
устойчив, риск 8 %» и «текущий режим удовлетворяет ограничениям», показывая рядом
Т95 364 °C при пределе 360 и перечисляя «Т95 ≤ 360 (жёсткое, прогноз)» как
проверенное.

Заметка про это агентом качества СОЗДАВАЛАСЬ («вероятность выхода 72 %») и
никуда не попадала: `q.notes` доходят до карточки только по ветке отказа.

На тестовом периоде так вышло в 10 пригодных срезах из 193, и во ВСЕХ десяти
сера была спокойна — то есть каждый раз карточка называла режим устойчивым.
Медианная вероятность нарушения Т95 в этих срезах 62 %, максимальная 81 %.
"""
from __future__ import annotations

import pytest

from nefte.agents.orchestrator import Orchestrator
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent
from tests.test_agents import make_state
from tests.test_decision_defects import NORMS, _optimizer


def _system(t95_value: float) -> Orchestrator:
    """Система, у которой оценщик Т95 всегда возвращает заданное значение."""
    quality = QualityAgent(t95_fn=lambda state, moves: t95_value)
    reliability = ReliabilityAgent(NORMS)
    return Orchestrator(quality, reliability, _optimizer(reliability), log_runs=False)


def _card(rec) -> str:
    """Всё, что видит оператор, одной строкой."""
    return " | ".join([rec.problem or "", rec.explanation or "",
                       "; ".join(rec.checked_constraints or [])])


def test_quiet_sulfur_does_not_hide_t95_over_the_limit():
    """Сера спокойна, Т95 за пределом — «режим устойчив» говорить нельзя."""
    rec = _system(364.0).run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    card = _card(rec)
    assert not rec.abstained
    assert "Т95" in card and "362" not in card
    assert "устойчив" not in card, (
        f"карточка называет режим устойчивым при Т95 за пределом: {rec.problem}")
    assert "ЗА ПРЕДЕЛОМ" in (rec.problem or ""), (
        f"нарушение второго обязательного показателя не вынесено в заголовок: "
        f"{rec.problem}")


def test_the_constraint_list_stops_promising_what_it_does_not_deliver():
    """«Т95 ≤ 360 (жёсткое)» без оговорки — обещание того, что не выполняется."""
    rec = _system(364.0).run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    t95_lines = [c for c in rec.checked_constraints if "Т95" in c]
    assert t95_lines, "строка про Т95 пропала из списка проверенного"
    assert any("УЖЕ выше" in c for c in t95_lines), (
        f"список проверенного обещает выполнение Т95, хотя он нарушен: {t95_lines}")


def test_a_normal_t95_leaves_the_card_alone():
    """Обратная сторона: пока Т95 в норме, никаких тревог про него быть не должно.

    Без этой половины первый тест проходил бы и на системе, которая кричит про
    Т95 всегда.
    """
    rec = _system(350.0).run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert "ЗА ПРЕДЕЛОМ" not in (rec.problem or "")
    assert any("жёсткое, прогноз" in c for c in rec.checked_constraints
               if "Т95" in c), "нормальный Т95 перестал показываться как жёсткое"


def test_the_trade_off_is_shown_for_both_indicators():
    """Размен виден с обеих сторон: и сера к бездействию, и Т95 к бездействию.

    Раньше показывалась только сера, и по Т95 оператор видел абсолютное число,
    из которого нельзя понять, двинул ли ход разгонку и в какую сторону.
    """
    rec = _system(350.0).run(make_state(lims=(9.5, 1.0), pak=(9.6, 0.1)))
    effect = rec.expected_effect or {}
    if "сера к бездействию" not in effect:
        pytest.skip("в этом срезе бездействие не оценивалось")
    assert "Т95 к бездействию" in effect, (
        f"размен показан наполовину: {sorted(effect)}")
