"""Измерение уже выше предела, а риск модели низкий: карточка не противоречит себе.

Было: «ФАКТ ВНЕ СПЕЦИФИКАЦИИ: lims 10.70 мг/кг… Режим устойчив, риск 8 %» и в
объяснении «Текущий режим удовлетворяет ограничениям, изменение уставок не
требуется». На часовом прогоне теста так было в 347 моментах из 5233.

Решение «держим» при этом проверено и оставлено: после пробы выше предела
следующая выше в 17–26 % случаев по периодам, а вероятность модели в таких
моментах не занижена. Тесты проверяют только карточку: она называет расхождение
измерения и прогноза и не называет режим устойчивым.
"""
from __future__ import annotations

from nefte.agents.orchestrator import Orchestrator
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent
from tests.test_agents import make_state
from tests.test_decision_defects import NORMS, _optimizer


class _CalmModel:
    """Модель, которая по текущим данным ждёт 8 мг/кг и даёт низкий риск."""
    horizon_hours = 0.0
    alarm_threshold = 0.18

    def predict_with_sigma(self, state):
        return 8.0, 0.6

    def risk_for_state(self, state):
        return 0.05


def _system() -> Orchestrator:
    reliability = ReliabilityAgent(NORMS)
    quality = QualityAgent(model=_CalmModel(), t95_fn=lambda state, moves: None)
    return Orchestrator(quality, reliability, _optimizer(reliability), log_runs=False)


def _card(rec) -> str:
    return " | ".join([rec.problem or "", rec.explanation or ""])


def test_fresh_lab_over_limit_with_low_risk_is_explained_not_called_stable():
    rec = _system().run(make_state(lims=(10.7, 6.0), pak=(8.1, 0.1)))
    card = _card(rec)
    assert rec.outcome() == "держим режим", rec.outcome()
    assert "ФАКТ ВНЕ СПЕЦИФИКАЦИИ" in rec.problem
    assert "устойчив" not in card, f"режим назван устойчивым при факте вне спецификации: {card}"
    assert "удовлетворяет ограничениям" not in card
    assert "Последнее измерение выше предела" in rec.problem
    assert "8.00" in rec.problem, "прогноз, по которому держим режим, не назван"
    assert "6 ч назад" in rec.explanation
    assert "в следующей пробе" in rec.explanation


def test_calm_measurement_keeps_the_usual_card():
    """Обратная сторона: без факта вне спецификации карточка прежняя."""
    rec = _system().run(make_state(lims=(8.0, 6.0), pak=(8.1, 0.1)))
    assert "Последнее измерение выше предела" not in (rec.problem or "")
    assert "устойчив" in (rec.problem or "")
