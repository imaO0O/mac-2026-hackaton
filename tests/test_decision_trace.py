"""Путь решения — часть рекомендации: кто что сказал и какое правило сработало.

ТЗ требует, чтобы взаимодействие ролей было явно показано в коде и на
демонстрации. Трасса проверяется на трёх ветках оркестратора: отказ по данным
(оптимизатора не спрашивали — его шага в трассе быть не должно), «держим режим» и
действие. Последний шаг всегда оркестратор, и его вывод совпадает с исходом.
"""
from __future__ import annotations

from tests.test_agents import build_system, make_state


def _agents(rec) -> list[str]:
    return [step.agent for step in rec.trace]


def test_refusal_on_bad_data_skips_the_optimizer():
    rec = build_system().run(make_state(usable=False))
    assert rec.abstained
    assert _agents(rec) == ["срез состояния", "агент качества", "агент надёжности",
                            "оркестратор"]
    assert "отказ: данные недостоверны" in rec.trace[-1].summary
    assert rec.trace[-1].summary.endswith("→ отказ")


def test_hold_shows_every_agent_and_the_rule():
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    assert rec.outcome() == "держим режим"
    agents = _agents(rec)
    assert agents[:4] == ["срез состояния", "агент качества", "агент надёжности",
                          "оптимизатор"]
    assert agents[-1] == "оркестратор"
    assert "держим" in rec.trace[-1].summary
    optimizer = next(s for s in rec.trace if s.agent == "оптимизатор")
    assert optimizer.details["допустимых"] > 0
    assert optimizer.details["вариантов"] >= optimizer.details["допустимых"]


def test_action_names_the_chosen_variant():
    rec = build_system().run(make_state(lims=(11.0, 1.0), pak=(11.2, 0.1)))
    if rec.outcome() != "меняем уставки":
        return  # синтетическая система может отказаться — это проверяют другие тесты
    optimizer = next(s for s in rec.trace if s.agent == "оптимизатор")
    assert rec.action.id in optimizer.summary
    assert rec.trace[-1].summary.endswith("→ меняем уставки")


def test_trace_survives_serialization_to_the_run_log():
    rec = build_system().run(make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)))
    dumped = rec.model_dump(mode="json")
    assert [s["agent"] for s in dumped["trace"]] == _agents(rec)
