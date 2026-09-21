"""Блок «что дальше» и строка возврата в норму: что в них обязано быть правдой.

Эти строки ничего не решают — они обещают. Обещание проверяется лабораторией через
несколько часов, и неверное обещание дороже молчания: оператор перестанет читать
карточку целиком. Тесты держат то, от чего зависит доверие к строке: диапазон
следующего анализа строится из ТОГО ЖЕ интервала, что показан в карточке; тонны
считаются от реального расхода и реального ожидания; возврат в норму предлагается
только тогда, когда оптимизатор действительно потерял тег; при пуске и останове
система не советует возвращать уставку, а говорит, что режим ведёт технолог.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from nefte.agents.outlook import (
    LAB_CADENCE_H,
    NEXT_LAB_SIGMA,
    TAU_H,
    TRANSITION_C,
    already_moving,
    cycles,
    next_lab,
    out_of_band,
    sentence,
    tonnes_at_risk,
)
from nefte.agents.schemas import QualityAssessment
from tests.test_agents import make_state

STEPS = {"temperature_c": 2.0, "pressure_mpa": 0.05, "flow_rel": 0.03}
BOUNDS = {"T5": (365.0, 375.0), "T11": (360.0, 370.0),
          "F26": (200.0, 300.0), "P13": (3.5, 4.2)}


def _quality(mean: float = 9.0, sigma: float = 1.2) -> QualityAssessment:
    return QualityAssessment(
        ts=datetime(2026, 4, 20, 12, 0),
        predictions={"product_sulfur_mgkg": mean},
        intervals={"product_sulfur_mgkg": (mean - 1.96 * sigma, mean + 1.96 * sigma)},
        spec_risk={"product_sulfur_mgkg": 0.2}, confidence=0.7)


class _Model:
    """Модель ровно в том объёме, в каком её видит блок «что дальше»."""

    def __init__(self, values: list[float], freq: str = "1h"):
        index = pd.date_range("2026-04-20 00:00", periods=len(values), freq=freq)
        self.feature_matrix = pd.DataFrame({"reg_wabt": values}, index=index)


# ---------------------------------------------------------------- следующий анализ
def test_next_lab_range_is_the_card_interval_widened_by_the_measured_factor():
    q = _quality(mean=9.0, sigma=1.2)
    block = next_lab(make_state(lims=(6.0, 4.0)), q)
    low, high = block["диапазон, мг/кг"]
    assert (high - low) / 2 == pytest.approx(NEXT_LAB_SIGMA * 1.2, abs=0.05)
    # обещание — не про «сейчас», а про момент анализа: он ещё впереди
    assert block["через, ч"] == pytest.approx(LAB_CADENCE_H - 4.0, abs=0.1)
    assert block["задерживается"] is False


def test_overdue_analysis_is_called_overdue_and_never_negative():
    block = next_lab(make_state(lims=(6.0, 30.0)), _quality())
    assert block["задерживается"] is True
    assert block["через, ч"] == 0.0


def test_no_lab_means_no_promise():
    assert next_lab(make_state(lims=None), _quality()) is None
    # без интервала обещать нечего, даже если анализ есть
    bare = QualityAssessment(ts=datetime(2026, 4, 20, 12, 0),
                             predictions={"product_sulfur_mgkg": 9.0})
    assert next_lab(make_state(), bare) is None


# ---------------------------------------------------------------- тонны под риском
def test_tonnes_are_flow_times_waiting():
    assert tonnes_at_risk(10.0, 250.0) == pytest.approx(2500.0, abs=10)
    assert tonnes_at_risk(None, 250.0) is None
    assert tonnes_at_risk(10.0, 0.0) is None


# ---------------------------------------------------------------- режим уже едет
def test_already_moving_sees_a_whole_step_within_the_response_time():
    # история должна быть длиннее окна отклика, иначе сравнивать не с чем
    model = _Model([360.0] * 8 + [363.0])       # +3 °C в конце ряда
    moving = already_moving(model, "2026-04-20 08:00", step_c=2.0, tau_h=TAU_H)
    assert moving is not None and moving["куда"] == "вверх"
    assert moving["ход, °C"] == pytest.approx(3.0, abs=0.01)
    # тот же ряд, но порог перехода в пять шагов — это уже не переход
    assert already_moving(model, "2026-04-20 08:00", step_c=TRANSITION_C) is None


def test_steady_regime_is_not_reported_as_moving():
    model = _Model([360.0, 360.2, 360.1, 360.3, 360.2, 360.1, 360.0, 360.2])
    assert already_moving(model, "2026-04-20 07:00", step_c=2.0) is None


def test_no_matrix_means_silence_not_a_crash():
    assert already_moving(None, "2026-04-20 04:00", step_c=2.0) is None
    assert already_moving(object(), "2026-04-20 04:00", step_c=2.0) is None


# ---------------------------------------------------------------- возврат в норму
def test_small_excursion_is_not_worth_a_line():
    """Выход меньше шага цикла оптимизатор отыгрывает сам — говорить не о чем."""
    state = make_state()
    state.telemetry_ht["T5"] = 375.5          # на 0.5 °C выше при шаге 2 °C
    assert out_of_band(state, BOUNDS, STEPS) is None


def test_excursion_beyond_a_step_names_the_tag_and_the_way_back():
    state = make_state()
    state.telemetry_ht["T5"] = 379.0          # на 4 °C выше верхней границы
    note = out_of_band(state, BOUNDS, STEPS)
    assert note is not None
    row = note["теги"][0]
    assert row["тег"] == "T5" and row["выход"] == pytest.approx(4.0, abs=0.01)
    assert row["циклов до возврата"] == 2     # по 2 °C за цикл
    assert "2 цикла" in note["строка"] and "°C" in note["строка"]
    # граница у нас своя, и карточка обязана это говорить
    assert "ДОПУЩЕНИЕ" in note["строка"]


def test_during_a_transition_the_system_does_not_advise_a_return():
    """Пуск и останов — работа технолога, а не повод возвращать уставку."""
    state = make_state()
    state.telemetry_ht["T5"] = 300.0
    transition = {"ход, °C": -103.0, "окно, ч": TAU_H, "куда": "вниз"}
    note = out_of_band(state, BOUNDS, STEPS, transition)
    assert note is not None and note["переход"] is transition
    assert "рекомендаций по" in note["строка"] and "технолог" in note["строка"]
    assert "циклов" not in note["строка"]


def test_cycles_are_written_in_russian():
    assert cycles(1) == "1 цикл"
    assert cycles(3) == "3 цикла"
    assert cycles(5) == "5 циклов"
    assert cycles(11) == "11 циклов"
    assert cycles(21) == "21 цикл"


# ---------------------------------------------------------------- строка целиком
def test_sentence_mentions_every_part_it_was_given():
    block = {
        "следующий анализ": {"через, ч": 6.0, "ожидается": "2026-04-20 18:00",
                             "задерживается": False, "диапазон, мг/кг": [7.4, 11.2],
                             "доля попаданий на истории": 0.8},
        "тонн под риском": 1500.0,
        "режим уже едет": {"ход, °C": 2.5, "окно, ч": TAU_H, "куда": "вверх"},
    }
    text = sentence(block)
    assert "6 ч" in text and "7.4–11.2" in text
    assert "1500" in text
    assert "ВНИМАНИЕ" in text and "2.5" in text
    assert f"{TAU_H:g}" in text            # когда проверять эффект


# ---------------------------------------------------------------- старение катализатора
AGING = {"°C": 5.5, "месяцев": 6.5, "скорость, °C/мес": 0.848}


def test_temperature_above_band_within_catalyst_aging_is_not_sent_back():
    """Диапазон взят с молодого катализатора: «вернуть вниз» подняло бы серу.

    На тесте все выходы T5 за верх диапазона больше чем на 2 °C (6.2 % моментов)
    объясняются дезактивацией 0.85 °C/мес — со сдвинутой границей их 0.0 %.
    """
    state = make_state()
    state.telemetry_ht["T5"] = 379.0          # на 4 °C выше, дрейф старения 5.5 °C
    note = out_of_band(state, BOUNDS, STEPS, aging=AGING)
    assert note["теги"][0]["старение катализатора"] is True
    assert "НЕ нужно" in note["строка"]
    assert "Возврат —" not in note["строка"]


def test_aging_does_not_excuse_a_bigger_excess_or_a_drop_below():
    state = make_state()
    state.telemetry_ht["T5"] = 385.0          # на 10 °C выше: больше дрейфа
    note = out_of_band(state, BOUNDS, STEPS, aging=AGING)
    assert note["теги"][0]["старение катализатора"] is False
    assert "Возврат —" in note["строка"]

    state.telemetry_ht["T5"] = 360.0          # НИЖЕ диапазона: старение тут ни при чём
    note = out_of_band(state, BOUNDS, STEPS, aging=AGING)
    assert note["теги"][0]["старение катализатора"] is False


CHANGES = ["2024-04-17 15:00", "2026-04-23 11:20"]      # журнал замен из конфига


def test_aging_drift_follows_catalyst_age_not_the_calendar():
    """Катализатор меняли посреди теста: после замены дрейф обнуляется.

    На тесте T5 выше верха диапазона больше чем на 2 °C в 11.1 % моментов ДО замены
    и в 0.0 % после — свежему катализатору нужна меньшая температура.
    """
    from nefte.agents.outlook import aging_drift

    early = aging_drift("2025-09-30", "2025-06-30", 0.848, CHANGES)
    late = aging_drift("2026-04-01", "2025-06-30", 0.848, CHANGES)
    assert 2.0 < early["°C"] < late["°C"]
    # до замены старение идёт по календарю: 9 месяцев × 0.848
    assert late["°C"] == pytest.approx(0.848 * 9.0, rel=0.03)
    # после замены катализатор МОЛОЖЕ того, с которого снят диапазон
    assert aging_drift("2026-06-01", "2025-06-30", 0.848, CHANGES) is None


def test_aging_drift_needs_both_rate_and_catalyst_log():
    from nefte.agents.outlook import aging_drift

    assert aging_drift("2026-01-01", "2025-06-30", None, CHANGES) is None
    assert aging_drift("2026-01-01", "2025-06-30", 0.848, []) is None
    assert aging_drift("2025-01-01", "2025-06-30", 0.848, CHANGES) is None


def test_aging_drift_is_capped_by_the_cycle_span_and_only_goes_up():
    """Просьбы участника 2: сдвиг не больше размаха цикла и только вверх.

    Устаревшая опора не должна уводить границу без предела, а сужать диапазон на
    свежем катализаторе значило бы ввести непроверенное ограничение.
    """
    from nefte.agents.outlook import aging_drift

    free = aging_drift("2026-04-01", "2025-06-30", 0.848, CHANGES)
    capped = aging_drift("2026-04-01", "2025-06-30", 0.848, CHANGES, cap_c=3.0)
    assert free["°C"] > 3.0
    assert capped["°C"] == pytest.approx(3.0)
    # катализатор моложе опорного — дрейфа нет вовсе, а не отрицательный
    assert aging_drift("2026-06-01", "2025-06-30", 0.848, CHANGES, cap_c=25.0) is None
