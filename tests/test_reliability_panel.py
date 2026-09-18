"""Блок надёжности для дашборда: содержание, а не вёрстка.

Данных не требует — отчёт катализатора подставлен выдержкой с настоящими числами.
"""
from __future__ import annotations

import copy
from datetime import datetime

import pytest

from nefte.agents.schemas import ReliabilityAssessment
from nefte.reliability_panel import catalyst_section, reliability_panel, severity_section

REPORT = {
    "срез_данных": "2026-08-07",
    "остановы": [
        {"начало": "2024-03-16", "конец": "2024-04-17", "смена катализатора": True},
        {"начало": "2026-04-15", "конец": "2026-04-23", "смена катализатора": True},
        {"начало": "2026-06-20", "конец": "2026-06-30", "смена катализатора": False},
    ],
    "циклы": [
        {"цикл": 0, "наблюдается с начала": False, "завершён": True, "длительность, сут": 438.0},
        {"цикл": 1, "наблюдается с начала": True, "завершён": True, "длительность, сут": 727.0},
        {"цикл": 2, "наблюдается с начала": True, "завершён": False, "длительность, сут": 108.0},
    ],
    "скорость_дезактивации": {"°C/мес": 0.848, "95% ДИ": [0.709, 0.983]},
    "уровень_вывода_°C": 379.7,
    "текущий_цикл": {"пуск": "2026-04-23", "наработка, сут": 108.0, "NWABT сейчас, °C": 363.2,
                     "запас до уровня вывода, °C": 16.6, "скорость в этом цикле, °C/мес": 4.15},
    "остаточный_ресурс_мес": {
        "по средней скорости": 19.6, "по средней скорости, ДИ": [16.9, 23.4],
        "по аналогии": [{"цикл": 0, "оставалось, мес": 14.4, "нижняя граница": True},
                        {"цикл": 1, "оставалось, мес": 13.0, "нижняя граница": False}],
        "по текущей скорости": 4.0,
        "по отставанию от эталона": {"эталон прожил ещё, мес": 20.3, "отставание, °C": 2.7,
                                     "отставание, мес наработки": 3.2, "остаток, мес": 17.2},
    },
    "контрольная_точка": {"сутки": 150, "уровень эталона, °C": 359.9, "достигнута": False,
                          "отставание": [{"сутки": 100.0, "отставание, °C": 2.82,
                                          "значимо": True}]},
    "бэктест_метода": [{"сутки": 108, "ошибка (а), %": 11.0}, {"сутки": 300, "ошибка (а), %": 74.0},
                       {"сутки": 500, "ошибка (а), %": 104.0}],
}
FACTORS = {"wabt": 0.8, "dp_r202": 0.5, "anomaly": 0.2, "ramp": 0.1, "catalyst": 0.3,
           "furnace": 0.6}


def _assessment(factors: dict, risk_class: str = "medium") -> ReliabilityAssessment:
    return ReliabilityAssessment(ts=datetime(2026, 7, 1), severity_index=0.475,
                                 risk_class=risk_class, factors=factors,
                                 constraints={"T5": (361.0, 365.0)})


def test_contributions_add_up_to_the_index_and_are_ranked():
    section = severity_section(_assessment(FACTORS), dp="level")
    contributions = [row["вклад"] for row in section["факторы"]]
    assert sum(contributions) == pytest.approx(0.475, abs=0.003)
    assert contributions == sorted(contributions, reverse=True)
    assert section["главный фактор"] == "WABT реакторного блока"
    assert "1 °C" in section["что значит класс"]


def test_missing_factor_is_named_and_weights_renormalize():
    factors = {k: v for k, v in FACTORS.items() if k != "anomaly"}
    section = severity_section(_assessment(factors), dp="level")
    assert section["нет данных"] == ["нетипичность режима"]
    assert sum(row["вклад"] for row in section["факторы"]) == pytest.approx(
        (0.475 - 0.15 * 0.2) / 0.85, abs=0.003)


def test_catalyst_label_follows_how_the_agent_measures_wear():
    age = severity_section(_assessment(FACTORS), basis="age")
    activity = severity_section(_assessment(FACTORS), basis="activity")
    label = {row["фактор"]: row["что это"] for row in age["факторы"]}
    assert "возраст" in label["catalyst"]
    label = {row["фактор"]: row["что это"] for row in activity["факторы"]}
    assert "приведённой к нагрузке" in label["catalyst"]


def test_first_half_of_cycle_shows_the_linear_estimate_and_the_range():
    section = catalyst_section(REPORT, "2026-08-07 12:00")
    life = section["ресурс"]
    assert section["сутки цикла"] == 106 and section["оценка на сутки"] == 108
    assert life["половина цикла"] == "первая"
    assert life["диапазон, мес"] == [13.0, 19.6]
    assert life["вероятнее, мес"] == 17.2
    assert any(m["способ"].startswith("(а)") for m in life["способы"])
    assert not section["по данным после момента"]
    # оговорка о второй половине есть всегда, с измеренной ошибкой
    assert any("364-х" in note and "+104 % на 500-х" in note for note in section["оговорки"])


def test_second_half_of_cycle_hides_the_linear_estimate():
    late = copy.deepcopy(REPORT)
    late["текущий_цикл"]["наработка, сут"] = 400.0
    life = catalyst_section(late, "2026-08-07")["ресурс"]
    assert life["половина цикла"] == "вторая"
    assert not any(m["способ"].startswith("(а)") for m in life["способы"])
    assert life["диапазон, мес"] == [13.0, 17.2]
    assert life["вероятнее, мес"] is None
    assert life["ошибка линейной оценки на эталоне"] == {"сутки": 300, "ошибка, %": 74.0}


def test_no_life_estimate_for_a_past_cycle():
    section = catalyst_section(REPORT, "2025-06-01")
    assert section["пуск цикла"] == "2024-04-17"
    assert section["сутки цикла"] == 410
    assert section["ресурс"] is None


def test_before_first_catalyst_change_the_cycle_day_is_unknown():
    section = catalyst_section(REPORT, "2023-06-01")
    assert section["сутки цикла"] is None and section["ресурс"] is None


def test_repair_without_change_does_not_restart_the_cycle():
    section = catalyst_section(REPORT, "2026-07-15")
    assert section["пуск цикла"] == "2026-04-23"
    assert section["по данным после момента"]


def test_checkpoint_criterion_is_taken_from_the_reference_cycle():
    point = catalyst_section(REPORT, "2026-08-07")["контрольная точка"]
    assert point["дата"] == "2026-09-20"
    assert point["пересмотреть вниз, если NWABT выше, °C"] == pytest.approx(364.9)
    assert point["или скорость на сутках 60…N выше, °C/мес"] == 1.0
    assert point["отставание от эталона"]["значимо"]


def test_panel_without_catalyst_report_still_renders_severity():
    panel = reliability_panel(_assessment(FACTORS), None)
    assert panel["катализатор"]["доступно"] is False
    assert panel["тяжесть"]["класс"] == "medium"


def test_disabled_pressure_drop_is_named_as_disabled_not_missing():
    """Выключенный перепад — не «нет данных»: его не считают намеренно."""
    factors = {k: v for k, v in FACTORS.items() if k != "dp_r202"}
    section = severity_section(_assessment(factors), dp="off")
    assert "перепад давления на Р-202" not in section["нет данных"]
    assert [x["фактор"] for x in section["выключены"]] == ["перепад давления на Р-202"]
    assert sum(row["вклад"] for row in section["факторы"]) == pytest.approx(
        (0.475 - 0.20 * 0.5) / 0.80, abs=0.003)


def test_pressure_drop_label_follows_the_mode():
    level = severity_section(_assessment(FACTORS), dp="level")
    growth = severity_section(_assessment(FACTORS), dp="growth")
    meaning = {m: {r["фактор"]: r["что это"] for r in s["факторы"]}["dp_r202"]
               for m, s in (("level", level), ("growth", growth))}
    assert "гидравлику" in meaning["level"]
    assert "от начала цикла" in meaning["growth"]
