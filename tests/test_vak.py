"""Тесты вычисления формул виртуальных анализаторов из справочника.

Главное, что проверяем: формулы считаются на правильной установке, битые не
«чинятся» молча, а формулы на ЛИМС не тянут значение текущего анализа.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.data.loaders import parse_vak_formula
from nefte.models.vak import LIMS_TOKENS, compile_formulas, evaluate


def test_parse_handles_both_notations():
    """В справочнике смешаны '0,52755xT66' и '0.27467*T42'."""
    assert parse_vak_formula("791,22872 - 5,30294x(F30/(F32+ F30))") == \
        "791.22872-5.30294*(F30/(F32+F30))"
    assert parse_vak_formula("0.27467*T42 - 0.32983x(F31/F57)") == \
        "0.27467*T42-0.32983*(F31/F57)"


def test_formula_is_repaired_only_by_a_declared_correction():
    """AVT6:240-350:CFPP была записана с непарной скобкой.

    Домысливать чужую формулу нельзя — и мы не домысливали, а получили поправку
    от организаторов. Теперь формула считается, но только потому, что поправка
    объявлена в конфиге вместе с причиной. Проверяем именно это: исправленная
    формула обязана иметь источник, иначе это самодеятельность.
    """
    usable, _ = compile_formulas()
    by_target = {i["target"]: i for i in usable}
    fixed = by_target["AVT6:240-350:CFPP"]
    assert fixed["corrected"] is True
    assert "организатор" in fixed["correction_reason"].lower()


def test_uncorrected_formulas_stay_untouched():
    """Формула без объявленной поправки берётся из выданного файла как есть."""
    usable, _ = compile_formulas()
    untouched = [i for i in usable if not i["corrected"]]
    assert untouched, "не все формулы поправлены — проверять есть что"
    assert all(i["correction_reason"] == "" for i in untouched)


def test_formulas_are_bound_to_their_own_unit():
    """T6 есть и на АВТ, и на 24-2000: формула обязана брать теги своей установки."""
    usable, _ = compile_formulas()
    by_target = {i["target"]: i for i in usable}
    assert by_target["24-2000:GODT:T50"]["unit"] == "ht"
    assert by_target["AVT6:350:I350"]["unit"] == "avt"


def _telemetry(n: int = 20):
    idx = pd.date_range("2024-01-01", periods=n, freq="10min", name="date")
    avt = pd.DataFrame({tag: np.linspace(1.0, 2.0, n) for tag in
                        ("F30", "F32", "T66", "T33", "F7", "F34", "F45", "F59", "F63",
                         "F36", "T37", "T40", "T58", "T42", "T48", "F31", "F57", "L43",
                         "T6", "T18", "T15", "T11", "F64", "T13", "T20", "P50", "P51",
                         "F53", "P67", "P4", "F65")}, index=idx)
    ht = pd.DataFrame({tag: np.linspace(1.0, 2.0, n) for tag in
                       ("T12", "F15", "W7", "T23", "F1", "F26", "P13", "F9", "T6", "T5",
                        "T11", "F25", "F14", "T16", "F22", "F2", "P8", "P24", "W4")},
                      index=idx)
    return avt, ht


def test_evaluate_computes_formula_arithmetic():
    """Проверка счёта как такового — с выключённой отбраковкой по диапазону."""
    avt, ht = _telemetry()
    out, _ = evaluate(avt, ht, check_plausibility=False)
    assert "vak_24_2000_GODT_T50" in out.columns
    # исправленная организаторами формула: 44.625 + 10.0224*P13 + 0.06981*F9 + 0.471*T6
    expected = 44.625 + 10.0224 * 1.0 + 0.06981 * 1.0 + 0.471 * 1.0
    assert out["vak_24_2000_GODT_T50"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_implausible_output_is_rejected():
    """T50 в 55 °C — не температура выкипания дизтоплива, признак не берём."""
    avt, ht = _telemetry()
    out, skipped = evaluate(avt, ht)
    reasons = {i["target"]: i["reason"] for i in skipped}
    assert "24-2000:GODT:T50" in reasons
    assert "вне физического диапазона" in reasons["24-2000:GODT:T50"]
    assert "vak_24_2000_GODT_T50" not in out.columns


def test_division_by_zero_becomes_missing_not_infinity():
    avt, ht = _telemetry()
    avt.loc[avt.index[0], "F57"] = 0.0          # формулы блока 350 делят на F57
    out, _ = evaluate(avt, ht, check_plausibility=False)
    assert not np.isinf(out["vak_AVT6_350_D15"]).any()
    assert np.isnan(out["vak_AVT6_350_D15"].iloc[0])


def test_lims_based_formulas_need_context():
    """Без ряда ЛИМС формула не считается — молча подставлять ноль нельзя."""
    avt, ht = _telemetry()
    out, skipped = evaluate(avt, ht)
    reasons = {i["target"]: i["reason"] for i in skipped}
    assert "24-2000:GODT:D15" in reasons and "нет входов" in reasons["24-2000:GODT:D15"]
    assert "vak_24_2000_GODT_D15" not in out.columns


def test_lims_context_enables_those_formulas():
    avt, ht = _telemetry()
    ctx = {name: pd.Series(830.0, index=ht.index) for name, _ in LIMS_TOKENS.values()}
    out, _ = evaluate(avt, ht, ctx)
    assert "vak_24_2000_GODT_D15" in out.columns
    assert np.isfinite(out["vak_24_2000_GODT_D15"]).all()


def test_all_expected_targets_are_covered():
    """Все 17 формул считаются: единственная битая исправлена поправкой."""
    usable, skipped = compile_formulas()
    assert len(usable) == 17
    assert skipped == []
    # шесть от организаторов плюс две наши; AVT6:240-350:CFPP есть в обоих
    # списках, и наше прочтение перекрывает присланное
    assert sum(1 for i in usable if i["corrected"]) == 7
    by_source = {i["target"]: i["correction_source"] for i in usable if i["corrected"]}
    assert by_source["AVT6:240-350:CFPP"] == "наше прочтение"
    assert by_source["24-2000:GODT:T95"] == "организаторы"


def test_our_own_reading_is_never_passed_off_as_confirmed():
    """Источник поправки обязан быть виден.

    «Прислано заказчиком» и «вывели сами» — разные вещи, и на защите они должны
    звучать по-разному. Поле correction_confirmed отделяет одно от другого; без
    него наше прочтение формулы неотличимо от ответа организаторов.
    """
    usable, _ = compile_formulas()
    for item in usable:
        if item["correction_source"] == "наше прочтение":
            assert item["correction_confirmed"] is False
            assert item["correction_reason"]
        elif item["correction_source"] == "организаторы":
            assert item["correction_confirmed"] is True
