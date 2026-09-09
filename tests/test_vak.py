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


def test_broken_formula_is_skipped_not_repaired():
    """AVT6:240-350:CFPP записана с непарной скобкой — домысливать её нельзя."""
    usable, skipped = compile_formulas()
    names = {i["target"] for i in skipped}
    assert "AVT6:240-350:CFPP" in names
    assert all("CFPP" not in i["target"] or "350:" not in i["target"]
               for i in usable if i["unit"] == "avt" and "240-350" in i["target"])


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
    # формула T50 = 44.625 + 10.0224*P13 + 0.06981*F9 + 0.8052*T6
    expected = 44.625 + 10.0224 * 1.0 + 0.06981 * 1.0 + 0.8052 * 1.0
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
    """16 формул из 17: одна битая в исходнике."""
    usable, skipped = compile_formulas()
    assert len(usable) == 16
    assert len(skipped) == 1
