"""Рабочие формулы и смысл тегов сверены с материалами организаторов от 15.09.

Две таблицы лежат в репозитории, потому что без них ошибки пакета не видны:

* ``configs/vak_official_2026-09-15.csv`` — все 17 формул ВАК с примерами расчёта.
  Рабочая формула каждого анализатора обязана совпадать с официальной численно:
  пакет, первые ответы организаторов и наши прочтения уже расходились с ней в
  четырёх местах, и молча такое не должно повториться;
* ``configs/tags_2026-09-15.csv`` — описания тегов. В листе «КИП» пакета описания
  24-2000 перемешаны со строками, и по ним в коде стояли три ошибки: перепад Р-202
  мерился расходом бензина, подпитка водородом — давлением К-201, квенч — тегом,
  не связанным с нагрузкой. Здесь закреплено, что теги ролей описаны именно так.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from nefte.agents.reliability import ReliabilityAgent
from nefte.config import ROOT
from nefte.data.loaders import load_tag_dictionary
from nefte.models import regime

OFFICIAL = ROOT / "configs" / "vak_official_2026-09-15.csv"
NAME = re.compile(r"\b([A-Z]{1,4}[0-9]{1,3}|LIMS_[A-Z0-9]+)\b")


def _official() -> dict[str, str]:
    table = pd.read_csv(OFFICIAL, encoding="utf-8")
    return {row.target: (row.formula.replace("LIMS.95%.T", "LIMS_T95")
                         .replace("LIMS.D15", "LIMS_D15"))
            for row in table.itertuples()}


def _value(expr: str, env: dict[str, float]) -> float:
    return float(eval(compile(expr, "<vak>", "eval"), {"__builtins__": {}}, dict(env)))  # noqa: S307


def test_official_table_has_all_seventeen_formulas():
    formulas = _official()
    assert len(formulas) == 17
    assert sum(t.startswith("AVT6:") for t in formulas) == 9


def test_every_working_formula_equals_the_official_one():
    from nefte.models.vak import compile_formulas

    usable, skipped = compile_formulas()
    assert skipped == []
    working = {item["target"]: item["expr"] for item in usable}
    official = _official()
    assert set(working) == set(official)

    rng = np.random.default_rng(0)
    for target, expr in official.items():
        names = set(NAME.findall(expr)) | set(NAME.findall(working[target]))
        # несколько случайных точек: совпадение в одной могло бы быть случайным
        for _ in range(3):
            env = {name: float(rng.uniform(50.0, 300.0)) for name in names}
            assert _value(working[target], env) == pytest.approx(_value(expr, env), rel=1e-9), (
                f"{target}: рабочая формула расходится с официальной таблицей 15.09")


@pytest.mark.parametrize("tag,words", [
    (ReliabilityAgent.DP_TAG, "перепад давления"),
    (regime.MAKEUP_H2, "свежего всг"),
    (regime.QUENCH, "квенча"),
    (regime.RECYCLE_GAS, "цк-201"),
    (regime.PRESSURE, "давление"),
])
def test_role_tags_are_described_as_their_role(tag: str, words: str):
    tags = load_tag_dictionary()
    row = tags[(tags["unit"] == "ht") & (tags["code"] == tag)]
    assert len(row) == 1, f"{tag} нет в таблице тегов 24-2000"
    assert words in str(row["description"].iloc[0]).lower(), (
        f"{tag}: в таблице организаторов «{row['description'].iloc[0]}», "
        f"а код использует его как «{words}»")


@pytest.mark.parametrize("tag", regime.REACTOR_TEMPS)
def test_reactor_temperatures_are_temperatures(tag: str):
    tags = load_tag_dictionary()
    row = tags[(tags["unit"] == "ht") & (tags["code"] == tag)]
    assert "температура" in str(row["description"].iloc[0]).lower()
    assert "р-20" in str(row["description"].iloc[0]).lower()


def test_package_sheet_is_not_the_default_dictionary():
    """Лист пакета остаётся для воспроизведения, но по умолчанию не читается."""
    tags = load_tag_dictionary()
    assert "quantity" in tags.columns
    assert len(tags[tags["unit"] == "ht"]) == 26
    assert len(tags[tags["unit"] == "avt"]) == 71
