"""Тесты разбора формул блока АВТ (участник 2).

Находки такого рода живут ровно до первой правки конфига, если их не закрепить.
Здесь закрепляются три вещи:

* прочтения, выведенные нами из данных, лежат ОТДЕЛЬНО от поправок организаторов
  и не попадают в рабочий расчёт молча;
* каждое такое прочтение действительно воспроизводит лабораторию — с числом, а
  не «мы посмотрели, стало лучше»;
* диагноз по ``AVT6:350:T50`` (в ячейке дубль формулы плотности) проверяем прямо,
  а не пересказом.

Тесты считают по выданным данным: без них они не имеют смысла, потому что
проверяют утверждение о данных, а не о коде.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from nefte.config import load_config
from nefte.data.cleaning import clean_telemetry
from nefte.data.loaders import (
    load_lims,
    load_telemetry,
    load_vak_formulas,
    parse_vak_formula,
)
from nefte.models.vak import compile_formulas

TAG_RE = re.compile(r"\b([A-Z]{1,4}[0-9]{1,3})\b")


@pytest.fixture(scope="module")
def avt() -> pd.DataFrame:
    frame, _ = clean_telemetry(load_telemetry("avt"), unit="avt")
    return frame


@pytest.fixture(scope="module")
def formulas() -> dict[str, str]:
    table = load_vak_formulas()
    return dict(zip(table["target"], table["formula"]))


def _evaluate(expr: str, avt: pd.DataFrame) -> pd.Series:
    tags = sorted(set(TAG_RE.findall(expr)))
    namespace = {tag: avt[tag] for tag in tags}
    value = eval(compile(expr, "<test>", "eval"),                  # noqa: S307
                 {"__builtins__": {}}, namespace)
    return pd.Series(value, index=avt.index).replace([np.inf, -np.inf], np.nan)


def _against_lab(series: pd.Series, point: str, param: str,
                 positive_only: bool = False) -> dict:
    lims = load_lims()
    sub = lims[(lims["unit"] == "АВТ") & (lims["point"] == point)
               & (lims["param"] == param)]
    lab = sub.set_index("ts")["value"].sort_index()
    # ноль в отгоне и в плотности — это пропуск, а не измерение
    if positive_only:
        lab = lab[lab > 0]
    position = series.index.searchsorted(lab.index, side="right") - 1
    known = position >= 0
    model, truth = series.to_numpy()[position[known]], lab.to_numpy()[known]
    good = ~np.isnan(model) & ~np.isnan(truth)
    model, truth = model[good], truth[good]
    return {"n": len(truth), "bias": float(np.mean(model - truth)),
            "mae": float(np.mean(np.abs(model - truth)))}


def test_data_derived_readings_never_pass_as_organizer_corrections():
    """Прочтение, выведенное нами, не должно выдаваться за присланное заказчиком.

    Сначала этот тест держал другое: что наши прочтения в рабочий расчёт вообще
    не попадают. Решение изменилось — участник 1 дал им приоритет над поправками
    организаторов, и правильно: у нашего прочтения есть измеримое свидетельство
    (MAE 2.3 °C против 62), а у присланного нет.

    Но тогда инвариант становится строже, а не слабее. Раз формула применяется,
    единственное, что отделяет «прислано заказчиком» от «вывели сами», — поле
    происхождения. На защите это разные вещи, и подменять одно другим нельзя.
    """
    vak = load_config()["vak"]
    proposed = vak.get("proposed_corrections") or {}
    assert proposed, "разобранные прочтения должны быть записаны в конфиге"

    usable, _ = compile_formulas()
    by_target = {item["target"]: item for item in usable}
    for target, item in proposed.items():
        assert item.get("assumption") is True
        assert "НЕ подтверждено" in item["source"], \
            f"{target}: у прочтения должен быть виден источник"

        compiled = by_target[target]
        assert compiled["expr"] == parse_vak_formula(item["formula"]), \
            f"{target}: применяется не наше прочтение"
        assert compiled["correction_source"] == "наше прочтение"
        assert compiled["correction_confirmed"] is False, \
            f"{target}: помечено как подтверждённое организаторами"


def test_organizer_corrections_are_still_marked_as_theirs():
    """Обратная сторона: то, что прислали, обязано остаться помеченным как их."""
    usable, _ = compile_formulas()
    proposed = set((load_config()["vak"].get("proposed_corrections") or {}))
    confirmed = [i for i in usable if i["correction_confirmed"]]
    assert confirmed, "поправки организаторов должны быть видны как подтверждённые"
    for item in confirmed:
        assert item["correction_source"] == "организаторы"
        assert item["target"] not in proposed, \
            f"{item['target']}: перекрытая поправка числится подтверждённой"


def test_proposed_cfpp_reading_reproduces_the_laboratory(avt, formulas):
    """Скобка не там: проверяем оба прочтения по лаборатории точки АВТ|3.

    Формула обещает предельную температуру фильтруемости фракции 240-350.
    В лаборатории это ряд ``FilterabilityLimit.T`` точки АВТ|3 — привязку блока
    к точке даёт ``scripts/check_avt_formulas.py``.
    """
    config = load_config()["vak"]
    sent = _evaluate(parse_vak_formula(
        config["corrections"]["AVT6:240-350:CFPP"]["formula"]), avt)
    ours = _evaluate(parse_vak_formula(
        config["proposed_corrections"]["AVT6:240-350:CFPP"]["formula"]), avt)

    sent_fit = _against_lab(sent, "3", "FilterabilityLimit.T")
    our_fit = _against_lab(ours, "3", "FilterabilityLimit.T")
    assert our_fit["n"] >= 50, "анализов слишком мало, вывод не на чем строить"

    # присланное прочтение промахивается на десятки градусов ПТФ
    assert sent_fit["mae"] > 40.0
    # наше — в пределах точности самой лаборатории по этому показателю
    assert our_fit["mae"] < 5.0
    assert abs(our_fit["bias"]) < 3.0


def test_proposed_i350_constant_restores_a_physical_level(avt):
    """Потерянная первая цифра константы: −265 % отгона против 93 % в лаборатории."""
    config = load_config()["vak"]
    fixed = _evaluate(parse_vak_formula(
        config["proposed_corrections"]["AVT6:350:I350"]["formula"]), avt)
    lab = _against_lab(fixed, "1", "I350", positive_only=True)
    assert lab["n"] >= 100
    assert abs(lab["bias"]) < 10.0
    assert 0.0 <= float(fixed.median()) <= 100.0

    # выданная константа даёт величину, которой отгон быть не может
    original = _evaluate(parse_vak_formula(
        load_vak_formulas().set_index("target").loc["AVT6:350:I350", "formula"]), avt)
    assert float(original.median()) < -100.0


def test_t50_cell_of_the_350_block_duplicates_the_density_formula(avt, formulas):
    """Диагноз проверяем счётом: две формулы совпадают до константы."""
    t50 = _evaluate(parse_vak_formula(formulas["AVT6:350:T50"]), avt)
    d15 = _evaluate(parse_vak_formula(formulas["AVT6:350:D15"]), avt)
    difference = (d15 - t50).dropna()
    assert difference.std() < 1e-3, "формулы отличаются не только свободным членом"
    assert float(difference.median()) == pytest.approx(2.027, abs=0.01)

    # и это плотность, а не температура 50 % отгона
    assert _against_lab(d15, "1", "D15", positive_only=True)["mae"] < 5.0


def test_avt_tags_behave_as_the_instrument_dictionary_says(avt):
    """Опровержение прошлого вывода: теги АВТ означают то, что написано.

    Берём теги из формул, которые лабораторию воспроизводят, и проверяем, что
    они лежат в диапазонах своих величин. Если бы короткие имена означали не те
    величины, рабочие формулы не сходились бы с лабораторией — а они сходятся.
    """
    ranges = {"T33": (300.0, 380.0), "T42": (250.0, 320.0), "T48": (330.0, 380.0),
              "T40": (140.0, 200.0), "F30": (80.0, 180.0), "F31": (350.0, 700.0),
              "F57": (5.0, 90.0), "L43": (40.0, 80.0)}
    for tag, (low, high) in ranges.items():
        median = float(avt[tag].median())
        assert low <= median <= high, f"{tag}: медиана {median} вне {low}…{high}"
