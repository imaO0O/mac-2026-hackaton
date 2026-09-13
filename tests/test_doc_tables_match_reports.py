# -*- coding: utf-8 -*-
"""Таблицы документации, переписанные из отчётов, обязаны сходиться с отчётами.

Сверялись две таблицы из `GPU_MODELS.md`, остальные числа переносились руками. За
один пересчёт на матрице версии 7 руками же пришлось чинить устаревшие Brier в
разборе калибровки, всю таблицу цены порога, разброс весов свёртки, таблицу
дрейфа по кварталам — и под устаревшей таблицей дрейфа нашлась ошибка знака,
из которой следовал успокаивающий вывод «модель завышает, это безопасно».

Переписывание руками надёжно ровно один раз — в момент переписывания. Этот тест
перечисляет, из какого поля какого отчёта взято каждое число таблицы, и падает,
если документация разошлась с отчётом.

Сверяются только строки, описывающие ТЕКУЩУЮ сборку. Исторические строки («было
при сроке годности шесть месяцев», «одна проба на сид») нарочно не входят: их
отчётов больше нет, и это не дефект, а история.

Точность сравнения — та, с которой число напечатано: «0.82» сверяется до 0.005,
«0.819» до 0.0005. Устаревший отчёт (другая версия матрицы) пропускается: об этом
уже говорит `test_report_freshness`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import pytest

from nefte.config import ROOT
from nefte.models.dataset import FEATURE_VERSION

NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


@lru_cache(maxsize=None)
def report(name: str) -> dict:
    path = ROOT / "reports" / name
    if not path.exists():
        pytest.skip(f"нет reports/{name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "feature_version" in data and data["feature_version"] != FEATURE_VERSION:
        pytest.skip(f"reports/{name} снят на другой матрице")
    return data


def numbers(cell: str) -> list[tuple[float, int]]:
    """Числа ячейки и число знаков после точки у каждого."""
    text = cell.replace("*", "").replace("−", "-").replace(" ", " ")
    out = []
    for token in NUMBER.findall(text):
        token = token.replace(",", ".")
        decimals = len(token.split(".")[1]) if "." in token else 0
        out.append((float(token), decimals))
    return out


def table_rows(doc: str, header: str, label_cells: int = 1) -> dict[str, list[str]]:
    lines = (ROOT / "docs" / doc).read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if start is None:
        pytest.fail(f"в docs/{doc} нет таблицы с заголовком «{header}»")
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        key = " | ".join(c.replace("*", "").strip() for c in cells[:label_cells])
        rows[key] = cells[label_cells:]
    return rows


@dataclass(frozen=True)
class Row:
    doc: str
    header: str
    label: str
    # по ячейкам: список ожидаемых чисел или None — ячейку не сверять
    expected: Callable[[], list[list[float] | None]]
    label_cells: int = 1

    @property
    def id(self) -> str:
        return f"{self.doc}: {self.header.strip('| ')[:28]} / {self.label}"


def q(h: int) -> dict:
    return report(f"quality_metrics_h{h}.json")["splits"]["test"]


def model_cells(h: int) -> list[list[float]]:
    m = q(h)["model"]
    return [[m["MAE"]], [m["coverage_80"]], [m["spec_precision"]], [m["spec_recall"]],
            [m["roc_auc"]]]


def drift_interval(label: str) -> list[list[float]]:
    row = next(r for r in report("drift_h0.json")["интервалы"] if r["интервал"] == label)
    return [[row["возраст, мес"]], [row["среднее факта"]], [row["MAE"]], [row["смещение"]],
            [row["покрытие 80%"]], [row["выигрыш у персистенции"]]]


def age_level(key: str) -> list[float]:
    cell = report("drift_h0.json")["возраст_или_уровень"]["таблица 2×2"][key]
    return [cell["медиана смещения"], cell["анализов"]]


def slope(split: str) -> list[list[float] | None]:
    s = report("calibration_h0.json")["выборки"][split]["платт"]["наклон_калибровки"]
    return [[s["b"]], [s["b_от"], s["b_до"]], None]


def budget(name: str) -> list[list[float]]:
    v = report(name)["выборки"]
    return [[v["val"]["средний_риск"], v["test"]["средний_риск"]],
            [100 * v["val"]["фиксированный"]["доля_тревог"],
             100 * v["test"]["фиксированный"]["доля_тревог"]]]


def budget_seq_test(kind: str) -> list[list[float]]:
    r = report("alarm_budget_seq_tcn48_pre_h2.json")["выборки"]["test"][kind]
    return [[100 * r["доля_тревог"]], [r["precision"]], [r["recall"]], [r["n"]]]


def stability(h: int) -> list[list[float]]:
    d = report(f"feature_stability_h{h}.json")
    mae = [r["MAE"] for r in d["метрики"]]
    auc = [r["ROC-AUC"] for r in d["метрики"]]
    return [[100 * len(d["ядро"]) / len(d["частота"])], [max(mae) - min(mae)],
            [max(auc) - min(auc)]]


def sweep(threshold: float) -> list[list[float]]:
    rows = report("test_period.json")["summary"]["порог вмешательства"]["перебор"]
    r = next(x for x in rows if abs(x["порог"] - threshold) < 1e-3)
    return [[r["поймано"]], [r["пропущено"], 100 * r["доля пропусков"]],
            [100 * r["доля пропусков с отказами"]],
            [r["ложных тревог"]], [100 * r["доля ложных тревог"]]]


def refusals_now() -> list[list[float]]:
    s = report("test_period.json")["summary"]
    w = next(x for x in s["порог вмешательства"]["перебор"] if x["рабочий"])
    return [[s["исходы"]["отказ"]], [s["причины отказа"]["недостоверные данные"]],
            [w["поймано"]], [w["пропущено"]], [w["ложных тревог"]]]


def weights() -> dict:
    return report("objective_weights.json")


QA, HC, OA = "QUALITY_AGENT.md", "HARD_CHECKS.md", "OPTIMIZER_AGENT.md"
MAIN = "| Горизонт | Выборка | Модель |"
HORIZONS = "| Горизонт | MAE модели |"

ROWS = [
    *[Row(QA, MAIN, f"{h} ч | test | модель", (lambda h=h: model_cells(h)), label_cells=3)
      for h in (0, 2)],
    *[Row(QA, MAIN, f"{h} ч | test | {name}", (lambda h=h, key=key: [[q(h)[key]["MAE"]]]),
          label_cells=3)
      for h in (0, 2) for name, key in (("ПАК", "baseline_pak"),
                                         ("пред. анализ", "baseline_lims_prev"))],
    Row(QA, HORIZONS, "0 ч", lambda: [[q(0)["model"]["MAE"]], [q(0)["baseline_pak"]["MAE"]]]
        + model_cells(0)[1:]),
    *[Row(QA, HORIZONS, f"{h} ч", (lambda h=h: [[q(h)["model"]["MAE"]],
                                                [q(h)["baseline_pak"]["MAE"]]]
                                              + model_cells(h)[1:]))
      for h in (1, 2, 3)],
    Row(QA, "| Тест, горизонт 0 | Brier |", "без поправки", lambda: [
        [report("calibration_h0.json")["выборки"]["test"]["сырой"]["Brier"]],
        [report("calibration_h0.json")["выборки"]["test"]["сырой"]["ECE"]],
        [report("calibration_h0.json")["выборки"]["test"]["сырой"]["средняя_заявленная"]
         - report("calibration_h0.json")["выборки"]["test"]["базовая_частота"]]]),
    Row(QA, "| | наклон | 90 % интервал |", "валидация", lambda: slope("val")),
    Row(QA, "| | наклон | 90 % интервал |", "тест", lambda: slope("test")),
    *[Row(QA, "| Интервал | Возраст, мес |", label, (lambda label=label: drift_interval(label)))
      for label in ("2025-09", "2025-12", "2026-03", "2026-06", "2026-09")],
    Row(QA, "| возраст модели, мес |", "смещение, мг/кг",
        lambda: [[r["смещение"]] for r in report("drift_h0.json")["интервалы"]]),
    Row(QA, "| медиана смещения, мг/кг |", "модель моложе 6.4 мес",
        lambda: [age_level("моложе, низкий уровень"), age_level("моложе, высокий уровень")]),
    Row(QA, "| медиана смещения, мг/кг |", "старше",
        lambda: [age_level("старше, низкий уровень"), age_level("старше, высокий уровень")]),
    Row(QA, "| Модель | средний риск val → test |", "CatBoost, горизонт 0",
        lambda: budget("alarm_budget_boost_h0.json")),
    Row(QA, "| Модель | средний риск val → test |", "TCN 48 + предобучение, горизонт 2",
        lambda: budget("alarm_budget_seq_tcn48_pre_h2.json")),
    Row(QA, "| Горизонт 2, тест, сеть |", "фиксированный порог",
        lambda: budget_seq_test("фиксированный")),
    Row(QA, "| Горизонт 2, тест, сеть |", "он же на общей выборке",
        lambda: budget_seq_test("фиксированный_на_общей")),
    Row(QA, "| Горизонт 2, тест, сеть |", "скользящий порог",
        lambda: budget_seq_test("скользящий")),
    *[Row(QA, "| | ядро признаков | разброс MAE |", f"горизонт {h}, три пробы",
          (lambda h=h: stability(h))) for h in (0, 1, 2)],
    *[Row(HC, "| Порог | Поймано | Пропущено |", label, (lambda t=t: sweep(t)))
      for label, t in (("0.10", 0.10), ("0.15", 0.15), ("0.177 (рабочий)", 0.177),
                       ("0.20", 0.20), ("0.30", 0.30), ("0.40", 0.40), ("0.50", 0.50))],
    Row(HC, "| | отказов всего |", "после", refusals_now),
    Row(OA, "| При разбросе весов ±50 % |", "исход (держим / меняем / отказ)",
        lambda: [[100 * weights()["исход сохраняется"]]]),
    Row(OA, "| При разбросе весов ±50 % |", "направление воздействия по T5",
        lambda: [[100 * weights()["направление сохраняется"]]]),
    Row(OA, "| При разбросе весов ±50 % |", "величина ΔT5",
        lambda: [[weights()["ΔT5 разброс, среднее"]]]),
    Row(OA, "| При разбросе весов ±50 % |", "прогноз серы у выбранного варианта",
        lambda: [[weights()["сера разброс, среднее"]]]),
    Row(OA, "| При разбросе весов ±50 % |", "моментов с действием, на которых это посчитано",
        lambda: [[weights()["моментов с действием"], weights()["stamps"]]]),
]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r.id)
def test_doc_row_matches_report(row: Row):
    cells = table_rows(row.doc, row.header, row.label_cells).get(row.label)
    assert cells is not None, f"в таблице «{row.header}» нет строки «{row.label}»"
    expected = row.expected()
    assert len(cells) >= len(expected), f"в строке меньше ячеек, чем сверяется: {cells}"
    for i, want in enumerate(expected):
        if want is None:
            continue
        got = numbers(cells[i])
        assert len(got) == len(want), (
            f"ячейка «{cells[i]}»: ожидалось {len(want)} числа, найдено {len(got)}")
        for (value, decimals), target in zip(got, want):
            tolerance = 0.5 * 10 ** -decimals + 1e-9
            assert abs(value - target) <= tolerance, (
                f"«{row.label}», ячейка «{cells[i]}»: в документации {value}, "
                f"в отчёте {target:.4g}")
