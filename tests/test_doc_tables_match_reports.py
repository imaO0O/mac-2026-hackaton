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
        # Обратная косая перед чертой внутри ячейки — символ, а не граница столбца:
        # в таблицах формул АВТ точка отбора пишется «АВТ\|3». Без этого все ячейки
        # правее сдвигались на одну, и сверка шла не с тем столбцом.
        escaped_pipe = chr(92) + "|"
        text = line.strip().strip("|").replace(escaped_pipe, "¦")
        cells = [c.strip().replace("¦", "|") for c in text.split("|")]
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


def t95(split: str, kind: str) -> dict:
    return report("t95_risk_calibration.json")["выборки"][split][kind]


def t95_row(split: str) -> list[list[float] | None]:
    raw = t95(split, "как_есть")
    over = round(raw["частота"] * raw["анализов"])
    s = raw["наклон"]
    return [[over, raw["анализов"]], [100 * raw["средняя_заявленная"]], [100 * raw["частота"]],
            [s["b"], s["b_от"], s["b_до"]], [raw["Brier"]], [raw["Brier_константы"]]]


def t95_fix_row(split: str) -> list[list[float]]:
    raw, fixed = t95(split, "как_есть"), t95(split, "с_поправкой")
    s = fixed["наклон"]
    return [[raw["Brier"]], [fixed["Brier"]], [raw["Brier_константы"]],
            [s["b"], s["b_от"], s["b_до"]]]


def edge(variant: str) -> list[list[float]]:
    v = report("upper_edge_risk.json")["итог"][variant]
    return [[v["принят_на_сидах"]], [v["на_тесте_хуже_по_полноте_или_форме"]]]


def seq_h2(name: str) -> list[list[float]]:
    d = report(name)["splits"]
    return [[d["val"]["model"]["roc_auc"]], [d["test"]["model"]["roc_auc"]],
            [d["test"]["model"]["MAE"]], [d["test"]["model"]["coverage_80"]]]


def boost_h2_seeds() -> list[list[float] | None]:
    m = report("feature_stability_h2.json")["метрики"]
    auc, mae = [r["ROC-AUC"] for r in m], [r["MAE"] for r in m]
    return [None, [min(auc), max(auc)], [min(mae), max(mae)], None]


def boost_h2_working() -> list[list[float]]:
    d = report("quality_metrics_h2.json")["splits"]
    return [[d["val"]["model"]["roc_auc"]], [d["test"]["model"]["roc_auc"]],
            [d["test"]["model"]["MAE"]], [d["test"]["model"]["coverage_80"]]]


def headline(block: str, key: str) -> list[list[float]]:
    v = report("headline_intervals.json")["числа"][block][key]
    return [[v["значение"]], v["90% интервал"]]


def event(run: str, window: str) -> dict:
    return report("event_response.json")["прогоны"][run]["окна"][window]


def event_row_percent(window: str) -> list[list[float]]:
    e = event("test_period_step1h.json", window)
    return [[100 * e["перед_превышением"]], [100 * e["перед_нормой"]], [100 * e["разница"]],
            [100 * x for x in e["90% интервал"]]]


def step_row(run: str) -> list[list[float]]:
    rows = report(run)["summary"]["порог вмешательства"]["перебор"]
    w = next(r for r in rows if r["рабочий"])
    per_day = report("event_response.json")["прогоны"][run]["вмешательств_в_сутки"]
    return [[100 * w["доля пропусков"]], [100 * w["доля ложных тревог"]], [per_day]]


def ablation(h: int, key: str) -> list[list[float]]:
    """Строка абляции формул справочника: [без ВАК], [с ВАК]."""
    def value(name: str) -> float:
        data = report(name)
        return data["n_features"] if key == "n_features" else data["splits"]["test"]["model"][key]
    return [[value(f"quality_metrics_h{h}_novak.json")], [value(f"quality_metrics_h{h}.json")]]


QA, HC, OA, GM = "QUALITY_AGENT.md", "HARD_CHECKS.md", "OPTIMIZER_AGENT.md", "GPU_MODELS.md"
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
    Row(HC, "| шаг | пропуски на рабочей точке |", "12 ч", lambda: step_row("test_period.json")),
    Row(HC, "| шаг | пропуски на рабочей точке |", "1 ч", lambda: step_row("test_period_step1h.json")),
    *[Row(HC, "| окно относительно отбора, шаг 1 ч |", label, (lambda w=w: event_row_percent(w)))
      for label, w in (("за 2 ч до отбора", "[-2, +0)"),
                       ("от отбора до публикации (+4 ч)", "[+0, +4)"),
                       ("от −2 до +4 ч", "[-2, +4)"),
                       ("за 6 ч до отбора", "[-6, +0)"),
                       ("за сутки до отбора", "[-24, +0)"))],
    Row("DEFENSE_QUALITY.md", "| утверждение | число | 90 % интервал |",
        "реакция на пробу с превышением, окно −2 … +4 ч от отбора",
        lambda: [[event("test_period_step1h.json", "[-2, +4)")["перед_превышением"]], None, None,
                 [1, event("test_period_step1h.json", "[-2, +4)")["перед_нормой"]]]),
    Row("DEFENSE_QUALITY.md", "| утверждение | число | 90 % интервал |",
        "разница с нормальной пробой в том же окне",
        lambda: [[event("test_period_step1h.json", "[-2, +4)")["разница"]],
                 event("test_period_step1h.json", "[-2, +4)")["90% интервал"], None, None]),
    Row("DEFENSE_QUALITY.md", "| утверждение | число | 90 % интервал |",
        "то же за сутки до отбора",
        lambda: [[event("test_period_step1h.json", "[-24, +0)")["разница"]],
                 event("test_period_step1h.json", "[-24, +0)")["90% интервал"], None, None]),
    # шпаргалка защиты: ровно эти числа произносятся вслух
    *[Row("DEFENSE_QUALITY.md", "| утверждение | число | 90 % интервал |", label,
          (lambda b=b, k=k: headline(b, k) + [None, None]))
      for label, b, k in (
          ("точность модели серы, nowcast", "бустинг h0", "MAE"),
          ("модель точнее поточного анализатора, nowcast", "бустинг h0", "MAE минус MAE ПАК"),
          ("то же на двух часах", "бустинг h2", "MAE минус MAE ПАК"),
          ("различение превышений, nowcast", "бустинг h0", "roc_auc"),
          ("полнота тревоги на рабочем пороге", "бустинг h0", "recall"),
          ("пропуски превышений на прогоне по тесту", "решения, рабочая точка бустинга",
           "доля пропусков"),
          ("ложные тревоги там же", "решения, рабочая точка бустинга", "доля ложных тревог"))],
    *[Row(QA, "| число | значение | 90 % интервал |", label,
          (lambda b=b, k=k: headline(b, k)))
      for label, b, k in (
          ("MAE, горизонт 0", "бустинг h0", "MAE"),
          ("MAE модели минус MAE ПАК, горизонт 0 (парно)", "бустинг h0", "MAE минус MAE ПАК"),
          ("MAE модели минус MAE ПАК, горизонт 2 (парно)", "бустинг h2", "MAE минус MAE ПАК"),
          ("покрытие 80 %, горизонт 0", "бустинг h0", "coverage_80"),
          ("ROC-AUC, горизонт 0", "бустинг h0", "roc_auc"),
          ("полнота тревоги, горизонт 0", "бустинг h0", "recall"),
          ("точность тревоги, горизонт 0", "бустинг h0", "precision"),
          ("ROC-AUC, горизонт 2", "бустинг h2", "roc_auc"),
          ("пропуски на рабочей точке (прогон по тесту)", "решения, рабочая точка бустинга",
           "доля пропусков"),
          ("ложные тревоги там же", "решения, рабочая точка бустинга", "доля ложных тревог"))],
    *[Row(GM, "| Модель, горизонт 2 ч |", label, (lambda f=f: seq_h2(f)))
      for label, f in (("TCN 48 + предобучение, сиды 42–44", "sequence_metrics_tcn48_pre_h2.json"),
                       ("то же, сиды 100–102", "sequence_metrics_tcn48_pre_s100_h2.json"),
                       ("то же, сиды 200–202", "sequence_metrics_tcn48_pre_s200_h2.json"),
                       ("TCN 48 без предобучения", "sequence_metrics_tcn48_h2.json"))],
    Row(GM, "| Модель, горизонт 2 ч |", "CatBoost, рабочая модель (сид 42)", boost_h2_working),
    Row(GM, "| Модель, горизонт 2 ч |", "CatBoost, пять сидов 42–46", boost_h2_seeds),
    *[Row(QA, "| | превышений | заявлено в среднем |", label, (lambda sp=sp: t95_row(sp)))
      for label, sp in (("обучение", "train"), ("валидация", "val"), ("тест", "test"))],
    *[Row(QA, "| | Brier как есть | Brier с поправкой |", label, (lambda sp=sp: t95_fix_row(sp)))
      for label, sp in (("валидация", "val"), ("тест (проверка)", "test"))],
    *[Row(QA, "| вариант | принят (сидов из 3) |", name, (lambda name=name: edge(name)))
      for name in ("эмпирический хвост", "классификатор", "смесь средним", "смесь рангов",
                   "хвост по квантилям")],
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


# --------------------------------------------------------------------------- #
# Участник 2: надёжность, ресурс катализатора, формулы АВТ.
#
# Добавлено по той же причине, что и всё выше: в этих документах числа тоже
# переносились руками, и при первом же прогоне сверки нашлись устаревшие ячейки —
# после причинной маски «залипшего» сигнала сдвинулись доля брака, медианы тегов и
# число анализов. Таблицы из разовых разборов без отчёта (разложение нормировки,
# расхождение масок) сюда не входят: сверять их не с чем.
# --------------------------------------------------------------------------- #

def catalyst() -> dict:
    return report("catalyst_life.json")


def avt_tags() -> dict:
    return report("avt_formulas.json")


def reliability() -> dict:
    return report("reliability_metrics.json")


def outage(start: str) -> list[list[float] | None]:
    r = next(x for x in catalyst()["остановы"] if x["начало"] == start)
    return [[r["длительность, ч"]], [r["NWABT до, °C"]], [r["NWABT после, °C"]],
            [r["шаг, °C"]], None]


def cycle(number: int) -> list[list[float] | None]:
    c = next(x for x in catalyst()["циклы"] if x["цикл"] == number)
    return [None, [c["длительность, сут"]], [c["анализов"]],
            [c["NWABT в начале, °C"], c["NWABT в конце, °C"]],
            [c["скорость, °C/мес"]], list(c["95% ДИ"])]


def sensitivity(axis: str, keys: tuple[str, ...]) -> list[list[float]]:
    values = catalyst()["чувствительность"][axis]
    return [[values[k]] for k in keys]


def remaining(method: str) -> list[list[float] | None]:
    r = catalyst()["остаточный_ресурс_мес"]
    if method == "а":
        return [[r["по средней скорости"], *r["по средней скорости, ДИ"]], None]
    if method == "б":
        left = {row["цикл"]: row["оставалось, мес"] for row in r["по аналогии"]}
        # в ячейке «13.0 мес (цикл 1), 14.4 (цикл 0, …)» номера циклов — тоже числа
        return [[left[1], 1, left[0], 0], None]
    if method == "в":
        return [[r["по текущей скорости"]], None]
    by_lag = r["по отставанию от эталона"]
    return [[by_lag["остаток, мес"]],
            [by_lag["эталон прожил ещё, мес"], by_lag["отставание, °C"],
             by_lag["отставание, мес наработки"]]]


def lag(day: int) -> list[list[float] | None]:
    rows = catalyst()["контрольная_точка"]["отставание"]
    r = next(x for x in rows if x["сутки"] == day)
    return [[r["цикл 1, °C"]], [r["цикл 2, °C"]], [r["отставание, °C"]],
            list(r["95% ДИ"]), None]


def backtest(day: int) -> list[list[float]]:
    r = next(x for x in catalyst()["бэктест_метода"] if x["сутки"] == day)
    return [[r["NWABT тогда, °C"]], [r["(а) линейно, мес"]], [r["(б) аналогия, мес"]],
            [r["ФАКТ, мес"]], [r["ошибка (а), %"]]]


def cetane_within(number: int) -> list[list[float]]:
    rows = catalyst()["цетановое_число"]["внутри циклов"]
    return [[next(x for x in rows if x["цикл"] == number)["наклон, ед/год"]]]


def cetane_jump(date: str) -> list[list[float]]:
    rows = catalyst()["цетановое_число"]["шаг при смене"]
    r = next(x for x in rows if x["смена"] == date)
    return [[r["ЦЧ до"]], [r["ЦЧ после"]], [r["шаг"]]]


def percent(text) -> float:
    return float(str(text).replace("%", "").strip())


def avt_formula(name: str) -> list[list[float] | None]:
    r = next(x for x in avt_tags()["формулы"] if x["формула"] == name)
    # точка отбора («АВТ|3») числом не сверяется; остальное — против СВОЕЙ точки блока
    head = [[r["медиана"]], [percent(r["в физ. диапазоне"])], None]
    if r.get("n (своя)") is None:
        return head + [None, None, None, None]
    return head + [[r["n (своя)"]], [r["смещение (своя)"]], [r["MAE (своя)"]],
                   [r["corr (своя)"]]]


def broken(name: str) -> dict:
    return next(x for x in avt_tags()["разбор_сломанных"] if x["формула"] == name)


def cfpp_reading(index: int) -> list[list[float]]:
    r = broken("AVT6:240-350:CFPP")["проверка"][index]
    return [[percent(r["в диапазоне"])], [r["смещение"]], [r["MAE"]]]


def i350_constant(index: int) -> list[list[float]]:
    r = broken("AVT6:350:I350")["проверка"][index]
    return [[r["смещение"]], [r["MAE"]]]


def tag_verdict(code: str) -> list[list[float] | None]:
    r = next(x for x in avt_tags()["вердикты_по_тегам"] if x["тег"] == code)
    return [None, [r["медиана"]], [r["p05"], r["p95"]], None, None]


def validity(unit: str) -> list[list[float] | None]:
    v = reliability()["validity"][unit]
    # «Мёртвых» числом не сверяется: в ячейке «1 (`D10`)» имя тега тоже число
    return [[v["n_tags"]], None, [len(v["suspicious_negative"])],
            [v["mean_bad_share_pct"]]]


def analyzer_failure(start: str) -> list[list[float]]:
    rows = reliability()["analyzer_failures"]["longest"]
    r = next(x for x in rows if x["start"].startswith(start))
    return [[r["value"]], [r["hours"]]]


def tradeoff(split: str) -> list[list[float]]:
    t = reliability()["tradeoff_quality_vs_severity"][split]
    return [[t["auc_severity"]], [t["auc_inverse"]]]


def robustness(spread: int) -> list[list[float] | None]:
    name = ("severity_robustness.json" if spread == 50
            else f"severity_robustness_s{spread}.json")
    data = report(name)
    rc, decision = data["risk_class"], data["decision"]
    near = (None if rc["устойчивость у границы"] is None
            else [100 * rc["устойчивость у границы"], rc["моментов у границы"],
                  rc["stamps"]])

    def outcome(label: str) -> list[float] | None:
        o = decision["по исходам"].get(label)
        if o is None:
            return None
        return [100 * o["устойчивость"], 100 * o["худший момент"], o["моментов"]]

    # Сколько моментов с действием лежат у границы класса. Без этого числа «исход
    # сохраняется» не отличить от «пограничных случаев в выборке просто нет».
    from nefte.agents.reliability import ReliabilityAgent
    low, high = reliability()["risk_thresholds"]
    margin = ReliabilityAgent.CLASS_BOUNDARY_MARGIN
    acting_near = sum(
        min(abs(r["severity"] - low), abs(r["severity"] - high)) <= margin
        for r in decision["per_stamp"] if r["базовый исход"] == "меняем уставки")
    return [[100 * rc["mean_stability"], 100 * rc["worst_stability"]], near,
            outcome("меняем уставки"), outcome("держим режим"),
            [decision["моментов с действием"], decision["stamps"], acting_near]]


CL, AT, RA = "CATALYST_LIFE.md", "AVT_TAGS.md", "RELIABILITY_AGENT.md"
VF = "VAK_FEATURES.md"
REMAINING = "| Способ | Оценка | На чём держится |"
EA_KEYS = ("70.0", "85.0", "100.0", "115.0", "130.0")
SULFUR_KEYS = ("5.0", "8.0", "10.0")
AVT_FORMULAS = ("AVT6:240-350:D15", "AVT6:240-350:T50", "AVT6:240-350:EBP",
                "AVT6:240-350:CFPP", "AVT6:350:T50", "AVT6:350:I350", "AVT6:350:D15",
                "AVT6:350-500:ViscosityK", "AVT6:350:CFPP")
DISPUTED_TAGS = ("F30", "F31", "F32", "F36", "F57", "F64", "F65", "L43", "P4", "P67",
                 "T6", "T11", "T15", "T18", "T33", "T37", "T40", "T42", "T48", "T58")

ROWS += [
    *[Row(CL, "| Останов | Длительность |", label, (lambda s=start: outage(s)))
      for label, start in (("2024-03-16 … 04-17", "2024-03-16"),
                           ("2026-04-15 … 04-23", "2026-04-15"),
                           ("2026-06-20 … 06-30", "2026-06-20"))],
    *[Row(CL, "| Цикл | Период | Длина |", str(n), (lambda n=n: cycle(n)))
      for n in (0, 1, 2)],
    Row(CL, "| Энергия активации, кДж/моль |", "Скорость, °C/мес",
        lambda: sensitivity("энергия активации", EA_KEYS)),
    Row(CL, "| Эталонная сера, мг/кг |", "Скорость, °C/мес",
        lambda: sensitivity("эталонная сера", SULFUR_KEYS)),
    Row(CL, REMAINING, "(а) по средней скорости", lambda: remaining("а")),
    Row(CL, REMAINING, "(б) по аналогии, сопоставление по УРОВНЮ", lambda: remaining("б")),
    Row(CL, REMAINING, "(в) по текущей скорости +4.15 °C/мес", lambda: remaining("в")),
    Row(CL, REMAINING, "(г) по отставанию, сопоставление по НАРАБОТКЕ",
        lambda: remaining("г")),
    *[Row(CL, "| Сутки | Цикл 1, °C |", str(d), (lambda d=d: lag(d)))
      for d in (40, 60, 80, 100)],
    *[Row(CL, "| Сутки цикла 1 | NWABT тогда |", str(d), (lambda d=d: backtest(d)))
      for d in (108, 150, 200, 300, 400, 500)],
    Row(CL, "| | Наклон, ед/год |", "по календарю, все 42 анализа",
        lambda: [[catalyst()["цетановое_число"]["наклон по календарю, ед/год"]]]),
    Row(CL, "| | Наклон, ед/год |", "внутри цикла 0 (17 анализов)",
        lambda: cetane_within(0)),
    Row(CL, "| | Наклон, ед/год |", "внутри цикла 1 (22 анализа)",
        lambda: cetane_within(1)),
    *[Row(CL, "| Смена катализатора | ЦЧ до |", date, (lambda d=date: cetane_jump(d)))
      for date in ("2024-04-17", "2026-04-23")],
    *[Row(AT, "| Формула | Медиана | В физ. диапазоне |", name,
          (lambda n=name: avt_formula(n)))
      for name in AVT_FORMULAS],
    Row(AT, "| Прочтение | В диапазоне", "как исправили организаторы: `F65/F32+F30`",
        lambda: cfpp_reading(0)),
    Row(AT, "| Прочтение | В диапазоне", "по аналогии с D15: `F65/(F32+F30)`",
        lambda: cfpp_reading(1)),
    Row(AT, "| Константа | Смещение | MAE |", "39.562 (как выдано)",
        lambda: i350_constant(0)),
    Row(AT, "| Константа | Смещение | MAE |", "399.562", lambda: i350_constant(1)),
    *[Row(AT, "| Тег | Описание КИП | Медиана |", code, (lambda c=code: tag_verdict(c)))
      for code in DISPUTED_TAGS],
    *[Row(RA, "| Установка | Тегов | Мёртвых |", label, (lambda u=unit: validity(u)))
      for label, unit in (("ЭЛОУ-АВТ-6", "avt"), ("24-2000", "ht"))],
    *[Row(RA, "| Начало | Значение | Длительность |", start,
          (lambda s=start: analyzer_failure(s)))
      for start in ("2024-03-16", "2026-06-18", "2026-04-15")],
    *[Row(RA, "| Выборка | ROC-AUC severity |", split, (lambda s=split: tradeoff(s)))
      for split in ("train", "test")],
    Row(RA, "| | Доля нетипичных часов |", "train (2023 — июнь 2025)",
        lambda: [[reliability()["anomaly_detector"]["train_%"], 99]]),
    Row(RA, "| | Доля нетипичных часов |", "test (2026)",
        lambda: [[reliability()["anomaly_detector"]["test_%"]]]),
    *[Row(RA, "| Разброс весов | risk_class сохраняется |", label,
          (lambda s=spread: robustness(s)))
      for label, spread in (("±20 %", 20), ("±50 %", 50))],
    # Абляция формул справочника. Таблицы простояли на старой сборке матрицы и
    # тестами не сверялись: при пересчёте на горизонте 0 разница сменила знак, а
    # «прирост ROC-AUC на 0.09» на горизонте 2 ч сжался до 0.009.
    *[Row(VF, "| Метрика (test, горизонт 0) |", label, (lambda k=key: ablation(0, k)))
      for label, key in (("MAE, мг/кг", "MAE"), ("ROC-AUC риска", "roc_auc"),
                         ("Покрытие 80 % интервала", "coverage_80"),
                         ("Precision тревоги", "spec_precision"),
                         ("Recall тревоги", "spec_recall"),
                         ("Признаков после отбора", "n_features"))],
    *[Row(VF, "| Метрика (test, горизонт 2) |", label, (lambda k=key: ablation(2, k)))
      for label, key in (("MAE, мг/кг", "MAE"), ("ROC-AUC риска", "roc_auc"),
                         ("Recall тревоги", "spec_recall"),
                         ("Признаков после отбора", "n_features"))],
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
