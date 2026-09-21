# -*- coding: utf-8 -*-
"""Числа README и журнала находок сходятся с отчётами.

14.09 README стал витриной на одну страницу, а подробный журнал переехал в
`docs/FINDINGS.md`. Сверяются оба: витрина — отдельным списком `README_CLAIMS`,
журнал — прежним `CLAIMS`.

README читают первым, а его числа стоят в тексте, не в таблицах, и сверка таблиц их
не видит. За два дня пересчётов числа в README пришлось править руками трижды:
«49 % против 71 %», «1.90 °C и 2.34 мг/кг», «против 0.41 у бустинга».

Каждая проверка — фраза README с числами и поля отчётов, из которых эти числа
взяты. Фраза обязана найтись ровно один раз: если её переписали, тест падает и
требует обновить и фразу, и сверку, а не молча перестаёт проверять. Перенос строк
и отступы README не важны — текст сравнивается с одинарными пробелами.

Точность — та, с которой число напечатано. Устаревший отчёт пропускается: об этом
уже говорит `test_report_freshness`.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import pytest

from nefte.config import ROOT
from nefte.models.dataset import FEATURE_VERSION


@lru_cache(maxsize=None)
def report(name: str) -> dict:
    path = ROOT / "reports" / name
    if not path.exists():
        pytest.skip(f"нет reports/{name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "feature_version" in data and data["feature_version"] != FEATURE_VERSION:
        pytest.skip(f"reports/{name} снят на другой матрице")
    return data


@lru_cache(maxsize=None)
def document(name: str) -> str:
    text = (ROOT / name).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text.replace("*", ""))


def split0(key):
    return report("quality_metrics_h0.json")["splits"]["test"][key]


def seq_h0(field: str) -> list[float]:
    names = ("gru24", "gru48", "tcn24", "tcn48", "tcn48_pre")
    return [report(f"sequence_metrics_{n}_h0.json")["splits"]["test"]["model"][field]
            for n in names]


def seq_h2_auc() -> list[float]:
    return [report(f"sequence_metrics_{n}_h2.json")["splits"]["test"]["model"]["roc_auc"]
            for n in ("tcn48_pre", "tcn48_pre_s100", "tcn48_pre_s200")]


def stability_auc(h: int) -> list[float]:
    return [r["ROC-AUC"] for r in report(f"feature_stability_h{h}.json")["метрики"]]


def working_miss() -> list[float]:
    rows = report("test_period.json")["summary"]["порог вмешательства"]["перебор"]
    return [100 * next(r for r in rows if r["рабочий"])["доля пропусков"]]


def curve_range(prefix: str, level: str) -> list[float]:
    models = report("decision_curves.json")["модели"]
    vals = [100 * m["пропуски при ложных не выше"][level]
            for label, m in models.items() if label.startswith(prefix)]
    return [min(vals), max(vals)]


def seq_working_fa() -> list[float]:
    models = report("decision_curves.json")["модели"]
    vals = [100 * m["рабочая точка"]["доля ложных тревог"]
            for label, m in models.items() if label.startswith("сеть h2")]
    return [min(vals), max(vals)]


def weights() -> dict:
    return report("objective_weights.json")


def event_window() -> dict:
    return report("event_response.json")["прогоны"]["test_period_step1h.json"]["окна"]["[-2, +4)"]


def simulation() -> dict:
    return report("simulation.json")["итог"]


def weaker_physics_interventions() -> list[float]:
    loops = report("kinetic_order.json")["замкнутый_контур"]
    counts = [r["вмешательств"] for r in loops if r["порядок процесса"] > 1.0]
    return [min(counts), max(counts)]


def kinetic_variant(letter: str) -> dict:
    """Вариант кинетики из отчёта по букве: «A: первый порядок…», «D: порядок 1.5»."""
    variants = report("kinetic_strength.json")["варианты"]
    return next(v for name, v in variants.items() if name.startswith(letter + ":"))


def shock(name: str) -> dict:
    """Строка отчёта о чувствительности прогноза к возмущению сценария."""
    rows = report("shock_sensitivity.json")["возмущения"]
    return next(row for row in rows if row["возмущение"] == name)


def first_order_response() -> float:
    rows = report("kinetic_order.json")["отклик_на_градус"]
    return next(r for r in rows if r["порядок"] == 1.0)["Δ серы на +1 °C, %"]


CLAIMS = [
    ("MAE {} мг/кг на тесте против {} у поточного анализатора",
     lambda: [split0("model")["MAE"], split0("baseline_pak")["MAE"]]),
    ("ROC-AUC риска {}, покрытие интервала {}, полнота тревоги {} против {} у ПАК",
     lambda: [split0("model")["roc_auc"], split0("model")["coverage_80"],
              split0("model")["spec_recall"], split0("baseline_pak")["spec_recall"]]),
    ("ROC-AUC гуляет от {} до {} на часовом горизонте",
     lambda: [min(stability_auc(1)), max(stability_auc(1))]),
    ("ROC-AUC {}–{} против {} и покрытие интервала {}–{} против {}",
     lambda: [min(seq_h0("roc_auc")), max(seq_h0("roc_auc")), split0("model")["roc_auc"],
              min(seq_h0("coverage_80")), max(seq_h0("coverage_80")),
              split0("model")["coverage_80"]]),
    ("{}–{} против {}, то есть лучшая из сетей (GRU-48) даёт {} — ничья",
     lambda: [min(seq_h0("MAE")), max(seq_h0("MAE")), split0("model")["MAE"],
              min(seq_h0("MAE"))]),
    ("ROC-AUC {}–{} на трёх тройках сидов; у бустинга на пяти сидах {}–{} (медиана {})",
     lambda: [min(seq_h2_auc()), max(seq_h2_auc()), min(stability_auc(2)),
              max(stability_auc(2)), sorted(stability_auc(2))[2]]),
    ("диапазоны перекрываются (сеть {}–{} %, бустинг {}–{} %)",
     lambda: curve_range("сеть h2", "25%") + curve_range("бустинг h0", "25%")),
    ("«{} % пропусков» — число удачного сида", working_miss),
    ("Собственная рабочая точка сети даёт {}–{} % ложных тревог", seq_working_fa),
    ("измерена на {} моментах ({} из них с действием)",
     lambda: [weights()["stamps"], weights()["моментов с действием"]]),
    ("сохраняется в {} % случаев, величина воздействия гуляет на {} °C, прогноз серы на {} мг/кг",
     lambda: [100 * weights()["исход сохраняется"], weights()["ΔT5 разброс, среднее"],
              weights()["сера разброс, среднее"]]),
]

# Витрина: каждое число таблицы результатов и раздела ограничений.
README_CLAIMS = [
    ("MAE {} мг/кг против {} у поточного анализатора, ROC-AUC риска {}",
     lambda: [split0("model")["MAE"], split0("baseline_pak")["MAE"], split0("model")["roc_auc"]]),
    ("реагирует на {} % проб с превышением против {} % нормальных",
     lambda: [100 * event_window()["перед_превышением"], 100 * event_window()["перед_нормой"]]),
    ("одноагентная система двигает уставки в {} раз больше ({} °C против {} за полгода)",
     lambda: [report("architectures.json")["конфигурации"]["одноагентная"]["суммарно °C"]
              / report("architectures.json")["конфигурации"]["полная"]["суммарно °C"],
              report("architectures.json")["конфигурации"]["одноагентная"]["суммарно °C"],
              report("architectures.json")["конфигурации"]["полная"]["суммарно °C"]]),
    ("число вмешательств за месяц — {}, они убирают {} из {} шагов с превышением, вклад в Т95 {} °C",
     lambda: [simulation()["вмешательств"],
              simulation()["сера_без_вмешательства_сим"]["выше предела без нас, шагов"]
              - simulation()["сера_без_вмешательства_сим"]["выше предела с нами, шагов"],
              simulation()["сера_без_вмешательства_сим"]["выше предела без нас, шагов"],
              simulation()["Т95_наш_вклад"]["средний сдвиг"]]),
    ("падает на {} °C/мес (95 % ДИ {}–{})",
     lambda: [report("catalyst_life.json")["скорость_дезактивации"]["°C/мес"],
              *report("catalyst_life.json")["скорость_дезактивации"]["95% ДИ"]]),
    ("{} % «зависаний» поточного анализатора приходятся на остановы",
     lambda: [report("reliability_metrics.json")["downtime"]["explained_%"]]),
    ("прогноз на 2 часа у бустинга не работает (ROC-AUC {})",
     lambda: [report("quality_metrics_h2.json")["splits"]["test"]["model"]["roc_auc"]]),
    ("названного диапазона; первый порядок давал {} — выше практики",
     lambda: [kinetic_variant("A")["обещанное снижение, мг/кг на градус"]]),
    ("кинетика — порядок 1.5, это {} мг/кг на градус",
     lambda: [kinetic_variant("D")["обещанное снижение, мг/кг на градус"]]),
    ("назвал {}–{} мг/кг на градус",
     lambda: report("kinetic_strength.json")["практика_заказчика_мг_кг_на_градус"]),
    ("средние за 6–144 ч ({} % решений на тесте)",
     lambda: [100 * report("risk_attribution.json")["выборки"]["test"]
              ["как часто группа ведущая без измерений серы"]["сырьё с АВТ"]]),
    ("утяжеления сырья на +5 °C всего {} мг/кг",
     lambda: [shock("утяжеление сырья +5 °C")["сдвиг прогноза, мг/кг"]]),
    ("+10 мг/кг к ней двигают прогноз на {}",
     lambda: [shock("сера сырья +10 мг/кг")["сдвиг прогноза, мг/кг"]]),
    ("замкнутый контур делает {}–{} вмешательств вместо {}",
     lambda: weaker_physics_interventions() + [simulation()["вмешательств"]]),
]

NUM = r"([-+−]?\d+(?:[.,]\d+)?)"


def _pattern(template: str) -> re.Pattern:
    parts = [re.escape(p) for p in template.split("{}")]
    return re.compile(NUM.join(parts))


def _check(text: str, where: str, template, expected):
    found = _pattern(template).findall(text)
    assert len(found) == 1, (
        f"фраза «{template}» найдена в {where} {len(found)} раз: если её переписали, "
        "обновите и фразу, и эту сверку")
    got = found[0] if isinstance(found[0], tuple) else (found[0],)
    want = expected()
    assert len(got) == len(want)
    for cell, target in zip(got, want):
        cell = cell.replace("−", "-").replace(",", ".")
        decimals = len(cell.split(".")[1]) if "." in cell else 0
        assert abs(float(cell) - target) <= 0.5 * 10 ** -decimals + 1e-9, (
            f"«{template}»: в {where} {cell}, в отчёте {target:.4g}")


@pytest.mark.parametrize("template,expected", CLAIMS, ids=[c[0][:40] for c in CLAIMS])
def test_findings_claim_matches_reports(template, expected):
    _check(document("docs/FINDINGS.md"), "docs/FINDINGS.md", template, expected)


@pytest.mark.parametrize("template,expected", README_CLAIMS,
                         ids=[c[0][:40] for c in README_CLAIMS])
def test_readme_claim_matches_reports(template, expected):
    _check(document("README.md"), "README", template, expected)
