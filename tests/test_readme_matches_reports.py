# -*- coding: utf-8 -*-
"""Числа README, относящиеся к агенту качества и сравнению моделей, сходятся с отчётами.

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


@lru_cache(maxsize=1)
def readme() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
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

NUM = r"([-+−]?\d+(?:[.,]\d+)?)"


def _pattern(template: str) -> re.Pattern:
    parts = [re.escape(p) for p in template.split("{}")]
    return re.compile(NUM.join(parts))


@pytest.mark.parametrize("template,expected", CLAIMS, ids=[c[0][:40] for c in CLAIMS])
def test_readme_claim_matches_reports(template, expected):
    found = _pattern(template).findall(readme())
    assert len(found) == 1, (
        f"фраза «{template}» найдена в README {len(found)} раз: если её переписали, "
        "обновите и фразу, и эту сверку")
    got = found[0] if isinstance(found[0], tuple) else (found[0],)
    want = expected()
    assert len(got) == len(want)
    for text, target in zip(got, want):
        text = text.replace("−", "-").replace(",", ".")
        decimals = len(text.split(".")[1]) if "." in text else 0
        assert abs(float(text) - target) <= 0.5 * 10 ** -decimals + 1e-9, (
            f"«{template}»: в README {text}, в отчёте {target:.4g}")
