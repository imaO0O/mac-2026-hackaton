# -*- coding: utf-8 -*-
"""Таблица «пропуски при равной строгости» в документации сверяется с отчётом.

Эта таблица сменила вывод защиты про нейросеть на горизонте 2 часа, и её
предшественница жила в черновике: число «49 % против 71 %» из неё стояло в README
и оказалось посчитанным на модели, затёртой проверкой сидов. Теперь таблицу даёт
``scripts/compare_decision_curves.py``, а этот тест не даёт документации уехать от
отчёта.

Как и сверка таблицы метрик, тест не требует свежести сам: устаревший отчёт уже
ловит ``test_report_freshness``.
"""
from __future__ import annotations

import json

import pytest

from nefte.config import ROOT
from nefte.models.dataset import FEATURE_VERSION

DOC = ROOT / "docs" / "GPU_MODELS.md"
REPORT = ROOT / "reports" / "decision_curves.json"
HEADER = "| ложных тревог не больше |"


def _doc_table() -> tuple[list[str], dict[str, list[int | None]]]:
    lines = DOC.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(HEADER)), None)
    if start is None:
        pytest.fail(f"в {DOC.name} нет таблицы с заголовком «{HEADER}»")
    levels = [c.strip().replace(" ", "") for c in lines[start].strip("|").split("|")[1:]]
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
        rows[cells[0]] = [None if c in {"—", "-", ""} else int(c.replace("%", "").strip())
                          for c in cells[1:]]
    return levels, rows


def test_decision_table_matches_report():
    if not REPORT.exists():
        pytest.skip("нет reports/decision_curves.json — запустите compare_decision_curves.py")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("feature_version") != FEATURE_VERSION:
        pytest.skip("отчёт снят на другой матрице — об этом говорит test_report_freshness")

    levels, rows = _doc_table()
    models = report["модели"]
    assert set(rows) == set(models), (
        f"строки таблицы {sorted(rows)} не совпадают с моделями отчёта {sorted(models)}")
    for label, cells in rows.items():
        stored = models[label]["пропуски при ложных не выше"]
        for level, cell in zip(levels, cells):
            value = stored.get(level)
            expected = None if value is None else round(100 * value)
            assert cell == expected, (
                f"{label}, ложных тревог ≤{level}: в документации {cell} %, "
                f"в отчёте {expected} % — перенесите число из "
                "reports/decision_curves.json")


VERDICT_HEADER = "| против рабочего бустинга, ложных не больше |"
VERDICT_ROWS = {"сеть, сиды 42–44": "сеть h2, сиды 42–44",
                "сеть, сиды 100–102": "сеть h2, сиды 100–102",
                "сеть, сиды 200–202": "сеть h2, сиды 200–202"}


def test_significance_table_matches_report():
    """Вывод «лучше / неразличимо» и разница в скобках — из парного бутстрэпа."""
    if not REPORT.exists():
        pytest.skip("нет reports/decision_curves.json")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("feature_version") != FEATURE_VERSION:
        pytest.skip("отчёт снят на другой матрице")
    lines = DOC.read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(VERDICT_HEADER)), None)
    assert start is not None, f"в {DOC.name} нет таблицы «{VERDICT_HEADER}»"
    levels = [c.strip().replace(" ", "") for c in lines[start].strip("|").split("|")[1:]]
    seen = set()
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
        label = cells[0]
        seen.add(label)
        stored = report["модели"][VERDICT_ROWS[label]]["против «бустинг h0»"]
        for level, cell in zip(levels, cells[1:]):
            want = stored[level]
            assert cell.split(" (")[0].strip() == want["вывод"], (
                f"{label}, ложных ≤{level}: в документации «{cell}», в отчёте «{want['вывод']}»")
            if "(" in cell:
                shown = float(cell.split("(")[1].rstrip(" %)").replace("−", "-"))
                assert abs(shown - 100 * want["разница"]) <= 0.5 + 1e-9, (
                    f"{label}, ложных ≤{level}: разница {shown} %, в отчёте "
                    f"{100 * want['разница']:.1f} %")
    assert seen == set(VERDICT_ROWS), f"строки таблицы значимости: {sorted(seen)}"
