# -*- coding: utf-8 -*-
"""Таблица сравнения моделей в документации обязана сходиться с отчётами.

Это главная таблица защиты — «сеть против бустинга», — и до сих пор она жила
отдельной жизнью: числа в неё переносились руками. Один раз это уже вышло боком
с таблицей бюджета тревог: там числа были получены разовым счётом, не попавшим ни
в один скрипт, и при пересчёте уехали все до одного.

Разница между «переписал из отчёта» и «сверяется тестом» вся в том, заметит ли
кто-нибудь расхождение. Переписывание руками надёжно ровно один раз — в момент
переписывания.

Проверка намеренно НЕ требует свежести сама: если отчёт снят на другой матрице,
тест пропускается, потому что об этом уже кричит `test_report_freshness`. Иначе
одна причина давала бы два падения, и второе пришлось бы разбирать заново.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nefte.config import ROOT
from nefte.models.dataset import FEATURE_VERSION

DOC = ROOT / "docs" / "GPU_MODELS.md"

# строка таблицы -> отчёт, из которого берутся её числа
ROWS = {
    "GRU, окно 24": "sequence_metrics_gru24_h0.json",
    "GRU, окно 48": "sequence_metrics_gru48_h0.json",
    "TCN, окно 24": "sequence_metrics_tcn24_h0.json",
    "TCN, окно 48": "sequence_metrics_tcn48_h0.json",
    "TCN 48 + предобучение": "sequence_metrics_tcn48_pre_h0.json",
    "CatBoost (рабочая)": "quality_metrics_h0.json",
}
# порядок колонок таблицы
COLUMNS = [("val", "MAE"), ("val", "roc_auc"),
           ("test", "MAE"), ("test", "roc_auc"), ("test", "coverage_80")]


def _numbers_in(cell: str) -> float | None:
    """Число из ячейки, игнорируя markdown-жирность и прочерки."""
    cleaned = cell.replace("*", "").strip()
    if cleaned in {"", "—", "-"}:
        return None
    try:
        return float(cleaned.replace(",", "."))
    except ValueError:
        return None


def _doc_rows() -> dict[str, list[float | None]]:
    out = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        name = cells[0].replace("*", "").strip()
        if name in ROWS and len(cells) >= 6:
            out[name] = [_numbers_in(c) for c in cells[1:6]]
    return out


@pytest.mark.parametrize("name", sorted(ROWS))
def test_table_row_matches_its_report(name: str):
    path = ROOT / "reports" / ROWS[name]
    if not path.exists():
        pytest.skip(f"нет отчёта {ROWS[name]}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("feature_version") != FEATURE_VERSION:
        pytest.skip("отчёт снят на другой матрице — об этом говорит "
                    "test_report_freshness, дублировать не нужно")

    rows = _doc_rows()
    assert name in rows, f"строка «{name}» пропала из таблицы в {DOC.name}"

    for (split, key), claimed in zip(COLUMNS, rows[name]):
        actual = report["splits"][split]["model"].get(key)
        if claimed is None or actual is None:
            continue
        assert claimed == pytest.approx(actual, abs=0.0015), (
            f"{name}, {split} {key}: в документации {claimed}, в отчёте "
            f"{actual:.4f} — таблицу переписывали руками и она отстала")


def test_the_table_is_still_there():
    """Сама таблица не должна тихо исчезнуть — иначе проверки выше пропускаются."""
    rows = _doc_rows()
    assert len(rows) >= 5, (
        f"в таблице {DOC.name} опознано строк: {len(rows)} — разметка изменилась, "
        "и сверка перестала что-либо проверять")
