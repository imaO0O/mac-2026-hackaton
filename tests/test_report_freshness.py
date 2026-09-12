"""Закоммиченные отчёты должны соответствовать текущему коду и конфигу.

README обещает, что числа воспроизводятся командой. Обещание легко потерять
молча: поменяли порог в конфиге или расчёт признаков — а отчёты в репозитории
остались прежними, и на защите звучат числа, которых код уже не даёт.

Такое в проекте случалось: `quality_metrics_h2.json` однажды не воспроизводился
командой из README, и выяснилось это случайно. Эти тесты ловят тот же класс
расхождений сразу.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nefte.config import ROOT, load_config
from nefte.models.dataset import FEATURE_VERSION

REPORTS = sorted((ROOT / "reports").glob("quality_metrics_*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_report_matches_the_current_feature_version(path: Path):
    """Отчёт, снятый на другой матрице признаков, — не отчёт, а история.

    Версия поднимается, когда меняется САМ РАСЧЁТ признаков. Если она разошлась,
    числа в отчёте получены не тем кодом, который лежит рядом.
    """
    report = _load(path)
    version = report.get("feature_version")
    if version is None:
        pytest.skip("отчёт снят до того, как версия стала записываться")
    assert version == FEATURE_VERSION, (
        f"{path.name} снят на матрице версии {version}, а код даёт "
        f"{FEATURE_VERSION}: перезапустите scripts/train_quality.py")


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_report_matches_the_current_limits(path: Path):
    """Предел показателя и бюджет тревог берутся из конфига, а не из памяти."""
    cfg = load_config()
    report = _load(path)
    target = report.get("target", "sulfur")
    expected_limit = (cfg["spec"]["product_sulfur_mgkg"]["max"] if target == "sulfur"
                      else cfg["spec"]["t95_c"]["max"])
    if "limit" in report:
        assert report["limit"] == pytest.approx(float(expected_limit))
    if report.get("alarm_budget") is not None:
        assert report["alarm_budget"] == pytest.approx(
            float(cfg["quality"]["alarm_budget"]))


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_report_matches_the_current_split(path: Path):
    """Границы train/val/test — часть постановки, а не деталь прогона.

    Сдвинули разбиение — прежние метрики относятся к другому эксперименту.
    """
    cfg = load_config()
    split = _load(path).get("split")
    if split is None:
        pytest.skip("отчёт снят до того, как разбиение стало записываться")
    for name in ("train", "val", "test"):
        assert list(split[name]) == list(cfg["split"][name]), (
            f"{path.name}: разбиение {name} разошлось с конфигом")


def test_there_are_reports_to_check():
    """Сам список не должен молча опустеть — иначе тесты выше ничего не проверяют."""
    assert REPORTS, "в reports/ нет ни одного quality_metrics_*.json"
