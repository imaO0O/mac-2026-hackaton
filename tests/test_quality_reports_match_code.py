# -*- coding: utf-8 -*-
"""Отчёты моделей качества сняты ТЕКУЩИМ кодом, а не только на текущей матрице.

Контракт свежести сверяет версию матрицы признаков. Он не видит другого
расхождения: код начал писать в отчёт новое поле или считать поле иначе, а отчёты
остались от прежнего запуска. Так и было — «рабочие» precision и recall стали
считаться при точном пороге тревоги вместо округлённого и записывать порог в
``spec_threshold``, а все десять отчётов моделей качества хранили старое значение
(precision 0.388 вместо 0.382 на тесте горизонта 0). Нашёл участник 2, переобучая
модели у себя, — тест свежести молчал, потому что версия матрицы та же.

Проверка — по полю, которое пишет только новый код, и по его смыслу: рабочие
precision и recall обязаны совпадать со значениями при записанном пороге.
"""
from __future__ import annotations

import json

import pytest

from nefte.config import ROOT

REPORTS = sorted((ROOT / "reports").glob("quality_metrics_*.json"))


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_quality_report_is_written_by_current_code(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for split, block in data.get("splits", {}).items():
        model = block.get("model", {})
        assert "spec_threshold" in model, (
            f"{path.name} [{split}]: нет spec_threshold — отчёт снят кодом, который "
            "считал рабочие precision/recall при округлённом пороге. Переобучите: "
            "bash scripts/retrain_quality_all.sh")
        assert abs(model["spec_threshold"] - model["alarm_threshold"]) < 1e-12, (
            f"{path.name} [{split}]: рабочие метрики посчитаны не при пороге тревоги")
