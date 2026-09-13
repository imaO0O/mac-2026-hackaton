# -*- coding: utf-8 -*-
"""«Рабочие» precision и recall считаются при ТОЧНОМ пороге тревоги.

Раньше они брались по ключу, округлённому до двух знаков: при 0.18 вместо 0.1771.
Доля тревог в той же функции при этом считалась при точном пороге, так что одна и
та же точка отчёта описывала два разных порога. На тесте горизонта 0 разница —
одна тревога: precision 0.388 против 0.382.
"""
import pandas as pd

from nefte.models.quality_model import interval_metrics


def test_spec_metrics_use_the_exact_threshold():
    threshold = 0.1771
    # момент с риском между точным и округлённым порогом — ложная тревога
    risk = pd.Series([0.05, 0.178, 0.30, 0.40])
    y = pd.Series([8.0, 8.0, 11.0, 12.0])
    pred = pd.DataFrame({"q50": [8.0, 8.0, 11.0, 12.0], "sigma": [1.0] * 4})
    out = interval_metrics(pred, y, risk, limit=10.0, alarm_threshold=threshold)
    assert out["spec_threshold"] == threshold
    assert out["spec_precision"] == 2 / 3, "тревога при 0.178 должна считаться"
    assert out["spec_recall"] == 1.0
    # и это то же правило, что у доли тревог
    assert out["alarm_rate"] == 3 / 4
