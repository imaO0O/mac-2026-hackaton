# -*- coding: utf-8 -*-
"""Порог непригодности среза: число назначено, но ответ от него не зависит.

Третий порог в ряду разобранных, и единственный, который проверку пережил без
правки. Его ценность не в значении 0.2, а в том, что распределение доли
пропусков двугорбое: срез либо почти полный, либо разорван больше чем наполовину.
Порог стоит в провале между горбами, поэтому любое значение из провала даёт
почти тот же ответ.

Это стоит держать проверяемым, а не в тексте: если данные изменятся и провал
заполнится, назначенное число снова станет важным — и тогда его придётся
обосновывать, а не наследовать.
"""
from __future__ import annotations

import pytest

from nefte.config import load_config
from nefte.pipeline import MAX_MISSING_SHARE


@pytest.fixture(scope="module")
def missing_shares():
    pd = pytest.importorskip("pandas")
    from nefte.pipeline import StateBuilder

    cfg = load_config()
    builder = StateBuilder(cfg)
    out = []
    for name in ("train", "val", "test"):
        lo, hi = cfg["split"][name]
        for ts in pd.date_range(lo, hi, freq="12h"):
            out.append(builder.build(ts).data_quality.missing_share)
    return out


def test_the_gap_the_threshold_stands_in_is_really_empty(missing_shares):
    """В провале [0.08, 0.40) лежит меньше 5 % срезов."""
    in_gap = sum(1 for v in missing_shares if 0.08 <= v < 0.40)
    share = in_gap / len(missing_shares)
    assert share < 0.05, (
        f"в провале оказалось {share:.1%} срезов — распределение перестало быть "
        "двугорбым, и значение MAX_MISSING_SHARE снова требует обоснования")


def test_the_answer_barely_moves_with_the_threshold(missing_shares):
    """Порог можно менять втрое, и доля непригодных сдвинется меньше чем на 1 п.п."""
    def share_at(limit: float) -> float:
        return sum(1 for v in missing_shares if v >= limit) / len(missing_shares)

    low, high = share_at(0.10), share_at(0.30)
    assert abs(low - high) < 0.01, (
        f"порог 0.10 даёт {low:.2%} непригодных, порог 0.30 — {high:.2%}: "
        "разница выросла, число перестало быть безразличным")
    assert 0.10 < MAX_MISSING_SHARE < 0.30, (
        "рабочий порог обязан стоять внутри проверенного провала")


def test_the_lims_outlier_cut_also_stands_in_a_gap():
    """Обрезка лабораторных выбросов — тот же случай: число назначено, промежуток пуст.

    Значения серы идут плотно до 46 мг/кг, дальше скачок на 107, 120, 2120.
    Порог 50 стоит в пустоте между ними, поэтому любое значение из [47, 106]
    отбрасывает ровно те же три точки. Проверяется именно пустота промежутка, а
    не само число: если в него попадут новые анализы, выбор снова станет важным.
    """
    pytest.importorskip("pandas")
    from nefte.data.loaders import lims_series, load_lims

    cfg = load_config()
    raw = lims_series(cfg["quality"]["target"]["lims_source"], load_lims()).dropna()
    raw = raw[raw > 0]
    cut = float(cfg["quality"]["lims_sulfur_outlier_above"])

    in_gap = int(((raw > 46.0) & (raw < 107.0)).sum())
    assert in_gap == 0, (
        f"в промежутке (46, 107) появилось {in_gap} анализов — порог обрезки "
        "перестал быть безразличным и требует обоснования")
    assert 46.0 < cut < 107.0, "рабочий порог обязан стоять внутри пустого промежутка"

    dropped = int((raw > cut).sum())
    assert dropped == int((raw > 80.0).sum()) == 3, (
        "число отброшенных точек изменилось: данные поехали, разбор устарел")
