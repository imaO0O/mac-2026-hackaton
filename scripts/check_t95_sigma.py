# -*- coding: utf-8 -*-
"""Неопределённость нашего знания о текущем Т95 как функция возраста анализа.

Уровень Т95 берётся из последнего лабораторного анализа, поэтому неопределённость
— это разброс того, насколько показатель успел уехать с момента отбора пробы.
Раньше он был ОДНИМ числом (6.64 °C), измеренным на медианном шаге в сутки, и
применялся при любом возрасте опорного анализа. Между тем анализ Т95 старше суток
в трети моментов решения.

Скрипт меряет разброс по парам анализов обучающего периода и отвечает на вопрос,
какой формы кривая: у случайного блуждания σ растёт как корень из времени, у
возвращающегося к среднему ряда выходит на полку. Ответ решает, надо ли вообще
усложнять: если полка низкая, константа почти верна.

Пишет reports/t95_sigma.json. Обучающий период — и только он: неопределённость
подбирается там же, где всё остальное.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nefte.config import load_config                       # noqa: E402
from nefte.data.cleaning import clean_lims_distillation    # noqa: E402
from nefte.data.loaders import lims_series, load_lims      # noqa: E402

# Границы вёдер по возрасту опорного анализа, часы. Выбраны по фактическому
# расписанию: сутки, двое, трое, дальше — шире, потому что пар меньше.
EDGES = [0.0, 30.0, 54.0, 80.0, 130.0, 200.0, 400.0]
MIN_PAIRS = 100


def measure(series: pd.Series) -> list[dict]:
    idx, values = series.index, series.values
    gaps, deltas = [], []
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            dt = (idx[j] - idx[i]).total_seconds() / 3600.0
            if dt > EDGES[-1]:
                break
            gaps.append(dt)
            deltas.append(values[j] - values[i])
    g, d = np.asarray(gaps), np.asarray(deltas)

    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (g >= lo) & (g < hi)
        if sel.sum() < MIN_PAIRS:
            continue
        out.append({
            "age_from_h": lo, "age_to_h": hi, "n_pairs": int(sel.sum()),
            "sigma_c": float(d[sel].std(ddof=1)),
            "mae_c": float(np.abs(d[sel]).mean()),
            "bias_c": float(d[sel].mean()),
        })
    return out


def main() -> None:
    cfg = load_config()
    lo, hi = cfg["split"]["train"]
    t95 = clean_lims_distillation(
        lims_series("Гидроочистка|2|95%.T", load_lims())).dropna().sort_index().loc[lo:hi]

    table = measure(t95)
    first, last = table[0], table[-1]
    ratio = last["sigma_c"] / first["sigma_c"]
    # Что дало бы случайное блуждание на том же плече
    mid_first = (first["age_from_h"] + first["age_to_h"]) / 2
    mid_last = (last["age_from_h"] + last["age_to_h"]) / 2
    walk = first["sigma_c"] * (mid_last / mid_first) ** 0.5

    print("Т95 на обучающем периоде: %d анализов" % len(t95))
    print()
    print("разброс изменения как функция возраста опорного анализа:")
    for row in table:
        print("   %3.0f-%3.0f ч: n=%5d  сигма %.2f  MAE %.2f  смещение %+.2f"
              % (row["age_from_h"], row["age_to_h"], row["n_pairs"],
                 row["sigma_c"], row["mae_c"], row["bias_c"]))
    print()
    print("форма кривой: сигма выросла в %.2f раза (%.2f -> %.2f)"
          % (ratio, first["sigma_c"], last["sigma_c"]))
    print("   случайное блуждание дало бы %.2f, то есть в %.2f раза"
          % (walk, walk / first["sigma_c"]))
    verdict = ("ряд возвращается к среднему: сигма выходит на полку, "
               "и константа занижает её не больше чем в %.2f раза" % ratio)
    if ratio > 0.8 * (walk / first["sigma_c"]):
        verdict = ("ряд ведёт себя как блуждание: сигма растёт без полки, "
                   "константу применять нельзя")
    print("   вывод: " + verdict)

    report = {
        "source": "Гидроочистка|2|95%.T",
        "train": [str(lo), str(hi)],
        "n_analyses": int(len(t95)),
        "buckets": table,
        "sigma_growth_ratio": float(ratio),
        "random_walk_would_give": float(walk),
        "verdict": verdict,
    }
    out = ROOT / "reports" / "t95_sigma.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("записано:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
