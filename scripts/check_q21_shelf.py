"""Полка Q21: детектор залипания с допуском. Только CPU.

    python scripts/check_q21_shelf.py

Зачем. С 20.09 оперативный анализатор серы — `Q21`. Найдено 21.09 на дашборде:
он уходит в состояние неисправности — «полку» около 24.88 мг/кг с шумом ±0.04, — а
детектор залипания (`data.cleaning.frozen_mask`) сравнивает значения на ТОЧНОЕ
равенство и такую полку не видит вовсе: 0 % её точек помечено. Длинные эпизоды —
858 ч в марте–апреле 2024 (замена катализатора), 307 ч в апреле 2026 (снова замена),
330 ч в июне–июле 2026. На тесте на полке 10.0 % показаний. Файловый ПАК на своей
полке стоит ровно на 18.45, поэтому для него точное равенство работало.

Кандидат — тот же признак «уже не меняется N отсчётов подряд», но с допуском:
размах значений за последние N отсчётов меньше порога. Считается только по прошлому,
как и прежний детектор (без заглядывания вперёд).

**Правило записано ДО счёта.** Эталон полки задаётся без детектора: показания Q21 в
24.7–25.1 мг/кг внутри непрерывных отрезков длиннее суток. Допуск выбирается на
ОБУЧАЮЩЕМ периоде (там полка 2024 года, 858 ч) из сетки 0.05 / 0.1 / 0.2 / 0.5 мг/кг
как наименьший, при котором одновременно:

1. детектор ловит не меньше 90 % точек эталонной полки;
2. на работающей установке вне полки помечает не больше 1 % точек — иначе он начнёт
   выключать живой прибор.

Не проходит ни один — измеренный отказ, детектор остаётся прежним. Тест — только
для отчёта.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import flat_mask, frozen_mask  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "q21_shelf.json"
SHELF = (24.7, 25.1)
SHELF_MIN_HOURS = 24.0
GRID = [0.05, 0.1, 0.2, 0.5]
MIN_CATCH = 0.90
MAX_FALSE = 0.01


def reference_shelf(series: pd.Series) -> pd.Series:
    """Эталон без детектора: 24.7–25.1 внутри отрезков длиннее суток."""
    inside = series.between(*SHELF)
    run = (inside != inside.shift()).cumsum()
    start = series.index.to_series().groupby(run).transform("min")
    end = series.index.to_series().groupby(run).transform("max")
    long_enough = (end - start).dt.total_seconds() / 3600.0 >= SHELF_MIN_HOURS
    return inside & long_enough


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)
    q21 = sb.q21_sulfur.dropna().sort_index()
    min_samples = int(cfg["telemetry"]["frozen_min_samples"])
    shelf = reference_shelf(q21)
    # работающая установка — как в детекторе останова агента надёжности
    ht = sb.ht.reindex(q21.index, method="ffill")
    running = ((ht["F26"] >= sb.ht["F26"].median() * 0.1)
               & (ht[["T5", "T6", "T11"]].mean(axis=1) >= 150.0))
    # прежний детектор — точное равенство; sb.q21_frozen уже считается с допуском
    old = frozen_mask(q21, min_samples).astype(bool)

    out: dict = {"эталон полки, мг/кг": list(SHELF), "полка не короче, ч": SHELF_MIN_HOURS,
                 "отсчётов подряд": min_samples}
    for split in ("train", "val", "test"):
        lo, hi = cfg["split"][split]
        sel = (q21.index >= pd.Timestamp(lo)) & (q21.index <= pd.Timestamp(hi) + pd.Timedelta(days=1))
        on_shelf = shelf[sel].to_numpy()
        normal = (~shelf[sel] & running[sel]).to_numpy()
        rows = {"точек": int(sel.sum()), "на полке": round(float(on_shelf.mean()), 4),
                "прежний детектор: ловит полку": (round(float(old[sel].to_numpy()[on_shelf].mean()), 3)
                                                   if on_shelf.any() else None)}
        for tol in GRID:
            flag = flat_mask(q21, min_samples, tol)[sel].to_numpy()
            rows[f"допуск {tol:g}"] = {
                "ловит полку": (round(float(flag[on_shelf].mean()), 3)
                                if on_shelf.any() else None),
                "ложно на работающей": (round(float(flag[normal].mean()), 4)
                                        if normal.any() else None),
            }
        out[split] = rows
        print(f"\n{split}: точек {rows['точек']}, на полке {rows['на полке']:.1%}, "
              f"прежний детектор ловит {rows['прежний детектор: ловит полку']}")
        for tol in GRID:
            print(f"  допуск {tol:g}: {rows[f'допуск {tol:g}']}")

    train = out["train"]
    passing = [tol for tol in GRID
               if (train[f"допуск {tol:g}"]["ловит полку"] or 0) >= MIN_CATCH
               and (train[f"допуск {tol:g}"]["ложно на работающей"] or 1) <= MAX_FALSE]
    choice = min(passing) if passing else None
    print("\nПравило приёмки (обучение):")
    print(f"  прошли: {passing or 'ни один'}")
    print(f"  ВЫВОД: {'допуск ' + format(choice, 'g') + ' мг/кг' if choice else 'детектор не меняется'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "условие: ловит полку не меньше": MIN_CATCH,
                                  "условие: ложно не больше": MAX_FALSE,
                                  **out, "прошли": passing, "выбор, мг/кг": choice},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
