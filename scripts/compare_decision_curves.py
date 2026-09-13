"""Сравнение моделей на уровне РЕШЕНИЙ: пропуски при заданной доле ложных тревог.

    python scripts/compare_decision_curves.py

Читает сохранённые прогоны по тестовому периоду (``reports/test_period*.json``) и
для каждого перебирает порог вмешательства на мелкой сетке тем же правилом, что и
сам прогон (``run_test_period.threshold_sweep``: вынужденные и отложенные
воздействия учтены). Цикл заново не считается — риск по каждому моменту уже
сохранён.

Зачем отдельный скрипт. Модели сравниваются не по рабочим точкам — у каждой она
своя, и у сети это 77–83 % ложных тревог, — а при ОДИНАКОВОЙ строгости. Эта
таблица однажды жила в черновике, а число из неё («49 % против 71 %») стояло в
README и оказалось посчитанным на модели, затёртой проверкой сидов.

Перед счётом скрипт проверяет, что перебор воспроизводит пороги, сохранённые в
самом отчёте. Не воспроизводит — отчёт снят другим правилом, и сравнивать его
нельзя: выход с ненулевым кодом.

Результат: таблица в консоли и reports/decision_curves.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.config import ROOT  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_test_period import MAIN_WINDOW, threshold_sweep  # noqa: E402

# строка таблицы -> отчёт прогона
RUNS = {
    "бустинг h0": "test_period.json",
    "сеть h2, сиды 42–44": "test_period_seq_h2.json",
    "сеть h2, сиды 100–102": "test_period_seq_s100_h2.json",
    "сеть h2, сиды 200–202": "test_period_seq_s200_h2.json",
}
# доли ложных тревог, при которых сравниваем
LEVELS = (0.25, 0.30, 0.40, 0.50)
GRID = np.round(np.arange(0.0, 1.0001, 0.005), 3)
CHECKED = ("порог", "поймано", "пропущено", "доля пропусков", "ложных тревог",
           "доля ложных тревог", "рабочий")
REPORT = ROOT / "reports" / "decision_curves.json"


def best_miss(curve: list[dict], level: float, key: str) -> float | None:
    """Наименьшая доля пропусков при доле ложных тревог не выше ``level``."""
    ok = [row[key] for row in curve if row["доля ложных тревог"] <= level + 1e-9]
    return min(ok) if ok else None


def main() -> int:
    use_utf8_console()
    table: dict[str, dict] = {}
    for label, name in RUNS.items():
        path = ROOT / "reports" / name
        if not path.exists():
            print(f"[пропуск] {label}: нет {name}")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report["summary"]
        act = summary["порог вмешательства"]["рабочий"]
        frame = pd.DataFrame(report["rows"])

        stored = summary["порог вмешательства"]["перебор"]
        # рабочий порог в отчёте округлён до трёх знаков, а функция добавляет его
        # точным сама — передаём только остальные, иначе строк станет на одну больше
        replay = threshold_sweep(frame, act, MAIN_WINDOW,
                                 thresholds=[row["порог"] for row in stored
                                             if not row["рабочий"]])
        if ([{k: r[k] for k in CHECKED} for r in replay]
                != [{k: r[k] for k in CHECKED} for r in stored]):
            print(f"ОШИБКА: {name} не воспроизводится текущим правилом перебора. "
                  "Отчёт снят другим кодом — перезапустите run_test_period.py.")
            return 1

        curve = threshold_sweep(frame, act, MAIN_WINDOW, thresholds=GRID)
        working = next(row for row in curve if row["рабочий"])
        table[label] = {
            "отчёт": name,
            "артефакт модели": summary.get("артефакт модели"),
            "feature_version": report.get("feature_version"),
            "рабочая точка": {k: working[k] for k in
                              ("порог", "доля пропусков", "доля пропусков с отказами",
                               "доля ложных тревог")},
            "пропуски при ложных не выше": {
                f"{int(level * 100)}%": best_miss(curve, level, "доля пропусков")
                for level in LEVELS},
            "пропуски с отказами при ложных не выше": {
                f"{int(level * 100)}%": best_miss(curve, level, "доля пропусков с отказами")
                for level in LEVELS},
        }

    if not table:
        print("Нет ни одного прогона.")
        return 1

    header = " ".join(f"≤{int(level * 100)}%".rjust(6) for level in LEVELS)
    print(f"\nНаименьшая доля пропусков при доле ложных тревог не выше (окно "
          f"{MAIN_WINDOW:.0f} ч)\n")
    print(f"{'':24s} {header}   рабочая точка")
    for label, row in table.items():
        cells = " ".join(("—" if v is None else f"{v:.0%}").rjust(6)
                         for v in row["пропуски при ложных не выше"].values())
        w = row["рабочая точка"]
        print(f"{label:24s} {cells}   пропуски {w['доля пропусков']:.0%} при "
              f"ложных {w['доля ложных тревог']:.0%}")

    seeds = [row for label, row in table.items() if label.startswith("сеть h2")]
    boost = table.get("бустинг h0")
    if boost and len(seeds) >= 2:
        print("\nВывод по уровням строгости:")
        for level in LEVELS:
            key = f"{int(level * 100)}%"
            b = boost["пропуски при ложных не выше"][key]
            vals = [r["пропуски при ложных не выше"][key] for r in seeds]
            if b is None or any(v is None for v in vals):
                continue
            if all(v < b for v in vals):
                verdict = "сеть лучше при ЛЮБОМ сиде"
            elif all(v > b for v in vals):
                verdict = "сеть хуже при любом сиде"
            else:
                verdict = "знак зависит от сида — выигрыша нет"
            print(f"  ложных ≤{key:4s}: бустинг {b:.0%}, сеть {min(vals):.0%}…"
                  f"{max(vals):.0%} (разброс {max(vals) - min(vals):.0%}) — {verdict}")

    REPORT.write_text(json.dumps({**report_provenance(), "окно, ч": MAIN_WINDOW,
                                  "модели": table}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
