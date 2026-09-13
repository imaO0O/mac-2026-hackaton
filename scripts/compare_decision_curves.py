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

**Неопределённость.** Превышений на тесте 65, и у доли пропусков широкий разброс
сам по себе. Интервалы строятся блочным бутстрэпом по неделям, а не формулой для
независимых наблюдений: моменты идут через 12 часов, факт берётся за 24 часа, и
соседние моменты делят одни и те же превышения. Модели сравниваются ПАРНО —
одни и те же недели в каждой бутстрэп-выборке, — потому что проверяются на одних
и тех же превышениях. Порог каждой модели при этом выбран по тем же данным, так
что интервалы скорее оптимистичны, чем пессимистичны.

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

# строка таблицы -> отчёт прогона; первая строка — точка отсчёта для парных сравнений
RUNS = {
    "бустинг h0": "test_period.json",
    "бустинг h0, сид 100": "test_period_boost_s100.json",
    "бустинг h0, сид 200": "test_period_boost_s200.json",
    "сеть h2, сиды 42–44": "test_period_seq_h2.json",
    "сеть h2, сиды 100–102": "test_period_seq_s100_h2.json",
    "сеть h2, сиды 200–202": "test_period_seq_s200_h2.json",
}
REFERENCE = "бустинг h0"
# доли ложных тревог, при которых сравниваем
LEVELS = (0.25, 0.30, 0.40, 0.50)
GRID = np.round(np.arange(0.0, 1.0001, 0.005), 3)
CHECKED = ("порог", "поймано", "пропущено", "доля пропусков", "ложных тревог",
           "доля ложных тревог", "рабочий")
BOOTSTRAP = 2000
REPORT = ROOT / "reports" / "decision_curves.json"


def decision_flags(frame: pd.DataFrame, act: float, hours: float,
                   threshold: float) -> pd.DataFrame:
    """Решение по каждому моменту при заданном пороге — те же правила, что в переборе.

    Сумма этих флагов обязана совпасть с ``threshold_sweep`` на каждом пороге —
    это проверяется в ``main``: две реализации одного правила не имеют права
    разойтись молча.
    """
    column = f"превышение за {hours:.0f} ч"
    known = frame[frame[column].notna() & frame["риск"].notna()]
    refused = known["исход"].eq("отказ")
    forced = known["исход"].eq("меняем уставки") & known["риск"].lt(act)
    postponed = known["исход"].eq("держим режим") & known["риск"].ge(act)
    over = known[column].astype(bool)
    acts = ~refused & ~postponed & (known["риск"].ge(threshold) | forced)
    return pd.DataFrame({"ts": pd.to_datetime(known["ts"]), "over": over,
                         "missed": over & ~acts & ~refused, "fa": ~over & acts})


def best_threshold(curve: list[dict], level: float) -> dict | None:
    """Порог с наименьшей долей пропусков при доле ложных тревог не выше ``level``."""
    ok = [row for row in curve if row["доля ложных тревог"] <= level + 1e-9]
    return min(ok, key=lambda r: (r["доля пропусков"], r["доля ложных тревог"])) if ok else None


def weekly_counts(flags: pd.DataFrame, weeks: pd.Index) -> np.ndarray:
    """Счётчики по неделям: [превышений, пропусков] — в порядке ``weeks``."""
    g = flags.assign(week=flags["ts"].dt.to_period("W").astype(str)).groupby("week")
    counts = pd.DataFrame({"over": g["over"].sum(), "missed": g["missed"].sum()})
    return counts.reindex(weeks, fill_value=0).to_numpy(dtype=float)


def main() -> int:
    use_utf8_console()
    table: dict[str, dict] = {}
    flags_at: dict[str, dict[str, pd.DataFrame]] = {}
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
        for row in curve[::10]:
            f = decision_flags(frame, act, MAIN_WINDOW, row["порог"])
            if (int(f["missed"].sum()) != row["пропущено"]
                    or int(f["fa"].sum()) != row["ложных тревог"]):
                print(f"ОШИБКА: флаги по моментам разошлись с перебором ({name}, порог "
                      f"{row['порог']}). Две реализации одного правила — так нельзя.")
                return 1

        working = next(row for row in curve if row["рабочий"])
        chosen = {f"{int(level * 100)}%": best_threshold(curve, level) for level in LEVELS}
        flags_at[label] = {key: decision_flags(frame, act, MAIN_WINDOW, row["порог"])
                           for key, row in chosen.items() if row is not None}
        table[label] = {
            "отчёт": name,
            "артефакт модели": summary.get("артефакт модели"),
            "feature_version": report.get("feature_version"),
            "рабочая точка": {k: working[k] for k in
                              ("порог", "доля пропусков", "доля пропусков с отказами",
                               "доля ложных тревог")},
            "пропуски при ложных не выше": {
                key: (None if row is None else row["доля пропусков"])
                for key, row in chosen.items()},
            "пропуски с отказами при ложных не выше": {
                key: (None if row is None else row["доля пропусков с отказами"])
                for key, row in chosen.items()},
            "порог при ложных не выше": {
                key: (None if row is None else row["порог"]) for key, row in chosen.items()},
        }

    if not table:
        print("Нет ни одного прогона.")
        return 1

    # ---- блочный бутстрэп по неделям, парный ----
    all_weeks = sorted({w for per in flags_at.values() for f in per.values()
                        for w in f["ts"].dt.to_period("W").astype(str)})
    weeks = pd.Index(all_weeks)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(weeks), size=(BOOTSTRAP, len(weeks)))
    for level in LEVELS:
        key = f"{int(level * 100)}%"
        rates = {}
        for label, per in flags_at.items():
            if key not in per:
                continue
            c = weekly_counts(per[key], weeks)
            over = c[draws, 0].sum(axis=1)
            missed = c[draws, 1].sum(axis=1)
            rates[label] = np.where(over > 0, missed / np.maximum(over, 1), np.nan)
            lo, hi = np.nanpercentile(rates[label], [5, 95])
            table[label].setdefault("90% интервал пропусков", {})[key] = [round(lo, 3),
                                                                          round(hi, 3)]
        if REFERENCE in rates:
            for label, r in rates.items():
                if label == REFERENCE:
                    continue
                diff = r - rates[REFERENCE]
                lo, hi = np.nanpercentile(diff, [5, 95])
                verdict = ("лучше" if hi < 0 else "хуже" if lo > 0 else "неразличимо")
                table[label].setdefault(f"против «{REFERENCE}»", {})[key] = {
                    "разница": round(float(np.nanmedian(diff)), 3),
                    "90% интервал": [round(lo, 3), round(hi, 3)], "вывод": verdict}

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

    print(f"\nПарно против «{REFERENCE}», блочный бутстрэп по неделям "
          f"({len(weeks)} недель, 90 % интервал разницы долей пропусков):")
    for label, row in table.items():
        cmp = row.get(f"против «{REFERENCE}»")
        if not cmp:
            continue
        parts = [f"≤{k}: {v['разница']:+.0%} [{v['90% интервал'][0]:+.0%}…"
                 f"{v['90% интервал'][1]:+.0%}] {v['вывод']}" for k, v in cmp.items()]
        print(f"  {label:24s} " + "; ".join(parts))

    groups = {"сеть h2": [r for l, r in table.items() if l.startswith("сеть h2")],
              "бустинг h0": [r for l, r in table.items() if l.startswith("бустинг h0")]}
    print("\nРазброс по сидам при одинаковой строгости:")
    for name, rows in groups.items():
        if len(rows) < 2:
            continue
        spreads = []
        for level in LEVELS:
            key = f"{int(level * 100)}%"
            vals = [r["пропуски при ложных не выше"][key] for r in rows]
            if all(v is not None for v in vals):
                spreads.append(f"≤{key}: {min(vals):.0%}…{max(vals):.0%}")
        print(f"  {name:12s} " + "; ".join(spreads))

    REPORT.write_text(json.dumps({**report_provenance(), "окно, ч": MAIN_WINDOW,
                                  "бутстрэп": {"блок": "неделя", "выборок": BOOTSTRAP,
                                               "недель": len(weeks)},
                                  "модели": table}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
