"""Что признак расхождения приборов сделает с РЕШЕНИЯМИ. Только CPU.

    python scripts/check_disagreement_decisions.py

Зачем. `scripts/check_analyzer_disagreement.py` показал, что при расхождении
приборов выше 2 мг/кг оперативное значение врёт втрое сильнее лаборатории. Это
довод не доверять данным, но ещё не довод отказываться решать: отказ — это молчание
системы, и молчать там, где вмешательство было оправданным, дороже. Меряем цену
признака на уже посчитанном прогоне валидации, ДО того как трогать код решения.

**Правило записано ДО счёта.** Признак уходит в путь отказа, если на ВАЛИДАЦИИ
одновременно:

1. доля превышений, оставшихся без вмешательства («пропуски»), растёт не больше
   чем на 2 процентных пункта;
2. среди вмешательств, которые признак погасит, доля действий без подтверждённого
   превышения за 24 ч выше, чем среди остальных вмешательств — то есть признак
   в первую очередь убирает необоснованные действия, а не полезные.

Не выполняется — измеренный отказ, признак остаётся пометкой достоверности в
карточке и в код решения не идёт.

**Проверка на устойчивость добавлена ПОСЛЕ счёта** и это честно сказано в отчёте:
то же правило считается на соседних порогах. Если условие выполняется только на
одном пороге сетки, вывод ненадёжен — так выглядит шум, а не эффект, — и мы берём
осторожную ветку (карточка), даже когда формальное правило выполнено.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "disagreement_decisions.json"
SNAPSHOT = ROOT / "reports" / "val_period_step1h_lock4.json"
THRESHOLD = 2.0
GRID = [1.5, 2.0, 3.0, 4.0]
MAX_MISS_GROWTH = 0.02
HOURS = 24.0


def asof(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    series = series.dropna().sort_index()
    pos = series.index.searchsorted(index, side="right") - 1
    return np.where(pos >= 0, series.to_numpy()[pos.clip(min=0)], np.nan)


def block(path: Path, sb: StateBuilder, threshold: float) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    index = pd.DatetimeIndex(rows["ts"])
    gap = np.abs(asof(sb.q21_sulfur, index) - asof(sb.pak_sulfur, index))
    flag = gap > threshold

    column = f"превышение за {HOURS:.0f} ч"
    acts = rows["исход"].eq("меняем уставки").to_numpy()
    known = rows[column].notna().to_numpy()
    over = np.where(known, rows[column].to_numpy() == True, False)  # noqa: E712

    def miss_share(silent: np.ndarray) -> float:
        real = over & known
        return float((real & silent).sum() / max(int(real.sum()), 1))

    killed, kept = acts & flag, acts & ~flag

    def groundless(mask: np.ndarray) -> float | None:
        sel = mask & known
        return round(float((~over[sel]).mean()), 3) if sel.any() else None

    return {
        "прогон": path.name,
        "порог расхождения, мг/кг": threshold,
        "моментов": int(len(rows)),
        "доля моментов с расхождением выше порога": round(float(np.nanmean(flag)), 3),
        "вмешательств сейчас": int(acts.sum()),
        "из них погаснет": int(killed.sum()),
        "отказов сейчас": int(rows["исход"].eq("отказ").sum()),
        "отказов станет": int((rows["исход"].eq("отказ").to_numpy() | flag).sum()),
        "пропусков сейчас": round(miss_share(~acts), 3),
        "пропусков станет": round(miss_share(~acts | flag), 3),
        "доля действий без превышения: погашенные": groundless(killed),
        "доля действий без превышения: оставшиеся": groundless(kept),
    }


def verdict(data: dict) -> tuple[dict, bool]:
    growth = data["пропусков станет"] - data["пропусков сейчас"]
    killed_bad = data["доля действий без превышения: погашенные"]
    kept_bad = data["доля действий без превышения: оставшиеся"]
    rule = {
        "1. пропуски растут не больше чем на 2 п.п.": bool(growth <= MAX_MISS_GROWTH + 1e-9),
        "2. гасит прежде всего необоснованные действия": bool(
            killed_bad is not None and kept_bad is not None and killed_bad > kept_bad),
    }
    return rule, all(rule.values())


def main() -> int:
    use_utf8_console()
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = load_config()
    sb = StateBuilder(cfg)

    out = {"валидация": block(SNAPSHOT, sb, THRESHOLD)}
    test = ROOT / "reports" / "test_period_step1h.json"
    if test.exists():
        out["тест"] = block(test, sb, THRESHOLD)
    for name, data in out.items():
        print(f"\n{name}:")
        for key, value in data.items():
            print(f"  {key}: {value}")

    rule, accepted = verdict(out["валидация"])
    print(f"\nПравило приёмки (валидация, порог {THRESHOLD}):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")

    stability = []
    for threshold in GRID:
        data = block(SNAPSHOT, sb, threshold)
        checks, ok = verdict(data)
        stability.append({**{k: data[k] for k in (
            "порог расхождения, мг/кг", "доля моментов с расхождением выше порога",
            "из них погаснет", "пропусков станет",
            "доля действий без превышения: погашенные",
            "доля действий без превышения: оставшиеся")},
            "правило выполнено": ok})
        print(f"  порог {threshold}: правило {'выполнено' if ok else 'НЕ выполнено'} "
              f"({checks})")
    passing = sum(1 for row in stability if row["правило выполнено"])
    stable = passing >= 2

    print(f"\nустойчивость: правило выполняется на {passing} порогах сетки из {len(GRID)}")
    print("  ВЫВОД: признак расхождения "
          + ("идёт в путь отказа" if accepted and stable else
             "остаётся пометкой достоверности в карточке, решения не меняет"))

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "порог расхождения, мг/кг": THRESHOLD,
                                  "условие: рост пропусков не больше": MAX_MISS_GROWTH,
                                  "выборки": out,
                                  "правило": rule,
                                  "правило выполнено": accepted,
                                  "устойчивость по порогу": stability,
                                  "порогов с выполненным правилом": passing,
                                  "устойчиво": stable,
                                  "решение": ("путь отказа" if accepted and stable
                                              else "только карточка")},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
