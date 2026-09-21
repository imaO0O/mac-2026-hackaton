"""Расхождение двух анализаторов как признак недостоверности. Только CPU.

    python scripts/check_analyzer_disagreement.py

Зачем. До таблицы тегов 15.09 поточный анализатор был один, и недостоверность мы
ловили двумя способами: залипание («полка») и заглушка 307. Теперь приборов два —
ряд из файла и тег `Q21` телеметрии, — и появляется третий признак, которого раньше
быть не могло: **приборы живы, но показывают разное**. В окне плохих данных так и
вышло: файл показывал 18.45, `Q21` — 24.9, лаборатория — 3.4.

Проверяем, стоит ли считать расхождение причиной недоверия. Мера — ошибка того
значения, которым система пользуется (`Q21`), относительно ближайшего лабораторного
анализа, в зависимости от расхождения приборов на тот же момент.

**Правило записано ДО счёта.** Признак включается, если на ВАЛИДАЦИИ одновременно:

1. при расхождении выше порога ошибка оперативного значения хотя бы вдвое больше,
   чем при расхождении ниже порога;
2. таких моментов не больше 20 % — иначе система будет молчать слишком часто.

Порог выбирается из сетки как наименьший, при котором условие 1 выполняется. Не
выполняется ни при одном — измеренный отказ, признак не вводим.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "analyzer_disagreement.json"
GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
ERROR_RATIO = 2.0
MAX_SHARE = 0.20


def asof(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    series = series.dropna().sort_index()
    pos = series.index.searchsorted(index, side="right") - 1
    return np.where(pos >= 0, series.to_numpy()[pos.clip(min=0)], np.nan)


def block(split: str, cfg: dict, sb: StateBuilder) -> dict:
    lo, hi = cfg["split"][split]
    lab = sb.lims_sulfur.loc[str(lo):str(hi)].dropna()
    if lab.empty:
        return {}
    index = pd.DatetimeIndex(lab.index)
    q21, pak = asof(sb.q21_sulfur, index), asof(sb.pak_sulfur, index)
    good = ~np.isnan(q21) & ~np.isnan(pak)
    q21, pak, truth = q21[good], pak[good], lab.to_numpy()[good]
    gap = np.abs(q21 - pak)
    error = np.abs(q21 - truth)

    rows = []
    for threshold in GRID:
        high = gap > threshold
        if high.sum() < 5 or (~high).sum() < 5:
            continue
        rows.append({
            "порог расхождения, мг/кг": threshold,
            "доля моментов": round(float(high.mean()), 3),
            "ошибка при расхождении выше порога": round(float(error[high].mean()), 2),
            "ошибка ниже порога": round(float(error[~high].mean()), 2),
            "во сколько раз хуже": round(float(error[high].mean() / max(error[~high].mean(), 1e-9)), 2),
        })
    return {"проб": int(len(truth)),
            "медиана расхождения": round(float(np.median(gap)), 2),
            "пороги": rows}


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)
    out = {split: block(split, cfg, sb) for split in ("val", "test")}
    for split, data in out.items():
        print(f"\n{split}: проб {data.get('проб')}, медиана расхождения "
              f"{data.get('медиана расхождения')}")
        for row in data.get("пороги", []):
            print(f"  {row}")

    rows = out.get("val", {}).get("пороги", [])
    passing = [r for r in rows
               if r["во сколько раз хуже"] >= ERROR_RATIO
               and r["доля моментов"] <= MAX_SHARE]
    choice = min(passing, key=lambda r: r["порог расхождения, мг/кг"]) if passing else None
    rule = {
        "1. ошибка при расхождении хотя бы вдвое больше": bool(passing),
        "2. таких моментов не больше 20 %": bool(passing),
    }
    print("\nПравило приёмки (валидация):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: признак расхождения "
          f"{'ПРИНЯТ, порог ' + str(choice['порог расхождения, мг/кг']) if choice else 'НЕ принят'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "условие: во сколько раз хуже": ERROR_RATIO,
                                  "условие: доля моментов не больше": MAX_SHARE,
                                  "выборки": out, "правило": rule,
                                  "принят": bool(choice),
                                  "порог": choice["порог расхождения, мг/кг"] if choice else None},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
