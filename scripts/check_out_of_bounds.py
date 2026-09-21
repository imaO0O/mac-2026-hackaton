"""Что делает система, когда уставка УЖЕ за границей допустимого. Только CPU.

    python scripts/check_out_of_bounds.py

Зачем. Эксперт 11.09 (25:14–26:54, восстановлено повторной расшифровкой): паспортные
ограничения оборудования конфиденциальны, участникам «придётся самостоятельно думать
над этим», и это даёт возможность «проигрывать сценарий нарушения этих ограничений»:
«расставьте границы так, чтобы иногда происходили нарушения, и дайте рекомендации,
систему рекомендаций по возврату технологического процесса в нормальный режим».

У нас границы — квантили обучающего периода p05–p95 (ДОПУЩЕНИЕ, `limits.mode`), то
есть режим выходит за них примерно в 10 % моментов ПО ПОСТРОЕНИЮ. Значит, сценарий
нарушения у нас уже есть в данных, и вопрос в том, что система в такие моменты
делает.

Считаем на тесте, сколько моментов уставка вне диапазона, и — главное — в скольких
из них тег ВЫПАДАЕТ из рассмотрения оптимизатора. Выпадает он тогда, когда
пересечение диапазона с шагом пусто: `max(lo, cur - step) > min(hi, cur + step)`.
Если это случается, система в нарушении не возвращает режим в норму, а молча
перестаёт трогать нарушенный тег — ровно противоположное тому, о чём просил эксперт.
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

REPORT = ROOT / "reports" / "out_of_bounds.json"
CONTROL_TAGS = ["T5", "T11", "F26", "P13"]


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)
    bounds = sb.model_bounds(CONTROL_TAGS, unit="ht")
    steps = cfg["limits"]["max_step_per_cycle"]

    out: dict = {"границы (p05–p95 обучения)": {t: [round(v, 3) for v in b]
                                                for t, b in bounds.items()}}
    # Остановы считать нельзя: на холодном реакторе уставка уходит на сотни
    # градусов ниже границы, но система в такие моменты и так отказывает
    # («установка не в работе»), и выпавший тег там ничего не меняет.
    feed_median = float(sb.ht["F26"].median())
    running = ((sb.ht["F26"] >= feed_median * 0.1)
               & (sb.ht[["T5", "T6", "T11"]].mean(axis=1) >= 150.0))
    print(f"работающих моментов: {running.mean():.1%} истории")

    for split in ("val", "test"):
        lo_ts, hi_ts = cfg["split"][split]
        frame = sb.ht.loc[str(lo_ts):str(hi_ts)]
        frame = frame[running.reindex(frame.index).fillna(False).to_numpy()]
        rows = []
        for tag, (lo, hi) in bounds.items():
            if tag not in frame:
                continue
            cur = frame[tag].astype(float).dropna()
            if cur.empty:
                continue
            step = (steps["temperature_c"] if tag.startswith("T") else
                    steps["pressure_mpa"] if tag.startswith("P") else
                    cur.abs() * steps["flow_rel"])
            lo_e = np.maximum(lo, cur - step)
            hi_e = np.minimum(hi, cur + step)
            dropped = lo_e > hi_e
            outside = (cur < lo) | (cur > hi)
            rows.append({
                "тег": tag,
                "моментов": int(len(cur)),
                "вне диапазона": round(float(outside.mean()), 3),
                "тег выпал из рассмотрения": round(float(dropped.mean()), 3),
                "макс. выход за границу": round(float(np.maximum(
                    (cur - hi).max(), (lo - cur).max())), 2),
                "шаг за цикл": (round(float(step), 3) if np.isscalar(step)
                                else round(float(np.median(step)), 3)),
            })
        out[split] = rows
        print(f"\n{split}:")
        for row in rows:
            print(f"  {row}")

    worst = max((row["тег выпал из рассмотрения"] for row in out.get("test", [])),
                default=0.0)
    print(f"\nхудший тег: выпадает в {worst:.1%} моментов теста")
    print("  ВЫВОД: " + ("возврат в норму НЕ обеспечен — при выходе за границу "
                         "больше чем на шаг тег перестаёт рассматриваться"
                         if worst > 0 else
                         "выхода больше чем на шаг нет: возврат обеспечен шагом"))

    REPORT.write_text(json.dumps({**report_provenance(cfg), **out,
                                  "худшая доля выпадения на тесте": worst},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
