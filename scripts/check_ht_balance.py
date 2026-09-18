"""Расходы 24-2000 против новой таблицы тегов: что с чем сходится (участник 2). Только CPU.

    python scripts/check_ht_balance.py

Зачем. Новая таблица тегов (15.09) называет F9 «сырьё, массовый», F15 «сырьё,
объёмный», F17 «ГО ДТ, массовый», F26 «ГО ДТ, объёмный». Данные с этим не сходятся:
F17 (274 т/ч) больше F9 (226 т/ч), то есть продукта по массе больше, чем сырья.
Прежде чем спрашивать организаторов, надо понять, какое прочтение данные вообще
допускают. Проверяется:

* **какие расходы — один поток**: отношение двух тегов постоянно, если это объём и
  масса одного потока (отношение — плотность) или один прибор, пересчитанный в АСУ;
* **вычислен тег или измерен**: у вычисленного отношение постоянно до четвёртого
  знака во все годы, у двух приборов — гуляет на проценты и дрейфует с сырьём;
* **сходится ли баланс**: сырьё = продукт + бензин + газ, разница двух объёмных
  расходов на входе и выходе должна быть порядка отбора бензина.

Результат: reports/ht_balance.json и сводка в консоли.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.data.loaders import load_telemetry  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

# Подписи: из новой таблицы — как их привёл участник 1 (15.09); из листа «КИП»
# исходного пакета — строки там перемешаны, поэтому это только для сравнения.
LABELS = {
    "F9": {"новая": "сырьё, массовый", "КИП": "расход газа поддува в К-201, массовый"},
    "F15": {"новая": "сырьё, объёмный", "КИП": "расход квенча в Р-202"},
    "F17": {"новая": "ГО ДТ, массовый", "КИП": "расход гидроочищенного ДТ в цех №8"},
    "F26": {"новая": "ГО ДТ, объёмный", "КИП": "расход сырья на установку, объёмный"},
    "W10": {"новая": "бензин, массовый (участник 1)", "КИП": "перепад давления Р-202"},
}
PAIRS = (("F9", "F26"), ("F19", "F26"), ("F17", "F26"), ("F17", "F19"), ("F9", "F17"),
         ("W10", "F1"), ("W4", "W10"))
REPORT = ROOT / "reports" / "ht_balance.json"


def ratio_block(run: pd.DataFrame, a: str, b: str) -> dict:
    pair = run[[a, b]].dropna()
    pair = pair[(pair[a] > 0) & (pair[b] > 0)]
    r = pair[a] / pair[b]
    by_year = r.groupby(r.index.year).median()
    return {"медиана": round(float(r.median()), 4),
            "ширина_p01_p99_доля": round(float((r.quantile(.99) - r.quantile(.01))
                                              / r.median()), 4),
            "по_годам": {str(y): round(float(v), 4) for y, v in by_year.items()},
            "corr": round(float(pair[a].corr(pair[b])), 3)}


def main() -> int:
    use_utf8_console()
    ht = load_telemetry("ht").resample("1h").mean()
    run = ht[ht["F26"] > 0.5 * ht["F26"].median()]
    flows = [c for c in ht.columns if c[0] in "FW"]

    levels = {t: [round(float(x), 2) for x in run[t].quantile([.05, .5, .95])] for t in flows}
    ratios = {f"{a}/{b}": ratio_block(run, a, b) for a, b in PAIRS}
    gap = run["F17"] - run["F26"]
    balance = {"F17−F26_медиана": round(float(gap.median()), 1),
               "F17−F26_p10_p90": [round(float(gap.quantile(.1)), 1),
                                   round(float(gap.quantile(.9)), 1)],
               "доля_от_F26": round(float((gap / run["F26"]).median()), 4),
               "F1_медиана": round(float(run["F1"].median()), 2),
               "corr_с_F1": round(float(gap.corr(run["F1"])), 3)}
    loose = {t: {o: round(float(run[t].corr(run[o])), 3) for o in ("F26", "F25", "F2", "F22")}
             for t in ("F15", "F25", "F22", "F2")}

    print("Уровни расходов в работе (p05 / p50 / p95):")
    for t, v in levels.items():
        label = LABELS.get(t, {})
        print(f"  {t:4s} {v}  {('новая: ' + label['новая']) if label else ''}")
    print("\nОтношения (медиана, ширина p01–p99 в долях, по годам, corr):")
    for k, v in ratios.items():
        print(f"  {k:8s} {v['медиана']:.4f}  {v['ширина_p01_p99_доля']:.4f}  "
              f"{v['по_годам']}  {v['corr']}")
    print(f"\nF17 − F26: {balance}")
    print(f"Связи газовых расходов: {loose}")

    REPORT.write_text(json.dumps({"подписи": LABELS, "уровни_p05_p50_p95": levels,
                                  "отношения": ratios, "баланс_F17_F26": balance,
                                  "связи": loose}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
