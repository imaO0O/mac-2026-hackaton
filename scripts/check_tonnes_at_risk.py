"""Сколько тонн стоит за решением: эффект в тоннах, а не в процентах. Только CPU.

    python scripts/check_tonnes_at_risk.py

Зачем. Карточка меряет эффект в мг/кг и процентах выпуска. Оператор и начальник
смены считают в тоннах и партиях: «сколько продукта уйдёт в парк, пока не придёт
анализ». Эта величина складывается из двух уже измеренных вещей и не требует ни
одной новой цены: расход товарного потока (тег, а не допущение) и время до
следующего лабораторного анализа (ритм суточный, момент предсказывается точно —
`scripts/check_next_lab.py`).

Тонны под риском = расход × часы до анализа. Это не деньги и не убыток: это объём
продукта, который будет сделан до того, как появится контрольный факт. Умножать на
названную заказчиком цену ошибки (50–100× запаса по качеству) здесь НЕ будем —
проверка `scripts/check_economic_ranking.py` уже показала, чем кончается прямая
подстановка этой цены в решение; в карточке она остаётся словами, а не множителем.

**Правило записано ДО счёта.** Строку показываем, если на ВАЛИДАЦИИ:

1. расход известен (тег есть и положителен) не реже чем в 80 % моментов решения —
   иначе строка будет то появляться, то пропадать без объяснимой причины;
2. величина осмысленна: медиана тонн под риском больше размера партии, которую
   вообще имеет смысл обсуждать (100 т), — иначе говорить не о чем.

Не выполняется — измеренный отказ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "tonnes_at_risk.json"
FLOW_TAG = "ht_F17"          # расход гидроочищенного компонента, т/ч (тег, не допущение)
MIN_KNOWN = 0.80
MIN_BATCH_T = 100.0


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    features = build_feature_matrix()
    masks = time_split(features.index, cfg)
    sb = StateBuilder(cfg)
    lab = sb.lims_sulfur.dropna().sort_index()
    times = lab.index.to_numpy()

    out: dict = {"тег расхода": FLOW_TAG}
    for split in ("val", "test"):
        frame = features[masks[split].to_numpy()]
        if frame.empty or FLOW_TAG not in frame:
            continue
        index = pd.DatetimeIndex(frame.index)
        pos = np.searchsorted(times, index.to_numpy(), side="right")
        ok = pos < len(times)
        when = np.where(ok, times[np.clip(pos, 0, len(times) - 1)], np.datetime64("NaT"))
        wait_h = (when - index.to_numpy()) / np.timedelta64(1, "h")

        flow = frame[FLOW_TAG].to_numpy(dtype=float)
        known = ~np.isnan(flow) & (flow > 0) & ~np.isnan(wait_h)
        tonnes = flow[known] * wait_h[known]
        out[split] = {
            "моментов": int(len(frame)),
            "доля моментов с известным расходом": round(float(known.mean()), 3),
            "расход, т/ч: медиана": round(float(np.median(flow[known])), 1),
            "часы до анализа: медиана": round(float(np.median(wait_h[known])), 1),
            "тонн под риском: медиана": round(float(np.median(tonnes)), 0),
            "тонн под риском: квартили": [round(float(x), 0)
                                          for x in np.percentile(tonnes, [25, 75])],
            "тонн под риском: максимум": round(float(np.max(tonnes)), 0),
        }
        print(f"\n{split}:")
        for key, value in out[split].items():
            print(f"  {key}: {value}")

    val = out.get("val", {})
    rule = {
        "1. расход известен не реже чем в 80 % моментов": bool(
            float(val.get("доля моментов с известным расходом", 0.0)) >= MIN_KNOWN),
        "2. медиана тонн под риском больше 100 т": bool(
            float(val.get("тонн под риском: медиана", 0.0)) > MIN_BATCH_T),
    }
    accepted = all(rule.values())
    print("\nПравило приёмки (валидация):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: тонны под риском в карточке "
          f"{'ПРИНЯТЫ' if accepted else 'НЕ приняты'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "условие: доля моментов с расходом": MIN_KNOWN,
                                  "условие: медиана больше, т": MIN_BATCH_T,
                                  **out, "правило": rule, "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
