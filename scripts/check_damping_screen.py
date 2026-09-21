"""Стоит ли глушить шаг, когда режим уже едет: отсев ДО дорогого прогона. Только CPU.

    python scripts/check_damping_screen.py

Зачем. Проверка `check_already_moving.py` показала: в трети вмешательств режим уже
прошёл целый шаг цикла за время отклика. Напрашивается правка уже не карточки, а
РЕШЕНИЯ: глушить собственный шаг, когда чужой ещё в пути. Но такая правка меняет
решения, а значит стоит многочасовой пересборки, и прежде чем её тратить, надо
понять, не выбросим ли мы полезные вмешательства.

Дешёвый отсев по уже посчитанному прогону валидации: среди вмешательств, которые
глушение ослабило бы, доля подтверждённых лабораторией превышений за 24 ч должна
быть НЕ ВЫШЕ, чем среди остальных. Если выше — глушение убирает действия там, где
они были нужны, и идея закрывается здесь же, без прогона.

**Правило записано ДО счёта.** Идея уходит на прогон, если на ВАЛИДАЦИИ:

1. доля подтверждённых превышений среди «уже едущих» вмешательств не выше, чем
   среди остальных (с запасом на шум — не выше на 2 п.п.);
2. таких вмешательств заметная часть, 10–50 %: меньше — правка не окупит прогона,
   больше — это уже не поправка, а другая система.
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
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "damping_screen.json"
WABT = "reg_wabt"
RESPONSE_H = 4.6
RUNS = {"валидация": "val_period_step1h_lock4.json",
        "тест": "test_period_step1h.json"}
TOLERANCE = 0.02
SHARE_BAND = (0.10, 0.50)


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    step_c = float(cfg["limits"]["max_step_per_cycle"]["temperature_c"])
    features = build_feature_matrix()
    masks = time_split(features.index, cfg)

    out: dict = {"порог движения, °C": step_c, "окно, ч": RESPONSE_H}
    for name, filename in RUNS.items():
        split = "val" if name == "валидация" else "test"
        path = ROOT / "reports" / filename
        frame = features[masks[split].to_numpy()]
        if not path.exists() or frame.empty:
            continue
        step_h = float(pd.Series(frame.index).diff().dt.total_seconds().median() / 3600)
        lag = max(int(round(RESPONSE_H / step_h)), 1)
        change = frame[WABT].astype(float) - frame[WABT].astype(float).shift(lag)

        rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
        rows["ts"] = pd.to_datetime(rows["ts"])
        joined = rows.set_index("ts").join(pd.DataFrame({"ход": change}), how="inner")
        acts = joined[joined["исход"].eq("меняем уставки")]
        known = acts[acts["превышение за 24 ч"].notna()]
        moving = known["ход"].abs() >= step_c
        confirmed = known["превышение за 24 ч"].astype(bool)

        out[name] = {
            "вмешательств": int(len(acts)),
            "из них с известным фактом": int(len(known)),
            "доля «режим уже едет»": round(float(moving.mean()), 3),
            "подтверждённых превышений: уже едет": round(
                float(confirmed[moving].mean()), 3) if moving.any() else None,
            "подтверждённых превышений: остальные": round(
                float(confirmed[~moving].mean()), 3) if (~moving).any() else None,
        }
        print(f"\n{name}:")
        for key, value in out[name].items():
            print(f"  {key}: {value}")

    val = out.get("валидация", {})
    moving_rate = float(val.get("доля «режим уже едет»", 0.0))
    hit_moving = val.get("подтверждённых превышений: уже едет")
    hit_rest = val.get("подтверждённых превышений: остальные")
    rule = {
        "1. глушим не более полезные действия": bool(
            hit_moving is not None and hit_rest is not None
            and hit_moving <= hit_rest + TOLERANCE),
        "2. доля таких вмешательств в 10–50 %": bool(
            SHARE_BAND[0] <= moving_rate <= SHARE_BAND[1]),
    }
    passed = all(rule.values())
    print("\nПравило отсева (валидация):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print("  ВЫВОД: правка решения "
          + ("заслуживает прогона" if passed
             else "закрыта здесь же, прогон не нужен"))

    REPORT.write_text(json.dumps({**report_provenance(cfg), **out,
                                  "условие: запас на шум": TOLERANCE,
                                  "условие: доля в полосе": list(SHARE_BAND),
                                  "правило": rule, "на прогон": passed},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
