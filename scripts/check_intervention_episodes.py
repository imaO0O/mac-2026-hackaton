"""Сколько раз за смену систему «дёргают»: вмешательства эпизодами, а не штуками.

    python scripts/check_intervention_episodes.py

Зачем. Частота вмешательств до сих пор считалась штуками: каждое новое действие —
отдельное вмешательство. Но два подъёма температуры подряд, второй через час после
первого, — это для оператора одно событие, а не два. Ровно на этой мере отклонены
оба рычага снижения частоты (запрет 8/14 ч и повтор при неизменном риске), и в
`docs/PLAN.md` записано, что меру надо переопределить прежде, чем пробовать третий.

Эпизод: действия, идущие подряд с промежутком не больше `GAP_HOURS`, считаются
одним. Промежуток взят тот же, что и в правиле повтора, — 14 ч (3τ отклика серы).

Считает по уже снятым прогонам валидации, ничего не пересчитывая.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "intervention_episodes.json"
GAP_HOURS = 14.0
RUNS = ["val_period_step1h_lock4.json", "val_period_step1h_lock8.json",
        "val_period_step1h_lock14.json", "val_period_step1h_repeat0.05.json",
        "val_period_step1h_repeat0.1.json", "test_period_step1h.json"]


def episodes(times: pd.DatetimeIndex, gap_hours: float) -> int:
    if not len(times):
        return 0
    gaps = times.to_series().diff().dt.total_seconds() / 3600.0
    return int(1 + (gaps > gap_hours).sum())


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    out = {}
    for name in RUNS:
        path = ROOT / "reports" / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = pd.DataFrame(data["rows"])
        rows["ts"] = pd.to_datetime(rows["ts"])
        acted = rows.loc[rows["исход"] == "меняем уставки", "ts"].sort_values()
        days = max((rows["ts"].iloc[-1] - rows["ts"].iloc[0]).total_seconds() / 86400, 1)
        count = int(len(acted))
        eps = episodes(pd.DatetimeIndex(acted), GAP_HOURS)
        out[name] = {
            "шагов": int(len(rows)),
            "действий": count,
            "действий в сутки": round(count / days, 2),
            "эпизодов": eps,
            "эпизодов в сутки": round(eps / days, 2),
            "действий в эпизоде": round(count / eps, 2) if eps else None,
        }
        print(f"{name:38s} {out[name]}")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "промежуток эпизода, ч": GAP_HOURS,
                                  "прогоны": out}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
