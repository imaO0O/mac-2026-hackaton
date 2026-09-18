"""Перепад Р-202 вариантом «прирост»: включать или нет (участник 2). Только CPU.

    python scripts/run_test_period.py --split val --every 1h --tag step1h --dp-factor off
    python scripts/run_test_period.py --split val --every 1h --tag step1h --dp-factor growth
    python scripts/check_dp_growth.py

Правило приёмки записано ДО счёта (docs/PLAN.md, участник 1) и исполняется как есть.
Вариант «прирост» включается, если на ВАЛИДАЦИИ одновременно:

1. доля тяжёлого класса не выше 5 %;
2. отказов «нет допустимых вариантов» не больше, чем у выключенного;
3. реакция на пробы с превышением в окне −2 … +4 ч (доля проб, перед которыми
   система вмешалась, `check_event_response.py`) падает не больше чем на 5 пунктов
   относительно выключенного;
4. фактор различим: разброс p05–p95 на валидации не меньше 0.05.

Не выполняется хотя бы одно — фактор остаётся выключенным. Шаг прогона 1 ч — как у
прочих правил на валидации с окном −2 … +4 ч (запрет частых воздействий, повтор).

Результат: reports/dp_growth.json и таблица в консоли.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.agents.reliability import ReliabilityAgent  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.loaders import load_telemetry  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.check_event_response import window_table  # noqa: E402

RUNS = {"off": "val_period_step1h_dp_off.json", "growth": "val_period_step1h_dp_growth.json"}
WINDOW = (-2, 4)
MAX_HIGH = 0.05
MAX_REACTION_DROP = 0.05
MIN_SPREAD = 0.05
REPORT = ROOT / "reports" / "dp_growth.json"


def run_metrics(name: str, lab: pd.Series, limit: float) -> dict:
    data = json.loads((ROOT / "reports" / RUNS[name]).read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    rows = rows.set_index("ts").sort_index()
    lo, hi = data["summary"]["период"]
    table = window_table(rows, lab.loc[str(lo):str(hi)], limit, *WINDOW)
    return {"моментов": len(rows),
            "reliability_settings": data.get("reliability_settings"),
            "тяжёлый класс": round(float(rows["risk_class"].eq("high").mean()), 4),
            "нет допустимых вариантов": int(rows["причина отказа"].eq(
                "нет допустимых вариантов").sum()),
            "вмешательств": int(rows["исход"].eq("меняем уставки").sum()),
            "реакция перед превышением": round(float(table.loc[table["over"], "acted"].mean()), 3),
            "перед нормальной": round(float(table.loc[~table["over"], "acted"].mean()), 3),
            "проб с превышением": int(table["over"].sum())}


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    sb = StateBuilder(cfg)
    missing = [f for f in RUNS.values() if not (ROOT / "reports" / f).exists()]
    if missing:
        print("нет прогонов: " + ", ".join(missing))
        return 1
    runs = {name: run_metrics(name, sb.lims_sulfur, limit) for name in RUNS}

    local = {**cfg, "reliability": {**(cfg.get("reliability") or {}), "dp_factor": "growth"}}
    agent = ReliabilityAgent.from_history(sb.avt, sb.ht, local, raw_ht=load_telemetry("ht"))
    factor = agent.dp_series
    spreads = {}
    for split in ("train", "val"):
        part = factor.loc[cfg["split"][split][0]:cfg["split"][split][1]].dropna()
        spreads[split] = {"p05": round(float(part.quantile(.05)), 3),
                          "p50": round(float(part.median()), 3),
                          "p95": round(float(part.quantile(.95)), 3),
                          "разброс": round(float(part.quantile(.95) - part.quantile(.05)), 3)}

    off, growth = runs["off"], runs["growth"]
    conditions = {
        "1. тяжёлый класс не выше 5 %": growth["тяжёлый класс"] <= MAX_HIGH,
        "2. отказов «нет вариантов» не больше, чем у off":
            growth["нет допустимых вариантов"] <= off["нет допустимых вариантов"],
        "3. реакция на превышение падает не больше 5 пунктов":
            growth["реакция перед превышением"] >= off["реакция перед превышением"]
            - MAX_REACTION_DROP,
        "4. разброс фактора p05–p95 не меньше 0.05": spreads["val"]["разброс"] >= MIN_SPREAD,
    }
    accepted = all(conditions.values())

    print(pd.DataFrame(runs).T.drop(columns="reliability_settings").to_string())
    print(f"\nФактор «прирост» (0…1.5): {spreads}")
    print("\nПравило приёмки (валидация, шаг 1 ч):")
    for name, ok in conditions.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print("\nВЫВОД: " + ("вариант «прирост» ПРИНЯТ — включается, тест прогоняется один раз"
                         if accepted else "вариант «прирост» НЕ принят — фактор остаётся "
                         "выключенным, второй измеренный отказ"))

    REPORT.write_text(json.dumps({**report_provenance(cfg), "окно, ч": list(WINDOW),
                                  "прогоны": runs, "фактор": spreads,
                                  "условия": conditions, "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
