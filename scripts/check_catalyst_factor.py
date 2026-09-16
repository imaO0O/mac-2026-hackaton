"""Износ катализатора в severity: какой вариант включать (участник 2, план 14.09, п. 1).

    python scripts/run_test_period.py --split test --every 4h --tag step4h \
        --catalyst-factor age --catalyst-reset outage_48h
    python scripts/run_test_period.py --split test --every 4h --tag step4h \
        --catalyst-factor age --catalyst-reset catalyst_log
    python scripts/run_test_period.py --split test --every 4h --tag step4h \
        --catalyst-factor activity --catalyst-reset outage_48h
    (то же с --split val; настройки явно — «было» не должно зависеть от конфига)
    python scripts/check_catalyst_factor.py
    python scripts/check_catalyst_factor.py --events-only     # только проверка износа

Зачем. Фактор ``catalyst`` в severity — «часы от последнего останова дольше 48 ч»,
то есть возраст. Из трёх длительных остановов истории катализатор меняли на двух;
на третьем, ремонте 20–30.06.2026, возраст обнулился, а катализатор продолжал
стареть. Варианты:

* **было** — возраст, сброс на любом останове дольше 48 ч;
* **(б)** — возраст, сброс только по журналу замен (``reliability.catalyst_changes``);
* **(в)** — уровень WABT, приведённой к нагрузке (models/catalyst.activity_level_series).

Правило приёмки записано в docs/PLAN.md ДО счёта, здесь оно исполняется как есть:

1. **износ** — после двух замен фактор падает не меньше чем на 0.2, после ремонта
   без замены — меньше чем на 0.1 (медиана за 30 суток до останова против медианы
   за 30–40 сутки после пуска);
2. **валидация** — доли классов риска, суженные границы, отказы «нет допустимых
   вариантов»; описательно;
3. **тест** — сколько решений поменялось (описательно); разница «перед превышением
   − перед нормой» на каждом окне не ниже нижней границы 90 % интервала у «было»;
   имитация без возврата на трёх окнах: шагов выше предела, вмешательств и качелей
   не больше, чем у «было».

Результат: таблицы в консоли и reports/catalyst_factor.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.agents.reliability import ReliabilityAgent  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.episodes import outage_intervals  # noqa: E402
from nefte.data.loaders import load_telemetry  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.backtest_reliability import severity_series  # noqa: E402
from scripts.check_event_response import WINDOWS, bootstrap_difference, window_table  # noqa: E402
from scripts.check_return_to_base import WINDOWS as SIM_WINDOWS  # noqa: E402
from scripts.check_return_to_base import run as run_sim  # noqa: E402

VARIANTS = {
    "было": {"settings": {"catalyst_factor": "age", "catalyst_reset": "outage_48h"},
             "suffix": "_catalyst_age_reset_outage_48h"},
    "б": {"settings": {"catalyst_factor": "age", "catalyst_reset": "catalyst_log"},
          "suffix": "_catalyst_age_reset_catalyst_log"},
    "в": {"settings": {"catalyst_factor": "activity", "catalyst_reset": "outage_48h"},
          "suffix": "_catalyst_activity_reset_outage_48h"},
}
RUN_TAG = "step4h"

LONG_OUTAGE_HOURS = 48.0
CHANGE_TOLERANCE = pd.Timedelta(days=2)     # пуск рядом с датой из журнала — замена
BEFORE_DAYS = 30
AFTER_DAYS = (30, 40)
CHANGE_MIN_DROP = 0.2
REPAIR_MAX_DROP = 0.1
SIM_KEYS = ("выше_предела_шагов_с_нами", "вмешательств", "качели")
REPORT = ROOT / "reports" / "catalyst_factor.json"


def num(value) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) \
        else round(float(value), 3)


def variant_cfg(cfg: dict, name: str) -> dict:
    return {**cfg, "reliability": {**(cfg.get("reliability") or {}),
                                   **VARIANTS[name]["settings"]}}


def wear_factor(agent: ReliabilityAgent) -> pd.Series:
    """Ровно тот ряд, который агент кладёт в severity как фактор ``catalyst``."""
    if agent.catalyst_series is not None:
        return agent.catalyst_series.clip(0.0, 1.5)
    return (agent.run_hours / agent.run_hours_scale).clip(0.0, 1.5)


def wear_check(factor: pd.Series, outages: list[tuple], changes: list[pd.Timestamp]) -> list[dict]:
    rows = []
    for start, end in outages:
        change = any(abs(end - mark) <= CHANGE_TOLERANCE for mark in changes)
        before = factor.loc[start - pd.Timedelta(days=BEFORE_DAYS):start].median()
        after = factor.loc[end + pd.Timedelta(days=AFTER_DAYS[0]):
                           end + pd.Timedelta(days=AFTER_DAYS[1])].median()
        drop = before - after
        if pd.isna(drop):
            ok = False
        else:
            ok = bool(drop >= CHANGE_MIN_DROP) if change else bool(drop < REPAIR_MAX_DROP)
        rows.append({"останов": str(start)[:16], "пуск": str(end)[:16],
                     "событие": "замена катализатора" if change else "ремонт без замены",
                     "фактор до": num(before), "фактор после": num(after),
                     "падение": num(drop), "проходит": ok})
    return rows


def class_shares(agent: ReliabilityAgent, severity: pd.Series, lo: str, hi: str) -> dict:
    part = severity.loc[lo:hi].dropna()
    medium, high = agent.thresholds
    low_share = float((part < medium).mean())
    high_share = float((part >= high).mean())
    return {"пороги": [num(medium), num(high)], "low": num(low_share),
            "medium": num(1.0 - low_share - high_share), "high": num(high_share)}


def load_run(split: str, name: str) -> tuple[dict, pd.DataFrame] | None:
    stem = "test_period" if split == "test" else "val_period"
    path = ROOT / "reports" / f"{stem}_{RUN_TAG}{VARIANTS[name]['suffix']}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    return data, rows.set_index("ts").sort_index()


def operations(rows: pd.DataFrame) -> dict:
    """Суженные границы — классы medium и high: агент сужает температуры реакторов."""
    n = len(rows)
    return {"моментов": n,
            "risk_class": {str(k): num(v / n)
                           for k, v in rows["risk_class"].value_counts().items()},
            "суженные_границы": num(rows["risk_class"].isin(["medium", "high"]).mean()),
            "нет_допустимых_вариантов": num(rows["причина отказа"].eq(
                "нет допустимых вариантов").mean()),
            "исходы": {str(k): int(v) for k, v in rows["исход"].value_counts().items()}}


def event_response(data: dict, rows: pd.DataFrame, lab: pd.Series, limit: float) -> dict:
    lo_p, hi_p = data["summary"]["период"]
    lab = lab.loc[str(lo_p):str(hi_p)]
    step_h = pd.Timedelta(data["summary"]["шаг"]).total_seconds() / 3600
    out = {}
    for lo, hi in WINDOWS:
        if hi - lo < step_h:
            continue
        table = window_table(rows, lab, limit, lo, hi)
        if table.empty or table["over"].nunique() < 2:
            continue
        hit = float(table.loc[table["over"], "acted"].mean())
        base = float(table.loc[~table["over"], "acted"].mean())
        # одно зерно на окно у всех вариантов: интервалы сравнимы между собой
        out[f"[{lo:+d}, {hi:+d})"] = {"разница": num(hit - base),
                                      "90% интервал": bootstrap_difference(
                                          table, f"catalyst/{lo}/{hi}")}
    return out


def not_more(value, base) -> bool:
    if value is None or base is None:
        return value is None and base is None
    return value <= base


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-only", action="store_true",
                    help="только проверка износа на длительных остановах")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    raw_ht = load_telemetry("ht")
    changes = [pd.Timestamp(x) for x in (cfg.get("reliability") or {}).get("catalyst_changes", [])]
    outages = outage_intervals(raw_ht["F26"], min_hours=LONG_OUTAGE_HOURS)
    val_lo, val_hi = cfg["split"]["val"]
    print(f"Длительных остановов (≥ {LONG_OUTAGE_HOURS:g} ч): {len(outages)}; "
          f"замены по журналу: {', '.join(str(c)[:16] for c in changes)}\n")

    result: dict = {}
    for name in VARIANTS:
        local = variant_cfg(cfg, name)
        agent = ReliabilityAgent.from_history(sb.avt, sb.ht, local, raw_ht=raw_ht)
        wear = wear_check(wear_factor(agent), outages, changes)
        block = {"настройки": VARIANTS[name]["settings"], "износ": wear,
                 "износ_проходит": all(row["проходит"] for row in wear) and len(wear) == 3}
        print(f"=== {name}: {VARIANTS[name]['settings']}")
        print(pd.DataFrame(wear).to_string(index=False))
        if not args.events_only:
            factors = severity_series(agent, sb.avt, sb.ht)
            block["валидация_severity"] = class_shares(agent, factors["severity"], val_lo, val_hi)
            print(f"  классы риска на валидации по severity: {block['валидация_severity']}")
        print()
        result[name] = block

    if args.events_only:
        return 0

    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    lab = sb.lims_sulfur
    base_test = load_run("test", "было")
    for name in VARIANTS:
        block = result[name]
        val_run = load_run("val", name)
        block["валидация_прогон"] = operations(val_run[1]) if val_run else None
        test_run = load_run("test", name)
        if test_run is None or base_test is None:
            block["тест"] = None
            continue
        data, rows = test_run
        joined = pd.concat([base_test[1]["исход"].rename("было"), rows["исход"].rename("стало")],
                           axis=1).dropna()
        block["тест"] = {
            "решений_поменялось": int((joined["было"] != joined["стало"]).sum()),
            "моментов": len(joined),
            "отклик_на_превышение": event_response(data, rows, lab, limit),
            "имитация_без_возврата": {w: run_sim(sb, variant_cfg(cfg, name), w, False)
                                      for w in SIM_WINDOWS},
        }

    base = result["было"].get("тест")
    for name in ("б", "в"):
        block, test = result[name], result[name].get("тест")
        if not base or not test:
            block["тест_проходит"] = False
            block["тест_почему"] = "нет прогонов по тесту"
            continue
        failures = []
        base_er, var_er = base["отклик_на_превышение"], test["отклик_на_превышение"]
        if set(base_er) != set(var_er):
            failures.append("разный набор окон отклика")
        for window, stats in base_er.items():
            got = var_er.get(window, {}).get("разница")
            if got is None or got < stats["90% интервал"][0]:
                failures.append(f"отклик {window}: {got} ниже {stats['90% интервал'][0]}")
        for window in SIM_WINDOWS:
            for key in SIM_KEYS:
                value = test["имитация_без_возврата"][window].get(key)
                reference = base["имитация_без_возврата"][window].get(key)
                if not not_more(value, reference):
                    failures.append(f"имитация {window}, {key}: {value} против {reference}")
        block["тест_проходит"] = not failures
        block["тест_почему"] = failures

    rows = []
    for name, block in result.items():
        test = block.get("тест") or {}
        val = block.get("валидация_прогон") or {}
        rows.append({"вариант": name, "износ": block["износ_проходит"],
                     "суженные границы (вал.)": val.get("суженные_границы"),
                     "нет вариантов (вал.)": val.get("нет_допустимых_вариантов"),
                     "решений поменялось (тест)": test.get("решений_поменялось"),
                     "тест": block.get("тест_проходит", "—")})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    for name in ("б", "в"):
        if result[name].get("тест_почему"):
            print(f"  {name}: " + "; ".join(result[name]["тест_почему"])
                  if isinstance(result[name]["тест_почему"], list) else result[name]["тест_почему"])

    passed = {name: result[name]["износ_проходит"] and result[name].get("тест_проходит", False)
              for name in ("б", "в")}
    choice = "в" if passed["в"] else "б" if passed["б"] else None
    print("\nВЫВОД: " + (f"включается вариант ({choice})" if choice else
                         "ни один вариант не прошёл — выключатели остаются выключенными"))

    REPORT.write_text(json.dumps({
        **report_provenance(cfg),
        "правило": {"длительный_останов_ч": LONG_OUTAGE_HOURS, "до_сут": BEFORE_DAYS,
                    "после_сут": list(AFTER_DAYS), "замена_падение_не_меньше": CHANGE_MIN_DROP,
                    "ремонт_падение_меньше": REPAIR_MAX_DROP, "шаг_прогона": "4h"},
        "длительные_остановы": [[str(a)[:16], str(b)[:16]] for a, b in outages],
        "варианты": result, "прошли": passed, "решение": choice,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
