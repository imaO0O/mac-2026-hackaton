"""Робастная гарантия: чего она стоит и что даёт. Только CPU.

    python scripts/run_test_period.py --every 1h --tag step1h --no-robust
    python scripts/check_robust_recommendation.py

Зачем. Отклик серы на уставки в системе принят: кинетика первого порядка даёт
−22 % серы на градус при литературных 5–10 %. Робастная гарантия
(`optimization.robust_kinetic_order`) требует, чтобы запас по сере у варианта
держался и при пессимистичной кинетике. Правило приёмки записано в docs/PLAN.md до
счёта:

1. ни при одном порядке процесса шагов с превышением не больше, чем без неё;
2. при порядке процесса 2 вмешательств меньше хотя бы на 20 %;
3. при порядке 1 суммарный ход уставок (в долях разрешённого шага) растёт не
   больше чем в 1.5 раза, вклад в Т95 — не больше чем на 0.1 °C;
4. на тесте (шаг 1 ч) реакция на пробы с превышением в окне −2 … +4 ч не хуже.

Имитация — июнь 2026, шаг 4 ч, процесс отвечает по порядку 1, 1.5 и 2; система —
с робастной гарантией и без. Гарантия включена по умолчанию, поэтому прогон по тесту
с ней — reports/test_period_step1h.json, а без неё делается отдельной командой выше
(около получаса). Пока гарантия была выключена, было наоборот: рабочий прогон — без
неё, отдельный (`--robust-order 2`) — с ней; скрипт понимает оба положения.

Результат: reports/robust_recommendation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.models.kinetics import make_kinetic_surrogate  # noqa: E402
from nefte.models.quality_model import make_model_surrogate  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.sim import ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.check_event_response import window_table  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "robust_recommendation.json"
ORDERS = (1.0, 1.5, 2.0)
EVENT_WINDOW = (-2, 4)


def step_share_movement(steps, cfg, sb) -> float:
    """Суммарный ход уставок в долях разрешённого шага — единая шкала для тегов.

    Шаг расхода задан долей от ТЕКУЩЕГО значения, поэтому база — расход в срезе на
    этот момент. В первой версии база бралась из накопленного сдвига, и каждый
    ход расхода засчитывался как 1/flow_rel шагов: мера давала сотни «шагов» и
    ничего не значила.
    """
    limits = cfg["limits"]["max_step_per_cycle"]
    total = 0.0
    for step in steps:
        if not step.moved:
            continue
        state = None
        for tag, delta in step.moved.items():
            if tag.startswith("T"):
                total += abs(delta) / float(limits["temperature_c"])
            elif tag.startswith("P"):
                total += abs(delta) / float(limits["pressure_mpa"])
            else:
                state = state or sb.build(step.ts)
                now = state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))
                if now:
                    total += abs(delta) / (abs(float(now)) * float(limits["flow_rel"]))
    return round(total, 3)


def closed_loop(sb, cfg, robust: float | None, process_order: float, stamps) -> dict:
    local = {**cfg, "optimization": {**cfg["optimization"], "robust_kinetic_order": robust}}
    system = build_system(sb, local)
    system.log_runs = False
    model = system.quality.model
    process = (system.optimizer.surrogate if process_order == 1.0 else
               make_kinetic_surrogate(model, base_surrogate=make_model_surrogate(model),
                                      order=process_order))
    sim = ClosedLoopSimulator(sb, system, process)
    steps = sim.run(stamps, hist_sulfur=sb.lims_sulfur)
    rep = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                    t95_limit=cfg["spec"]["t95_c"]["max"])
    paired = rep.get("сера_без_вмешательства_сим") or {}
    temps = sum(abs(d) for s in steps for t, d in s.moved.items() if t.startswith("T"))
    return {"вмешательств": rep["вмешательств"],
            "выше предела с нами, шагов": paired.get("выше предела с нами, шагов"),
            "выше предела без нас, шагов": paired.get("выше предела без нас, шагов"),
            "сера с нами": paired.get("среднее с нами"),
            "ход уставок, долей шага": step_share_movement(steps, cfg, sb),
            "ход температур, °C": round(float(temps), 2),
            "вклад в Т95, °C": (rep.get("Т95_наш_вклад") or {}).get("средний сдвиг")}


def event_reaction(path: Path, cfg) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    rows = rows.set_index("ts").sort_index()
    lo, hi = data["summary"]["период"]
    lab = StateBuilder(cfg).lims_sulfur.loc[str(lo):str(hi)]
    table = window_table(rows, lab, float(cfg["spec"]["product_sulfur_mgkg"]["max"]),
                         *EVENT_WINDOW)
    days = max((rows.index[-1] - rows.index[0]).total_seconds() / 86400, 1)
    return {"перед превышением": round(float(table.loc[table["over"], "acted"].mean()), 3),
            "перед нормой": round(float(table.loc[~table["over"], "acted"].mean()), 3),
            "вмешательств в сутки": round(float(rows["исход"].eq("меняем уставки").sum() / days), 2)}


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust-order", type=float, default=2.0)
    ap.add_argument("--window", default="quality_risk")
    ap.add_argument("--every", default="4h")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][args.window])
    stamps = pd.date_range(lo, hi, freq=args.every)

    loops = {}
    for order in ORDERS:
        loops[str(order)] = {
            "без гарантии": closed_loop(sb, cfg, None, order, stamps),
            "с гарантией": closed_loop(sb, cfg, args.robust_order, order, stamps),
        }
        for name, row in loops[str(order)].items():
            print(f"процесс порядка {order:g}, {name}: {row}", flush=True)

    default = cfg["optimization"].get("robust_kinetic_order")
    on_by_default = default is not None and float(default) == args.robust_order
    reports = ROOT / "reports"
    events = {
        "без гарантии": event_reaction(
            reports / ("test_period_step1h_norobust.json" if on_by_default
                       else "test_period_step1h.json"), cfg),
        "с гарантией": event_reaction(
            reports / ("test_period_step1h.json" if on_by_default
                       else f"test_period_step1h_robust{args.robust_order:g}.json"), cfg),
    }
    print("\nТест, шаг 1 ч, окно −2 … +4 ч:", events)

    base1, rob1 = loops["1.0"]["без гарантии"], loops["1.0"]["с гарантией"]
    base2, rob2 = loops["2.0"]["без гарантии"], loops["2.0"]["с гарантией"]
    rules = {
        "1. превышений не больше ни при одном порядке": all(
            (v["с гарантией"]["выше предела с нами, шагов"] or 0)
            <= (v["без гарантии"]["выше предела с нами, шагов"] or 0) for v in loops.values()),
        "2. при порядке 2 вмешательств меньше на 20 %": (
            rob2["вмешательств"] <= 0.8 * base2["вмешательств"]),
        "3. при порядке 1 ход ≤ 1.5× и Т95 +≤0.1 °C": (
            rob1["ход уставок, долей шага"] <= 1.5 * max(base1["ход уставок, долей шага"], 1e-9)
            and (rob1["вклад в Т95, °C"] or 0) <= (base1["вклад в Т95, °C"] or 0) + 0.1),
        "4. реакция на пробы с превышением не хуже": (
            events["с гарантией"] is not None and events["без гарантии"] is not None
            and events["с гарантией"]["перед превышением"]
            >= events["без гарантии"]["перед превышением"]),
    }
    accepted = all(rules.values())
    print("\nПравило приёмки:")
    for name, ok in rules.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: робастная гарантия {'ПРИНЯТА' if accepted else 'НЕ принята'} по умолчанию")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "порядок гарантии": args.robust_order,
                                  "окно": args.window, "шаг": args.every,
                                  "замкнутый_контур": loops, "тест_события": events,
                                  "правило": rules, "принята": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
