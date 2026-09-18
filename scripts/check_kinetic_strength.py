"""Какую кинетику отклика брать рабочей: измерение против принятого. Только CPU.

    python scripts/check_kinetic_strength.py

Зачем. Отклик серы на уставки в системе был ПРИНЯТ (первый порядок, 100 кДж/моль), а
теперь ИЗМЕРЕН по 48 чистым эпизодам (`docs/SULFUR_RESPONSE.md`, участник 2):
наблюдаемое изменение серы — 0.035 от того, что обещает первый порядок (на
валидации 0.056, 90 % интервал −0.038…0.145). Первый порядок измерением исключён;
второй (0.14) лежит в интервале валидации, но не обучения.

Три варианта рабочей кинетики (`optimization.kinetic_order`, `kinetic_strength`):

* **A** — первый порядок в полную силу, как сейчас: −19 % серы на градус;
* **B** — второй порядок: −2.9 % на градус (0.14 от первого);
* **C** — первый порядок с измеренной силой 0.035: −0.7 % на градус.

**Правило приёмки записано ДО счёта** (`docs/PLAN.md`): берётся вариант, у которого
обещанный отклик ближе всего к измеренному НА ВАЛИДАЦИИ (сила 0.035 подобрана на
обучающем периоде, поэтому судим по валидации), при двух обязательных условиях,
считаемых на валидационном окне имитации:

1. шагов с превышением по сере не больше, чем у варианта A;
2. вклад в Т95 не выше, чем у A, больше чем на 0.1 °C.

Ни один не проходит условия — остаётся A, и это записывается как измеренный отказ.
Робастная гарантия в сравнении не участвует: она проверяет запас при более слабой
кинетике и остаётся включённой при любом выборе.

Результат: reports/kinetic_strength.json.
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
from nefte.models.regime import REACTOR_TEMPS  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import reliability_provenance, report_provenance  # noqa: E402
from nefte.sim import ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "kinetic_strength.json"
MEASURED = ROOT / "reports" / "sulfur_response.json"
# окно валидационного периода: выбор делается только на нём
WINDOW = "stable"
VARIANTS = {
    "A: первый порядок, как принято": {"kinetic_order": 1.0, "kinetic_strength": 1.0},
    "B: второй порядок": {"kinetic_order": 2.0, "kinetic_strength": 1.0},
    "C: первый порядок, измеренная сила": {"kinetic_order": 1.0, "kinetic_strength": 0.035},
}
T95_TOLERANCE = 0.1


def measured_beta() -> tuple[float | None, list | None]:
    """Измеренная доля первого порядка на ВАЛИДАЦИИ и её 90 % интервал."""
    if not MEASURED.exists():
        return None, None
    data = json.loads(MEASURED.read_text(encoding="utf-8"))
    block = ((data.get("итог") or {}).get("все чистые") or {}).get("val") or {}
    return block.get("β"), block.get("90% интервал")


def predicted_beta(system, sb, stamps, base_fn) -> float:
    """Во сколько раз обещанный вариантом отклик слабее первого порядка.

    Считается тем же способом, что и измерение: подъём реакторных температур на
    1 °C и относительное изменение серы, усреднённое по моментам окна.
    """
    ours, base = [], []
    for ts in stamps:
        state = sb.build(ts)
        temps = {t: state.telemetry_ht.get(t) for t in REACTOR_TEMPS}
        if any(v is None for v in temps.values()):
            continue
        hotter = {t: v + 1.0 for t, v in temps.items()}
        for fn, box in ((system.optimizer.surrogate, ours), (base_fn, base)):
            now = fn(state, {})["product_sulfur_mgkg"]
            up = fn(state, hotter)["product_sulfur_mgkg"]
            if now == now and up == up and now > 0:
                box.append((up - now) / now)
    if not ours or not base:
        return float("nan")
    return float(pd.Series(ours).mean() / pd.Series(base).mean())


def run_variant(sb, cfg, knobs: dict, stamps) -> dict:
    local = {**cfg, "optimization": {**cfg["optimization"], **knobs}}
    system = build_system(sb, local)
    system.log_runs = False
    sim = ClosedLoopSimulator(sb, system, system.optimizer.surrogate)
    steps = sim.run(stamps, hist_sulfur=sb.lims_sulfur)
    rep = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                    t95_limit=cfg["spec"]["t95_c"]["max"])
    paired = rep.get("сера_без_вмешательства_сим") or {}
    return {
        "настройки": knobs,
        "вмешательств": rep["вмешательств"],
        "шагов": rep["шагов"],
        "выше предела с нами, шагов": paired.get("выше предела с нами, шагов"),
        "выше предела без нас, шагов": paired.get("выше предела без нас, шагов"),
        "сера с нами": paired.get("среднее с нами"),
        "вклад в Т95, °C": (rep.get("Т95_наш_вклад") or {}).get("средний сдвиг"),
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default=WINDOW, help="окно валидационного периода")
    ap.add_argument("--every", default="4h")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][args.window])
    stamps = pd.date_range(lo, hi, freq=args.every)
    beta, interval = measured_beta()
    print(f"измеренная доля первого порядка на валидации: {beta} {interval}")
    print(f"окно {args.window}: {lo:%Y-%m-%d} … {hi:%Y-%m-%d}, шаг {args.every}\n")

    base_system = build_system(sb, {**cfg, "optimization": {
        **cfg["optimization"], "kinetic_order": 1.0, "kinetic_strength": 1.0}})
    base_fn = base_system.optimizer.surrogate

    rows = {}
    for name, knobs in VARIANTS.items():
        system = build_system(sb, {**cfg, "optimization": {**cfg["optimization"], **knobs}})
        system.log_runs = False
        row = run_variant(sb, cfg, knobs, stamps)
        row["обещанная доля первого порядка"] = round(
            predicted_beta(system, sb, stamps[::4], base_fn), 4)
        if beta is not None:
            row["расхождение с измеренным"] = round(
                abs(row["обещанная доля первого порядка"] - float(beta)), 4)
        rows[name] = row
        print(f"{name}: {row}", flush=True)

    base = rows["A: первый порядок, как принято"]
    verdict = {}
    for name, row in rows.items():
        over_ok = (row["выше предела с нами, шагов"] or 0) <= (
            base["выше предела с нами, шагов"] or 0)
        t95_ok = (row["вклад в Т95, °C"] or 0.0) <= (base["вклад в Т95, °C"] or 0.0) + T95_TOLERANCE
        verdict[name] = {"превышений не больше, чем у A": bool(over_ok),
                         "вклад в Т95 не хуже A на 0.1 °C": bool(t95_ok),
                         "допустим": bool(over_ok and t95_ok)}
    allowed = [n for n, v in verdict.items() if v["допустим"]]
    choice = min(allowed, key=lambda n: rows[n].get("расхождение с измеренным", 1e9)) \
        if allowed and beta is not None else "A: первый порядок, как принято"

    print("\nПравило приёмки:")
    for name, v in verdict.items():
        print(f"  {name}: {v}")
    print(f"  ВЫБОР: {choice}")

    REPORT.write_text(json.dumps({
        **report_provenance(cfg), **reliability_provenance(cfg),
        "окно": args.window, "шаг": args.every,
        "измеренная_доля_первого_порядка": {"валидация": beta, "90% интервал": interval},
        "варианты": rows, "правило": verdict, "выбор": choice,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
