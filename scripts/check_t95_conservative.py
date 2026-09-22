"""Уровень Т95 в жёстком ограничении: последний анализ или верхняя граница доверия.

    python scripts/check_t95_conservative.py

Зачем. Т95 — второе жёсткое ограничение, и самое слабое: MAE оценок около 5 °C при
запасе до предела 3–5 °C. `scripts/check_t95_combo.py` показал: «последний анализ
+ σ(возраст)» ловит 64 % превышений на валидации и 80 % на тесте против 36 % и 0 %
у последнего анализа, ценой 21.8 % ложных против 5.3 %. Для Т95 «ложное» — не
остановка, а сужение набора вариантов, поэтому решать надо по РЕШЕНИЯМ, а не по
долям ложных.

Кандидат `quality.t95_estimate: last_plus_sigma` поднимает уровень Т95 на σ ухода
показателя — только в проверке жёсткого ограничения оптимизатора, не в прогнозе и не
в риске.

**Правило приёмки записано до счёта** (docs/PLAN.md, «Т95: четыре попытки…»).
Кандидат заменит нынешнюю оценку, если на валидации одновременно:

1. шагов с Т95 выше предела не больше, чем сейчас — по имитации замкнутого контура:
   в прогоне без контура решения на Т95 не влияют, и мерить там нечего;
2. выпуск падает не более чем на 1 % — по среднему сдвигу расхода сырья в имитации;
3. число вмешательств растёт не более чем на четверть — по прогону по валидации.

Прогоны по валидации — по файлу на режим: `val_period_step1h_t95_<режим>.json`
(`run_test_period.py --split val --every 1h --tag step1h --t95-estimate <режим>`);
режим из конфига берётся из базового `val_period_step1h_lock4.json`, чтобы после
включения кандидат не сравнивался сам с собой. Имитация — окно `stable` валидации.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.sim import ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "t95_conservative.json"
MODES = ("last", "last_plus_sigma")
BASE = "val_period_step1h_lock4.json"
MAX_THROUGHPUT_DROP = 0.01
MAX_ACTION_GROWTH = 0.25
FEED = "F26"


def with_mode(cfg: dict, mode: str) -> dict:
    return {**cfg, "quality": {**cfg["quality"], "t95_estimate": mode}}


def simulate(sb, cfg: dict, mode: str) -> dict:
    lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"]["stable"])
    system = build_system(sb, with_mode(cfg, mode))
    system.log_runs = False
    sim = ClosedLoopSimulator(sb, system, system.optimizer.surrogate)
    steps = sim.run(pd.date_range(lo, hi, freq="4h"), hist_sulfur=sb.lims_sulfur)
    rep = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                    t95_limit=cfg["spec"]["t95_c"]["max"])
    feed_level = float(sb.ht[FEED].loc[lo:hi].mean())
    feed_shift = [step.offsets.get(FEED, 0.0) for step in steps]
    paired = rep.get("сера_без_вмешательства_сим") or {}
    return {"шагов": rep["шагов"], "вмешательств": rep["вмешательств"],
            "Т95 выше предела, доля шагов": (rep.get("Т95") or {}).get("доля выше предела"),
            "вклад в Т95, °C": (rep.get("Т95_наш_вклад") or {}).get("средний сдвиг"),
            "сдвиг сырья, % уровня": round(100 * float(pd.Series(feed_shift).mean())
                                           / feed_level, 3) if feed_level else None,
            "сера выше предела с нами, шагов": paired.get("выше предела с нами, шагов")}


def run_name(mode: str, default: str) -> str | None:
    """Прогон режима: свой файл, а для режима из конфига — базовый прогон."""
    explicit = f"val_period_step1h_t95_{mode}.json"
    if (ROOT / "reports" / explicit).exists():
        return explicit
    return BASE if mode == default else None


def actions(name: str | None) -> int | None:
    path = ROOT / "reports" / name if name else None
    if path is None or not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return int(summary["исходы"].get("меняем уставки", 0))


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    default = str(cfg["quality"].get("t95_estimate", "last"))
    runs = {mode: run_name(mode, default) for mode in MODES}
    acts = {mode: actions(name) for mode, name in runs.items()}
    if any(v is None for v in acts.values()):
        print("нет прогонов по валидации — см. докстринг")
        return 1
    sb = StateBuilder(cfg)
    sims = {mode: simulate(sb, cfg, mode) for mode in MODES}
    print(f"валидация, вмешательств: {acts}")
    for mode, stats in sims.items():
        print(f"  имитация {mode:16s} {stats}")

    base, cand = sims["last"], sims["last_plus_sigma"]
    rule = {
        "1. Т95 выше предела — не чаще, чем сейчас":
            (cand["Т95 выше предела, доля шагов"] or 0) <= (base["Т95 выше предела, доля шагов"] or 0),
        "2. выпуск падает не более чем на 1 %":
            (cand["сдвиг сырья, % уровня"] or 0) - (base["сдвиг сырья, % уровня"] or 0)
            >= -100 * MAX_THROUGHPUT_DROP - 1e-9,
        "3. вмешательств больше не более чем на четверть":
            acts["last_plus_sigma"] <= acts["last"] * (1 + MAX_ACTION_GROWTH),
    }
    accepted = all(rule.values())
    print("\nПравило приёмки:")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: t95_estimate = {'last_plus_sigma' if accepted else 'last (кандидат отклонён)'}")

    REPORT.write_text(json.dumps({
        **report_provenance(cfg), "окно имитации": cfg["demo_windows"]["stable"],
        "прогоны": runs, "вмешательств на валидации": acts, "имитация": sims,
        "правило": {k: bool(v) for k, v in rule.items()}, "принят": bool(accepted),
        "выбор": "last_plus_sigma" if accepted else "last",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
