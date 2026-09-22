"""Поправка границ на старение катализатора: включать или нет (участник 2). Только CPU.

    python scripts/run_test_period.py --split val --every 4h --tag step4h --aging-bounds off
    python scripts/run_test_period.py --split val --every 4h --tag step4h --aging-bounds on
    python scripts/check_aging_bounds.py

Зачем. Диапазоны уставок сняты с обучающего периода, где катализатору было 14.4 мес,
а он стареет на 0.85 °C/мес (`reports/catalyst_life.json`). На старшем катализаторе
верхняя граница T5 — устаревшая: оптимизатор не может поднять температуру туда, где
она нужна, а при выходе больше шага цикла тег вовсе выпадает из рассмотрения.

Правило приёмки записано участником 1 ДО счёта (docs/PLAN.md) и исполняется как есть:
поправка включается, если на ВАЛИДАЦИИ одновременно

1. шагов с превышением серы не больше, чем без неё;
2. вклад в Т95 выше не более чем на 0.5 °C;
3. отказов «нет допустимых вариантов» становится меньше.

Первые два — из имитации замкнутого контура на валидационном окне (то же окно и тот
же код, что у `scripts/check_return_to_base.py`), третий — из прогонов по валидации.

Результат: reports/aging_bounds.json.
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

WINDOW = "stable"                      # окно валидационного периода
RUNS = {"off": "val_period_step4h_aging_off.json", "on": "val_period_step4h_aging_on.json"}
MAX_T95_RISE = 0.5
REPORT = ROOT / "reports" / "aging_bounds.json"


def simulate(sb, cfg: dict, enabled: bool) -> dict:
    lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][WINDOW])
    local = {**cfg, "optimization": {**cfg["optimization"], "aging_bounds": enabled}}
    system = build_system(sb, local)
    system.log_runs = False
    sim = ClosedLoopSimulator(sb, system, system.optimizer.surrogate)
    steps = sim.run(pd.date_range(lo, hi, freq="4h"), hist_sulfur=sb.lims_sulfur)
    rep = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                    t95_limit=cfg["spec"]["t95_c"]["max"])
    without = rep.get("сера_без_вмешательства_сим") or {}
    return {"шагов": rep["шагов"], "вмешательств": rep["вмешательств"],
            "выше предела, шагов": without.get("выше предела с нами, шагов"),
            "выше предела без нас, шагов": without.get("выше предела без нас, шагов"),
            "сера среднее": rep["сера_сим"]["среднее"],
            "вклад в Т95, °C": (rep.get("Т95_наш_вклад") or {}).get("средний сдвиг"),
            "сдвиг границы применялся": bool(enabled)}


def refusals(name: str) -> dict | None:
    path = ROOT / "reports" / RUNS[name]
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    return {"моментов": summary["моментов"], "исходы": summary["исходы"],
            "нет допустимых вариантов": summary["причины отказа"].get(
                "нет допустимых вариантов", 0)}


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)
    sim = {name: simulate(sb, cfg, name == "on") for name in ("off", "on")}
    runs = {name: refusals(name) for name in RUNS}

    print(f"Имитация на окне «{WINDOW}» ({cfg['demo_windows'][WINDOW][0]} … "
          f"{cfg['demo_windows'][WINDOW][1]}):")
    print(pd.DataFrame(sim).T.to_string())
    if all(runs.values()):
        print("\nПрогон по валидации, шаг 4 ч:")
        print(pd.DataFrame({k: {"моментов": v["моментов"],
                                "нет допустимых вариантов": v["нет допустимых вариантов"],
                                **v["исходы"]} for k, v in runs.items()}).T.to_string())

    steps_ok = (sim["on"]["выше предела, шагов"] is not None
                and sim["on"]["выше предела, шагов"] <= sim["off"]["выше предела, шагов"])
    t95_off, t95_on = sim["off"]["вклад в Т95, °C"], sim["on"]["вклад в Т95, °C"]
    t95_ok = t95_off is None or t95_on is None or t95_on <= t95_off + MAX_T95_RISE
    refuse_ok = (all(runs.values())
                 and runs["on"]["нет допустимых вариантов"]
                 < runs["off"]["нет допустимых вариантов"])
    conditions = {
        "1. шагов с превышением не больше": bool(steps_ok),
        "2. вклад в Т95 выше не более чем на 0.5 °C": bool(t95_ok),
        "3. отказов «нет допустимых вариантов» меньше": bool(refuse_ok),
    }
    accepted = all(conditions.values())
    print("\nПравило приёмки (валидация):")
    for name, ok in conditions.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print("\nВЫВОД: " + ("поправка на старение ПРИНЯТА — включается" if accepted else
                         "поправка не принята — остаётся выключенной, измеренный отказ"))

    REPORT.write_text(json.dumps({**report_provenance(cfg), "окно": WINDOW,
                                  "имитация": sim, "прогоны": runs,
                                  "условия": conditions, "принято": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
