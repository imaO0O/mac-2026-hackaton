"""Ранжирование по цене ошибки вместо назначенных весов свёртки.

    python scripts/check_economic_ranking.py

Зачем. Веса свёртки (качество 0.45, выпуск 0.25, энергия 0.15, тяжесть 0.15)
назначены по смыслу — это последнее крупное место, где числа взяты из головы.
Заказчик при этом назвал цену ошибки: партия, ушедшая в некондицию, стоит «в 50–100
раз дороже, чем просто получить запас по качеству» (`docs/transcripts/qa_2026-09-15.txt`,
07:26). Из этого получается ранжирование без назначенных весов:

    ценность = выпуск · (1 − K · P(сера > предела))

то есть ожидаемый выпуск за вычетом ожидаемой потери от некондиции, в тоннах
дизеля. K = 50 — нижняя граница названного диапазона. Тяжесть режима остаётся
жёстким ограничением агента надёжности (она и так сужает диапазон до ранжирования),
энергозатраты в эту формулу не входят: цены энергии в пакете нет, а выдумывать её
ради красоты формулы — ровно то, от чего мы уходим.

**Правило приёмки записано ДО счёта** (`docs/PLAN.md`, план доработки 20.09).
Имитация замкнутого контура на валидационном окне, экономическое ранжирование
против нынешнего:

1. шагов с превышением по сере не больше;
2. вмешательств не больше чем на 20 %;
3. выпуск (средний расход сырья) не ниже.

Не выполняется — остаётся нынешнее ранжирование, и это записывается как измеренный
отказ. Если при этом решения почти не расходятся, вывод отдельный: назначенные веса
согласуются с экономикой заказчика, и менять их незачем.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.agents.optimizer import OptimizerAgent  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import reliability_provenance, report_provenance  # noqa: E402
from nefte.sim import ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "economic_ranking.json"
MISS_COST = 50.0
WINDOW = "stable"
EVERY = "4h"
MORE_ACTIONS = 1.2


def economic_rank(self: OptimizerAgent, cands: list) -> list:
    """Та же выдача, что у rank(), но счёт — ожидаемая ценность в тоннах."""
    feas = [c for c in cands if c.feasible]
    if not feas:
        return []
    for c in feas:
        risk = float((c.spec_risk or {}).get("product_sulfur_mgkg", 0.0))
        throughput = float(c.throughput or 0.0)
        c.score = throughput * (1.0 - MISS_COST * risk)
    # фронт Парето и порядок гарантий сохраняем: это не про веса, а про
    # безопасность и про то, что оператор видит альтернативы
    objectives = np.column_stack([[-float((c.spec_risk or {}).get("product_sulfur_mgkg", 0.0))
                                   for c in feas],
                                  [float(c.throughput or 0.0) for c in feas]])
    grid = np.round(objectives / 1e-3)
    for i, c in enumerate(feas):
        dominated = (np.all(grid >= grid[i], axis=1)
                     & np.any(grid > grid[i], axis=1)).sum()
        c.pareto_rank = int(dominated)
    feas.sort(key=lambda c: (not c.guaranteed, not c.guaranteed_nominal,
                             -(c.score or 0.0), c.pareto_rank))
    return feas


def run(sb, cfg, stamps, economic: bool) -> dict:
    system = build_system(sb, cfg)
    system.log_runs = False
    if economic:
        system.optimizer.rank = economic_rank.__get__(system.optimizer, OptimizerAgent)
    sim = ClosedLoopSimulator(sb, system, system.optimizer.surrogate)
    steps = sim.run(stamps, hist_sulfur=sb.lims_sulfur)
    rep = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                    t95_limit=cfg["spec"]["t95_c"]["max"])
    paired = rep.get("сера_без_вмешательства_сим") or {}
    offsets = pd.DataFrame([s.offsets for s in steps]).fillna(0.0)
    return {
        "вмешательств": rep["вмешательств"],
        "шагов": rep["шагов"],
        "выше предела с нами, шагов": paired.get("выше предела с нами, шагов"),
        "сера с нами": paired.get("среднее с нами"),
        "вклад в Т95, °C": (rep.get("Т95_наш_вклад") or {}).get("средний сдвиг"),
        "средний сдвиг сырья F26": (round(float(offsets["F26"].mean()), 2)
                                    if "F26" in offsets else 0.0),
    }, [s.moved for s in steps]


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)
    lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][WINDOW])
    stamps = pd.date_range(lo, hi, freq=EVERY)
    print(f"окно {WINDOW}: {lo:%Y-%m-%d} … {hi:%Y-%m-%d}, шаг {EVERY}, K={MISS_COST:g}\n")

    base, base_moves = run(sb, cfg, stamps, economic=False)
    print("нынешнее ранжирование:", base, flush=True)
    econ, econ_moves = run(sb, cfg, stamps, economic=True)
    print("экономическое:        ", econ, flush=True)

    differing = sum(1 for a, b in zip(base_moves, econ_moves)
                    if {k: round(v, 3) for k, v in a.items()}
                    != {k: round(v, 3) for k, v in b.items()})
    rule = {
        "1. превышений не больше": bool((econ["выше предела с нами, шагов"] or 0)
                                        <= (base["выше предела с нами, шагов"] or 0)),
        "2. вмешательств не больше чем на 20 %": bool(
            econ["вмешательств"] <= MORE_ACTIONS * max(base["вмешательств"], 1)),
        "3. выпуск не ниже": bool(econ["средний сдвиг сырья F26"]
                                  >= base["средний сдвиг сырья F26"] - 1e-9),
    }
    accepted = all(rule.values())
    print(f"\nрешений разошлось: {differing} из {len(stamps)}")
    print("Правило приёмки:")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: экономическое ранжирование "
          f"{'ПРИНЯТО' if accepted else 'НЕ принято'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), **reliability_provenance(cfg),
                                  "окно": WINDOW, "шаг": EVERY, "K": MISS_COST,
                                  "нынешнее": base, "экономическое": econ,
                                  "решений разошлось": differing,
                                  "шагов всего": len(stamps),
                                  "правило": rule, "принято": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
