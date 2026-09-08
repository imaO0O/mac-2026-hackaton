"""Сквозной прогон цикла МАС на одном моменте времени или на демо-окне.

    python scripts/run_cycle.py --ts "2026-04-20 12:00"
    python scripts/run_cycle.py --window bad_data_frozen_pak --every 6h

Это «скелет» демонстрации: агенты пока на базовых реализациях, но полный путь
данные → 4 агента → рекомендация оператору работает и логируется в reports/runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.optimizer import OptimizerAgent, linear_surrogate  # noqa: E402
from nefte.agents.orchestrator import Orchestrator  # noqa: E402
from nefte.agents.quality import QualityAgent  # noqa: E402
from nefte.agents.reliability import ReliabilityAgent, SeverityNorms  # noqa: E402
from nefte.config import load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402

# Кандидатные управляющие воздействия для базового прогона (см. configs/config.yaml).
CONTROL_TAGS = ["T5", "T11", "F26", "P13"]

# Линейные чувствительности серы к уставкам — ЗАГЛУШКА до обученной модели.
# Знак взят из физики и подтверждён корреляциями (рост температуры → падение серы).
SENSITIVITIES = {"T5": -0.15, "T11": -0.15, "F26": 0.01, "P13": -0.5}


def build_system(sb: StateBuilder, cfg: dict) -> Orchestrator:
    norms = SeverityNorms.fit(
        pd.concat([sb.ht[["T5", "T6", "T11", "W10"]], sb.avt[["T55"]]], axis=1)
        .assign(wabt=lambda d: d[["T5", "T6", "T11"]].mean(axis=1)),
        columns=["wabt", "W10", "T55"],
    )
    bounds = sb.model_bounds(CONTROL_TAGS, unit="ht")
    optimizer = OptimizerAgent(bounds=bounds, surrogate=linear_surrogate(SENSITIVITIES), cfg=cfg)
    return Orchestrator(QualityAgent(cfg=cfg), ReliabilityAgent(norms), optimizer, cfg=cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", help="момент времени, например '2026-04-20 12:00'")
    ap.add_argument("--window", help="имя окна из configs/config.yaml: demo_windows")
    ap.add_argument("--every", default="12h", help="шаг обхода окна")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    system = build_system(sb, cfg)

    if args.window:
        lo, hi = cfg["demo_windows"][args.window]
        stamps = pd.date_range(lo, hi, freq=args.every)
    else:
        stamps = [pd.Timestamp(args.ts or "2026-04-20 12:00")]

    for ts in stamps:
        rec = system.run(sb.build(ts))
        print(rec.to_operator_text())
        print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
