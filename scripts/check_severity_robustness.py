"""Устойчивость решений к весам severity. Только CPU.

    python scripts/check_severity_robustness.py                # разброс весов ±50 %
    python scripts/check_severity_robustness.py --spread 0.2   # ±20 %, отдельный отчёт

Веса severity (0.30 WABT, 0.20 ΔP, 0.15 аномалия, 0.15 скорость, 0.10 наработка,
0.10 печь) подобраны по смыслу: разметки отказов нет, обучать их не на чем. Это
самое слабое место агента надёжности, поэтому вопрос «а если бы веса были другие»
нужно закрыть числом, а не словами.

Скрипт возмущает веса случайно и смотрит, что меняется:

1. **risk_class** — на нём висит сужение диапазонов для оптимизатора;
2. **итоговое решение** — держим режим, меняем уставки или отказываемся.

Если решение переживает разумный разброс весов, значит оно опирается на данные,
а не на конкретные числа в конфиге.

**Средняя устойчивость сама по себе ничего не доказывает, и это уже стоило
выводов.** Момент, где система держит режим далеко от порога, сохраняет исход
при любых весах. Прежний вывод «исход сохраняется в 99 %» был посчитан на восьми
моментах, из которых действие было ровно в одном, — то есть это была
устойчивость выборки, а не системы. У участника 1 то же случилось со свёрткой
оптимизатора. Поэтому устойчивость считается отдельно по каждому базовому исходу,
а для risk_class — отдельно у границы класса, и скрипт называет, на скольких
нетривиальных моментах стоит вывод.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nefte.agents.reliability import ReliabilityAgent  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.loaders import load_telemetry  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402


def perturbed_weights(base: dict[str, float], rng: np.random.Generator,
                      spread: float = 0.5) -> dict[str, float]:
    """Случайные веса вокруг базовых: каждый умножается на множитель из [1-s, 1+s]."""
    factors = rng.uniform(1 - spread, 1 + spread, len(base))
    values = np.array(list(base.values())) * factors
    values = values / values.sum()
    return dict(zip(base.keys(), values))


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100, help="случайных наборов весов")
    ap.add_argument("--stamps", type=int, default=24, help="моментов времени")
    ap.add_argument("--decision-draws", type=int, default=10,
                    help="наборов весов для полного цикла (дороже)")
    ap.add_argument("--decision-stamps", type=int, default=40,
                    help="моментов для полного цикла; при 8 действие было в одном")
    ap.add_argument("--spread", type=float, default=0.5, help="разброс весов, доля")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    base_agent = ReliabilityAgent.from_history(sb.avt, sb.ht, cfg,
                                               raw_ht=load_telemetry("ht"))
    rng = np.random.default_rng(cfg["optimization"]["random_seed"])

    lo, hi = cfg["split"]["test"]
    stamps = pd.date_range(lo, hi, periods=args.stamps)
    states = [sb.build(ts) for ts in stamps]

    # ---------- 1. устойчивость risk_class --------------------------------
    print(f"[1/2] risk_class на {len(states)} моментах × {args.draws} наборах весов…")
    base_classes = [base_agent.assess(s).risk_class for s in states]
    base_severity = [base_agent.assess(s).severity_index for s in states]

    flips = np.zeros(len(states))
    severity_spread = [[] for _ in states]
    original = dict(ReliabilityAgent.WEIGHTS)
    try:
        for _ in range(args.draws):
            ReliabilityAgent.WEIGHTS = perturbed_weights(original, rng, args.spread)
            for i, state in enumerate(states):
                result = base_agent.assess(state)
                severity_spread[i].append(result.severity_index)
                if result.risk_class != base_classes[i]:
                    flips[i] += 1
    finally:
        ReliabilityAgent.WEIGHTS = original

    class_stability = 1 - flips / args.draws
    severity_range = [float(np.percentile(v, 95) - np.percentile(v, 5))
                      for v in severity_spread]
    print(f"      risk_class сохраняется в {class_stability.mean():.0%} случаев "
          f"(худший момент: {class_stability.min():.0%})")
    print(f"      разброс severity (p05…p95): в среднем {np.mean(severity_range):.3f}, "
          f"максимум {np.max(severity_range):.3f}")

    # Класс вдали от порога сохраняется при любых весах, и такие моменты тянут
    # среднее к единице. Вопрос осмыслен только у границы — считаем её отдельно.
    thresholds = base_agent.thresholds
    margin = ReliabilityAgent.CLASS_BOUNDARY_MARGIN
    near = np.array([min(abs(value - t) for t in thresholds) <= margin
                     for value in base_severity])
    n_near = int(near.sum())
    near_stability = float(class_stability[near].mean()) if n_near else None
    far_stability = float(class_stability[~near].mean()) if (~near).any() else None
    print(f"      у границы класса (±{margin} от порогов {thresholds[0]:.2f} / "
          f"{thresholds[1]:.2f}): {n_near} моментов из {len(states)}")
    if near_stability is not None and far_stability is not None:
        print(f"      устойчивость класса у границы {near_stability:.0%}, "
              f"вдали от неё {far_stability:.0%}")
    if n_near < 5:
        print(f"      ВНИМАНИЕ: у границы всего {n_near} моментов — средняя "
              "устойчивость класса держится на тривиальных случаях")

    # ---------- 2. устойчивость итогового решения -------------------------
    from run_cycle import build_system

    print(f"[2/2] полный цикл на {args.decision_stamps} моментах × "
          f"{args.decision_draws} наборах весов…")
    system = build_system(sb, cfg)
    system.log_runs = False
    decision_stamps = pd.date_range(lo, hi, periods=args.decision_stamps)
    decision_states = [sb.build(ts) for ts in decision_stamps]

    def run_once(state):
        """Один независимый прогон момента.

        Лимит частоты воздействий — состояние оркестратора, и между независимыми
        наборами весов его надо сбрасывать. Без сброса второй прогон того же
        момента попадал в ветку «недавно уже вмешивались» и возвращал «держим
        режим» — то есть измерялась не чувствительность к весам, а собственная
        память системы. Именно так и появилась «неустойчивая точка» в прошлом
        отчёте: неустойчивым оказывался ровно тот момент, где базовый исход был
        «меняем уставки».
        """
        system._last_action_ts = None
        return system.run(state).outcome()

    base_outcomes = [run_once(s) for s in decision_states]

    changed = np.zeros(len(decision_states))
    try:
        for _ in range(args.decision_draws):
            ReliabilityAgent.WEIGHTS = perturbed_weights(original, rng, args.spread)
            for i, state in enumerate(decision_states):
                if run_once(state) != base_outcomes[i]:
                    changed[i] += 1
    finally:
        ReliabilityAgent.WEIGHTS = original

    decision_stability = 1 - changed / args.decision_draws
    print(f"      исход решения сохраняется в {decision_stability.mean():.0%} случаев "
          f"(худший момент: {decision_stability.min():.0%})")
    print(f"      базовые исходы: {pd.Series(base_outcomes).value_counts().to_dict()}")

    # По исходам отдельно: «держим режим» далеко от порога тривиально устойчив, и
    # среднее по выборке, где таких семь из восьми, говорит о выборке.
    by_outcome = {}
    for label in ("меняем уставки", "держим режим", "отказ"):
        idx = [i for i, outcome in enumerate(base_outcomes) if outcome == label]
        if not idx:
            continue
        values = decision_stability[idx]
        by_outcome[label] = {"моментов": len(idx),
                             "устойчивость": round(float(values.mean()), 3),
                             "худший момент": round(float(values.min()), 3)}
        print(f"      {label}: {len(idx)} моментов, исход сохраняется в "
              f"{values.mean():.0%} (худший момент {values.min():.0%})")
    n_acting = by_outcome.get("меняем уставки", {}).get("моментов", 0)
    if n_acting < 5:
        print(f"      ВНИМАНИЕ: моментов с действием {n_acting} из "
              f"{len(decision_states)} — вывод об устойчивости решения на них "
              "не держится, поднимите --decision-stamps")

    per_stamp = pd.DataFrame({
        "момент": [str(ts)[:16] for ts in decision_stamps],
        "базовый исход": base_outcomes,
        "устойчивость": [round(float(x), 2) for x in decision_stability],
        "severity": [round(base_agent.assess(s).severity_index, 2) for s in decision_states],
        "risk_class": [base_agent.assess(s).risk_class for s in decision_states],
    })
    print()
    print(per_stamp.to_string(index=False))

    report = {
        "spread": args.spread,
        "weights": original,
        "risk_class": {
            "draws": args.draws, "stamps": args.stamps,
            "mean_stability": round(float(class_stability.mean()), 3),
            "worst_stability": round(float(class_stability.min()), 3),
            "severity_range_mean": round(float(np.mean(severity_range)), 3),
            "моментов у границы": n_near,
            "устойчивость у границы": (None if near_stability is None
                                       else round(near_stability, 3)),
            "устойчивость вдали": (None if far_stability is None
                                   else round(far_stability, 3)),
        },
        "decision": {
            "draws": args.decision_draws, "stamps": args.decision_stamps,
            "mean_stability": round(float(decision_stability.mean()), 3),
            "worst_stability": round(float(decision_stability.min()), 3),
            "base_outcomes": pd.Series(base_outcomes).value_counts().to_dict(),
            "моментов с действием": n_acting,
            "по исходам": by_outcome,
            "per_stamp": per_stamp.to_dict("records"),
        },
    }
    # Основной отчёт — разброс ±50 %; другие пишутся рядом, чтобы таблица
    # документации сверялась с отчётом по каждой строке, а не по одной.
    suffix = "" if abs(args.spread - 0.5) < 1e-9 else f"_s{round(args.spread * 100)}"
    out = ROOT / "reports" / f"severity_robustness{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {**report_provenance(), **report}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
