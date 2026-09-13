"""Устойчивость рекомендации к весам свёртки оптимизатора. Только CPU.

    python scripts/check_objective_weights.py
    python scripts/check_objective_weights.py --draws 30 --stamps 60 --spread 0.5

Веса свёртки (0.45 запас по качеству, 0.25 выпуск, 0.15 энергия, 0.15 тяжесть
режима) выбраны по смыслу: экономических данных в пакете нет, обучать их не на
чем. Тот же вопрос, что и к весам severity, — «а если бы они были другие» —
закрываем числом.

Важно, что именно проверяется. Веса участвуют **только в ранжировании уже
допустимых вариантов**: жёсткие ограничения отсекают недопустимое до свёртки, а
решение «вмешиваться или нет» принимает оркестратор по риску. Поэтому исход
(держим / меняем / отказ) от весов почти не зависит, и мерить надо не его, а
конкретные числа рекомендации:

1. сохраняется ли НАПРАВЛЕНИЕ воздействия по реакторным температурам;
2. насколько гуляет сама величина ΔT5;
3. насколько гуляет прогноз серы у выбранного варианта.

Если направление сохраняется, а величина плавает в пределах доли градуса, веса
влияют на оттенок решения, а не на его смысл. Это и надо сказать на защите.

Результат: reports/objective_weights.json и таблица в консоли.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "objective_weights.json"


def perturbed(base: dict[str, float], rng: np.random.Generator,
              spread: float) -> dict[str, float]:
    """Случайные веса вокруг базовых, сумма сохраняется."""
    factors = rng.uniform(1 - spread, 1 + spread, len(base))
    values = np.array(list(base.values()), dtype=float) * factors
    values = values / values.sum() * sum(base.values())
    return dict(zip(base.keys(), values))


def action_summary(rec) -> dict[str, float | str | None]:
    """Что именно рекомендовано: исход, шаг по T5 и прогноз серы."""
    if rec.abstained or rec.action is None:
        return {"исход": "отказ", "dT5": None, "сера": None}
    return {
        # правило исхода — одно на всю систему, в Recommendation.outcome()
        "исход": rec.outcome(),
        "dT5": float(rec.action.deltas.get("T5", 0.0)),
        "сера": rec.action.predicted_quality.get("product_sulfur_mgkg"),
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20, help="случайных наборов весов")
    # 40, а не 10: на десяти моментах с действием оказывался ОДИН, и средняя
    # «величина гуляет» была значением в одной точке (docs/OPTIMIZER_AGENT.md)
    ap.add_argument("--stamps", type=int, default=40, help="моментов времени")
    ap.add_argument("--spread", type=float, default=0.5, help="разброс весов, доля")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    system = build_system(sb, cfg)
    system.log_runs = False
    rng = np.random.default_rng(cfg["optimization"]["random_seed"])

    lo, hi = cfg["split"]["test"]
    stamps = pd.date_range(lo, hi, periods=args.stamps)
    states = [sb.build(ts) for ts in stamps]

    base_weights = dict(cfg["optimization"]["objective_weights"])
    print(f"[1/2] базовый прогон на {len(states)} моментах…")
    base = [action_summary(system.run(s)) for s in states]

    print(f"[2/2] {args.draws} наборов весов (разброс ±{args.spread:.0%})…")
    per_stamp: list[dict] = [{"dT5": [], "сера": [], "исход": []} for _ in states]
    try:
        for _ in range(args.draws):
            cfg["optimization"]["objective_weights"] = perturbed(base_weights, rng,
                                                                 args.spread)
            for i, state in enumerate(states):
                # лимит частоты воздействий — состояние оркестратора: без сброса
                # второй прогон одного и того же момента проверял бы первый
                system._last_action_ts = None
                result = action_summary(system.run(state))
                per_stamp[i]["исход"].append(result["исход"])
                per_stamp[i]["dT5"].append(result["dT5"])
                per_stamp[i]["сера"].append(result["сера"])
    finally:
        cfg["optimization"]["objective_weights"] = base_weights

    rows = []
    for i, ts in enumerate(stamps):
        outcomes = per_stamp[i]["исход"]
        deltas = [d for d in per_stamp[i]["dT5"] if d is not None]
        sulfur = [v for v in per_stamp[i]["сера"] if v is not None]
        base_delta = base[i]["dT5"]
        same_outcome = np.mean([o == base[i]["исход"] for o in outcomes]) if outcomes else 1.0
        same_sign = (np.mean([np.sign(d) == np.sign(base_delta) for d in deltas])
                     if deltas and base_delta is not None else None)
        rows.append({
            "момент": str(ts)[:16],
            "базовый исход": base[i]["исход"],
            "исход сохраняется": round(float(same_outcome), 2),
            "направление ΔT5": None if same_sign is None else round(float(same_sign), 2),
            "ΔT5 базовый": None if base_delta is None else round(base_delta, 2),
            "ΔT5 разброс": (round(float(np.percentile(deltas, 95)
                                        - np.percentile(deltas, 5)), 2) if deltas else None),
            "сера разброс": (round(float(np.percentile(sulfur, 95)
                                         - np.percentile(sulfur, 5)), 3) if sulfur else None),
        })

    frame = pd.DataFrame(rows)
    print()
    print(frame.to_string(index=False))

    acting = frame[frame["базовый исход"] == "меняем уставки"]
    summary = {
        "spread": args.spread,
        "draws": args.draws,
        "stamps": args.stamps,
        "weights": base_weights,
        "исход сохраняется": round(float(frame["исход сохраняется"].mean()), 3),
        # Сколько моментов реально несут информацию о величине воздействия.
        # Без этого числа средние ниже читаются как устойчивая статистика, а они
        # считаются ТОЛЬКО по моментам с действием — и однажды их оказался ровно
        # ОДИН из десяти. Между двумя сборками средняя «величина ΔT5 гуляет»
        # сменилась с 1.90 на 0.315 просто потому, что сменился тот один момент.
        "моментов с действием": int(len(acting)),
        "направление сохраняется": (round(float(acting["направление ΔT5"].mean()), 3)
                                    if len(acting) and acting["направление ΔT5"].notna().any()
                                    else None),
        "ΔT5 разброс, среднее": (round(float(acting["ΔT5 разброс"].mean()), 3)
                                 if len(acting) else None),
        "сера разброс, среднее": (round(float(acting["сера разброс"].mean()), 3)
                                  if len(acting) else None),
        "per_stamp": rows,
    }

    print(f"\nИсход сохраняется в {summary['исход сохраняется']:.0%} случаев.")
    if summary["направление сохраняется"] is not None:
        print(f"Направление воздействия по T5 сохраняется в "
              f"{summary['направление сохраняется']:.0%} случаев, величина гуляет "
              f"в среднем на {summary['ΔT5 разброс, среднее']:.2f} °C, "
              f"прогноз серы — на {summary['сера разброс, среднее']:.3f} мг/кг.")
        n_acting = summary["моментов с действием"]
        print(f"ВНИМАНИЕ: эти два средних посчитаны по {n_acting} моментам с "
              f"действием из {args.stamps}. Остальные — бездействие, у которого "
              f"разброса величины нет по определению.")
        if n_acting < 5:
            print("  Меньше пяти точек: числа выше — иллюстрация, а не оценка. "
                  "Берите их с --stamps побольше, если нужно утверждение.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    summary = {**report_provenance(), **summary}
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
