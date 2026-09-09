"""Блок смешения: рецептура товарного ДТ на заданный момент. Только CPU.

    python scripts/run_blending.py --ts "2026-02-28 00:00"
    python scripts/run_blending.py --ts "2026-02-28 00:00" --additive 500

Показывает компоненты с их свойствами, подобранную рецептуру, проверку жёстких
ограничений (включая сумму долей = 100 %) и предельную долю прямогонной фракции.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.blending import BlendingAgent, components_from_data  # noqa: E402
from nefte.config import load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="2026-02-28 00:00", help="момент времени")
    ap.add_argument("--additive", type=float, default=0.0,
                    help="дозировка депрессорной присадки, ppm")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    ts = pd.Timestamp(args.ts)

    components = components_from_data(sb, ts)
    print(f"\nКомпоненты на {ts:%Y-%m-%d %H:%M}\n")
    print(pd.DataFrame([{
        "компонент": c.name,
        "сера, мг/кг": round(c.sulfur_mgkg, 1),
        "D15": None if c.density_15c is None else round(c.density_15c, 1),
        "Т95": c.t95_c,
        "ПТФ": c.cfpp_c,
        "расход, т/ч": round(c.available_tph, 1),
        "допущение": "да" if c.is_assumption else "нет",
    } for c in components]).to_string(index=False))

    agent = BlendingAgent(cfg)
    recipe = agent.optimize(components, additive_ppm=args.additive)

    print(f"\nРецептура (сумма долей {recipe.fractions_sum():.4f}):")
    for name, share in recipe.fractions.items():
        print(f"  {name:28s} {share * 100:6.2f} %")
    print("\nСвойства смеси:", {k: round(v, 2) for k, v in recipe.properties.items()})
    print(f"Выпуск: {recipe.throughput_tph:.1f} т/ч; присадка: {recipe.additive_ppm:.0f} ppm")
    print(f"Допустима: {'да' if recipe.feasible else 'НЕТ'}")
    for v in recipe.violations:
        print(f"  нарушение: {v}")
    for n in recipe.notes:
        print(f"  {n}")

    base = min(components, key=lambda c: c.sulfur_mgkg)
    print("\nПредельная доля компонентов при разбавлении базовым "
          f"«{base.name}» (сера {base.sulfur_mgkg:.1f} мг/кг):")
    for c in components:
        if c is base:
            continue
        print(f"  {c.name:28s} не более {agent.max_share_of(base, c) * 100:.4f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
