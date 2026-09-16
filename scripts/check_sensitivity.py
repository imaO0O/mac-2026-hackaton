"""Чувствительность прогноза серы к управляющим уставкам. Только CPU.

    python scripts/check_sensitivity.py
    python scripts/check_sensitivity.py --horizon 2 --sample 300

Оптимизатор сравнивает сценарии через суррогат «режим → качество». Если модель
почти не реагирует на изменение уставок, все варианты выглядят одинаково и выбор
становится бессмысленным. Этот скрипт измеряет реакцию напрямую: берёт моменты
из тестового периода, сдвигает каждую уставку на типичный шаг цикла и смотрит,
насколько меняется прогноз.

Ориентир: сдвиг реакторной температуры на 2 °C должен менять прогноз заметно —
десятые доли мг/кг, а не сотые.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

# Шаги, на которых меряем реакцию: соответствуют limits.max_step_per_cycle
# (расходы — 3 % от медианы: квенч F14 ~6 т/ч, свежий ВСГ F25 ~13 600 нм3/ч).
STEPS = {"T5": 2.0, "T11": 2.0, "T6": 2.0, "P13": 0.05, "F26": -8.0,
         "F14": 0.2, "F25": 400.0}


def measure(model: SulfurModel, matrix: pd.DataFrame, steps: dict[str, float],
            sample: int = 200, seed: int = 42) -> pd.DataFrame:
    """Средний сдвиг прогноза при изменении каждой уставки."""
    try:
        from nefte.models.regime import apply_moves_to_rows
    except ImportError:                       # признаков режима ещё нет
        apply_moves_to_rows = None

    rows = matrix.dropna(subset=[c for c in matrix.columns if c.startswith("ht_")][:1])
    if len(rows) > sample:
        rows = rows.sample(sample, random_state=seed).sort_index()

    base = model.predict_frame(rows[model.features])["q50"]
    out = []
    for tag, step in steps.items():
        column = f"ht_{tag}" if f"ht_{tag}" in matrix.columns else f"avt_{tag}"
        if column not in matrix.columns:
            continue
        moved = rows.copy()
        if apply_moves_to_rows is not None:
            moved = apply_moves_to_rows(moved, {tag: step}, relative=True)
        else:
            moved[column] = moved[column] + step
        pred = model.predict_frame(moved[model.features])["q50"]
        delta = pred - base
        out.append({
            "уставка": tag,
            "шаг": step,
            "средний |Δ| прогноза": round(float(delta.abs().mean()), 4),
            "медианный Δ": round(float(delta.median()), 4),
            "доля реагирующих точек": round(float((delta.abs() > 0.01).mean()), 3),
        })
    return pd.DataFrame(out).sort_values("средний |Δ| прогноза", ascending=False)


def measure_surrogates(model: SulfurModel, cfg: dict, matrix: pd.DataFrame,
                       n: int = 12) -> list[dict]:
    """Реакция того суррогата, которым реально пользуется оптимизатор.

    Сравниваем чисто статистический (прогноз модели с пересчётом режима) и
    гибридный кинетический: именно разница между ними объясняет, почему без
    физики оптимизатор выбирал из неразличимых вариантов.
    """
    from nefte.models.kinetics import make_kinetic_surrogate
    from nefte.models.quality_model import make_model_surrogate
    from nefte.pipeline import StateBuilder

    sb = StateBuilder(cfg)
    model.attach(matrix)
    statistical = make_model_surrogate(model)
    kinetic = make_kinetic_surrogate(model, base_surrogate=statistical)

    lo, hi = cfg["split"]["test"]
    stamps = pd.date_range(lo, hi, periods=n)

    rows = []
    for name, fn in (("статистический", statistical), ("кинетический", kinetic)):
        deltas = []
        for ts in stamps:
            state = sb.build(ts)
            temps = {t: state.telemetry_ht.get(t) for t in ("T5", "T6", "T11")}
            if any(v is None for v in temps.values()):
                continue
            base = fn(state, {})["product_sulfur_mgkg"]
            hotter = fn(state, {t: v + 2.0 for t, v in temps.items()})["product_sulfur_mgkg"]
            if base == base and hotter == hotter:
                deltas.append(hotter - base)
        if deltas:
            series = pd.Series(deltas)
            rows.append({
                "суррогат": name,
                "n": len(series),
                "средний Δ при +2 °C": round(float(series.mean()), 3),
                "средний |Δ|": round(float(series.abs().mean()), 3),
                "доля реагирующих": round(float((series.abs() > 0.01).mean()), 2),
            })

    if rows:
        print()
        print('Реакция суррогата на подъём реакторных температур на 2 °C:')
        print()
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        print('Отрицательный Δ — правильное направление: выше температура, ниже сера.')
    return rows


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=0.0)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--no-surrogate", action="store_true",
                    help="не мерить суррогат, которым пользуется оптимизатор")
    args = ap.parse_args()

    cfg = load_config()
    model = SulfurModel.load(SulfurModel.default_path(args.horizon))
    matrix = build_feature_matrix()

    masks = time_split(matrix.index, cfg)
    test = matrix[masks["test"].to_numpy()]
    table = measure(model, test, STEPS, sample=args.sample)

    print(f"\nЧувствительность прогноза (горизонт {args.horizon:g} ч, "
          f"признаков в модели {len(model.features)}):\n")
    print(table.to_string(index=False))

    total = float(table["средний |Δ| прогноза"].sum())
    print(f"\nСуммарная реакция по всем уставкам: {total:.4f} мг/кг")
    if total < 0.05:
        print("ВЫВОД: модель практически не реагирует на уставки — оптимизатор "
              "выбирает из неразличимых вариантов.")

    surrogate_rows = []
    if not args.no_surrogate:
        surrogate_rows = measure_surrogates(model, cfg, matrix, n=12)

    out = ROOT / "reports" / f"sensitivity_h{args.horizon:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report_provenance(),
                               "horizon_hours": args.horizon,
                               "n_features": len(model.features),
                               "total_response": total,
                               "rows": table.to_dict(orient="records"),
                               "surrogates": surrogate_rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
