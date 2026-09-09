"""Обучение агента качества (участник 1). Только CPU, без внешних сервисов.

    python scripts/train_quality.py                 # горизонт 2 ч
    python scripts/train_quality.py --horizon 6     # прогноз на 6 часов вперёд

Пишет модель в models/sulfur/ и отчёт в reports/quality_metrics.json.
Сравнение всегда с базовыми: показание ПАК и предыдущий лабораторный анализ.
Модель, которая их не бьёт, в систему не идёт.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import (  # noqa: E402
    build_feature_matrix,
    build_training_table,
    persistence_baselines,
)
from nefte.models.regime import INSTANT_FEATURES  # noqa: E402
from nefte.models.quality_model import (  # noqa: E402
    SulfurModel,
    baseline_metrics,
    controllable_features,
)

# Кандидатные управляющие воздействия: модель обязана их видеть, иначе оптимизатор
# не сможет оценивать сценарии (см. configs/config.yaml → controls).
CONTROL_TAGS = ["T5", "T11", "F26", "P13", "F15", "P24"]
CONTROL_COLUMNS = ([f"ht_{t}" for t in CONTROL_TAGS] + ["avt_T55"]
                   + INSTANT_FEATURES)   # признаки режима тоже обязаны остаться


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=2.0, help="горизонт прогноза, часов")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--top-features", type=int, default=40,
                    help="сколько признаков оставить после первого прохода (0 — все)")
    ap.add_argument("--no-monotone", action="store_true",
                    help="не зашивать физическое направление отклика в модель")
    args = ap.parse_args()

    cfg = load_config()
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]

    print("[1/4] матрица признаков…")
    feats = build_feature_matrix()
    print(f"      {feats.shape[0]} моментов × {feats.shape[1]} признаков")

    print(f"[2/4] обучающая таблица, горизонт {args.horizon} ч…")
    X, y = build_training_table(horizon_hours=args.horizon, features=feats)
    masks = time_split(X.index, cfg)
    parts = {k: (X[m.to_numpy()], y[m.to_numpy()]) for k, m in masks.items()}
    for name, (xx, _) in parts.items():
        print(f"      {name}: {len(xx)} анализов")

    print("[3/4] обучение CatBoost (CPU)…")
    model = SulfurModel(horizon_hours=args.horizon, iterations=args.iterations,
                        monotone=not args.no_monotone)
    model.fit(*parts["train"], *parts["val"], top_features=args.top_features or None,
              must_keep=CONTROL_COLUMNS)
    print(f"      признаков после отбора: {len(model.features)}; "
          f"управляющих среди них: {len(controllable_features(model, CONTROL_TAGS))}")

    scale = model.calibrate(*parts["val"])
    print(f"      конформная поправка σ: ×{scale:.2f} "
          f"(калибровка по val, честное покрытие смотрим на test)")

    a, b = model.calibrate_risk(*parts["val"])
    print(f"      калибровка вероятности (Платт): a={a:.2f}, b={b:.2f}")
    src = model.select_risk_source(*parts["val"])
    print(f"      источник вероятности: {src}; PR-AUC "
          f"{ {k: round(v, 3) for k, v in getattr(model, 'risk_source_scores', {}).items()} }, "
          f"разброс { {k: round(v, 3) for k, v in getattr(model, 'risk_source_spreads', {}).items()} }")
    thr = model.select_alarm_threshold(*parts["val"])
    if model.alarm_reliable:
        print(f"      порог тревоги (F1.5 при precision >= 1.5x базовой частоты): {thr:.2f}")
    else:
        print("      ВНИМАНИЕ: на валидации ни один порог не даёт тревогу информативнее "
              "базовой частоты нарушений.")
        print(f"      Переходим на прозрачное правило «прогноз выше предела» "
              f"(источник риска: {model.risk_source}, порог {thr:.2f}).")

    print("[4/4] оценка…")
    report = {"horizon_hours": args.horizon, "n_features": len(model.features),
              "monotone": model.monotone,
              "sigma_scale": model.sigma_scale, "alarm_threshold": model.alarm_threshold,
              "splits": {}}
    for name in ("train", "val", "test"):
        Xp, yp = parts[name]
        if not len(yp):
            continue
        block = {"model": model.evaluate(Xp, yp, limit)}
        for bname, bpred in persistence_baselines(Xp, yp).items():
            if bpred.notna().any():
                block[f"baseline_{bname}"] = baseline_metrics(bpred, yp, limit)
        report["splits"][name] = block

    model.metrics = report
    path = model.save()
    out = ROOT / "reports" / f"quality_metrics_h{args.horizon:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nмодель: {path}\nотчёт:  {out}\n")
    rows = []
    for split, block in report["splits"].items():
        for who, m in block.items():
            def _r(key):
                v = m.get(key)
                return None if v is None else round(v, 3)
            rows.append({"split": split, "модель": who, "n": m["n"],
                         ">10": m.get("n_over_limit"),
                         "MAE": round(m["MAE"], 3), "RMSE": round(m["RMSE"], 3),
                         "bias": round(m["bias"], 3),
                         "покрытие80": _r("coverage_80"),
                         "prec@порог": _r("spec_precision"),
                         "rec@порог": _r("spec_recall"),
                         "ROC-AUC": _r("roc_auc")})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nТоп признаков:")
    print(model.feature_importance(12).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
