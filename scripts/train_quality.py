"""Обучение агента качества (участник 1). Только CPU, без внешних сервисов.

    python scripts/train_quality.py                 # горизонт 2 ч
    python scripts/train_quality.py --horizon 6     # прогноз на 6 часов вперёд
    python scripts/train_quality.py --target t95 --horizon 0   # второй показатель

Пишет модель в models/<показатель>_h<горизонт>/ и отчёт в
reports/quality_metrics_*.json. Сравнение всегда с базовыми: показание ПАК,
предыдущий лабораторный анализ, а для Т95 ещё и формула виртуального анализатора
из справочника. Модель, которая их не бьёт, в систему не идёт.

**Про Т95.** Организаторы назвали его обязательным показателем наравне с серой.
До сих пор он держался на формуле ВАК: она даёт верный УРОВЕНЬ (смещение −0.8 °C),
но MAE 5.5 °C и корреляцию 0.30 — на таком числе жёсткий предел 360 °C был бы
ложной точностью, поэтому оптимизатор брал от формулы только приращение. Между тем
лабораторных анализов Т95 на обучающем периоде **900** — столько же, сколько по
сере. То есть показатель можно моделировать полноценно, и главная проверка здесь
одна: бьёт ли модель формулу справочника.
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
    FEATURE_VERSION,
    QUALITY_TARGETS,
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
from nefte.utils import use_utf8_console  # noqa: E402

# Кандидатные управляющие воздействия: модель обязана их видеть, иначе оптимизатор
# не сможет оценивать сценарии (см. configs/config.yaml → controls).
CONTROL_TAGS = ["T5", "T11", "F26", "P13", "F15", "P24"]
CONTROL_COLUMNS = ([f"ht_{t}" for t in CONTROL_TAGS] + ["avt_T55"]
                   + INSTANT_FEATURES)   # признаки режима тоже обязаны остаться


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=2.0, help="горизонт прогноза, часов")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--top-features", type=int, default=40,
                    help="сколько признаков оставить после первого прохода (0 — все)")
    ap.add_argument("--no-monotone", action="store_true",
                    help="не зашивать физическое направление отклика в модель")
    ap.add_argument("--no-vak", action="store_true",
                    help="выбросить признаки vak_* — абляция вклада формул справочника")
    ap.add_argument("--seed", type=int, default=42,
                    help="сид бустинга и проб отбора. Не 42 — это проверка устойчивости: "
                         "модель и отчёт получают суффикс _s<сид> и рабочую не трогают")
    ap.add_argument("--tag", default="",
                    help="суффикс имени модели и отчёта: чтобы абляция не затирала рабочую модель")
    ap.add_argument("--target", default="sulfur", choices=sorted(QUALITY_TARGETS),
                    help="какой показатель прогнозируем")
    args = ap.parse_args()

    cfg = load_config()
    spec = QUALITY_TARGETS[args.target]
    node = cfg
    for key in spec["limit_key"]:
        node = node[key]
    limit = float(node)
    print(f"показатель: {spec['name']} ({spec['unit']}), предел {limit:g}")

    print("[1/4] матрица признаков…")
    feats = build_feature_matrix()
    if args.no_vak:
        # Абляция: считаем ту же матрицу и снимаем ровно признаки справочника.
        # Пересборка матрицы без них дала бы другой кэш и другие пропуски, и
        # сравнение перестало бы быть сравнением одного и того же.
        dropped = [c for c in feats.columns if c.startswith("vak_")]
        feats = feats.drop(columns=dropped)
        print(f"      абляция: снято {len(dropped)} признаков vak_*")
    print(f"      {feats.shape[0]} моментов × {feats.shape[1]} признаков")

    print(f"[2/4] обучающая таблица, горизонт {args.horizon} ч…")
    X, y = build_training_table(horizon_hours=args.horizon, features=feats,
                                train_bounds=tuple(cfg["split"]["train"]),
                                target=args.target)
    masks = time_split(X.index, cfg)
    parts = {k: (X[m.to_numpy()], y[m.to_numpy()]) for k, m in masks.items()}
    for name, (xx, _) in parts.items():
        print(f"      {name}: {len(xx)} анализов")

    print("[3/4] обучение CatBoost (CPU)…")
    model = SulfurModel(horizon_hours=args.horizon, iterations=args.iterations,
                        monotone=not args.no_monotone, limit=limit,
                        target=args.target, seed=args.seed)
    model.fit(*parts["train"], *parts["val"], top_features=args.top_features or None,
              must_keep=CONTROL_COLUMNS)
    print(f"      признаков после отбора: {len(model.features)}; "
          f"управляющих среди них: {len(controllable_features(model, CONTROL_TAGS))}")

    scale = model.calibrate(*parts["val"])
    print(f"      конформная поправка σ: ×{scale:.2f} "
          f"(калибровка по val, честное покрытие смотрим на test)")

    # Порядок важен: сначала выбираем источник вероятности, потом калибруем ЕГО.
    # В обратном порядке поправка настраивалась на классификатор, а решения
    # принимались по интервалу, и она не применялась вообще.
    src = model.select_risk_source(*parts["val"])
    print(f"      источник вероятности: {src}; PR-AUC "
          f"{ {k: round(v, 3) for k, v in getattr(model, 'risk_source_scores', {}).items()} }, "
          f"разброс { {k: round(v, 3) for k, v in getattr(model, 'risk_source_spreads', {}).items()} }")
    train_rate = float((parts["train"][1] > limit).mean())
    a, b = model.calibrate_risk(*parts["val"], train_base_rate=train_rate)
    if model.risk_calibration_note:
        print(f"      калибровка вероятности: {model.risk_calibration_note}")
    else:
        print(f"      калибровка вероятности (Платт) для «{src}»: a={a:.2f}, b={b:.2f}")
    thr = model.select_alarm_threshold(*parts["val"],
                                      budget=cfg["quality"].get("alarm_budget"))
    budget = cfg["quality"].get("alarm_budget")
    print(f"      порог тревоги {thr:.4f} по бюджету тревог {budget:.0%} "
          f"(компромисс F1.5 дал бы {model.alarm_threshold_fbeta:.4f})")
    if model.alarm_reliable:
        pass
    else:
        print("      ВНИМАНИЕ: на валидации ни один порог не даёт тревогу информативнее "
              "базовой частоты нарушений.")
        print(f"      Переходим на прозрачное правило «прогноз выше предела» "
              f"(источник риска: {model.risk_source}, порог {thr:.2f}).")

    print("[4/4] оценка…")
    report = {"horizon_hours": args.horizon, "target": args.target,
              "limit": limit, "n_features": len(model.features),
              "monotone": model.monotone,
              # Версия схемы признаков — чтобы по отчёту было видно, на какой
              # матрице он получен. Без неё устаревший отчёт неотличим от
              # свежего, и «воспроизводится командой из README» остаётся
              # обещанием: матрицу меняли, а числа лежат прежние.
              "feature_version": FEATURE_VERSION,
              "split": {k: list(v) for k, v in cfg["split"].items()
                        if isinstance(v, (list, tuple))},
              "sigma_scale": model.sigma_scale, "alarm_threshold": model.alarm_threshold,
              "alarm_threshold_fbeta": model.alarm_threshold_fbeta,
              "alarm_budget": cfg["quality"].get("alarm_budget"),
              "splits": {}}
    for name in ("train", "val", "test"):
        Xp, yp = parts[name]
        if not len(yp):
            continue
        block = {"model": model.evaluate(Xp, yp, limit)}
        for bname, bpred in persistence_baselines(Xp, yp, args.target).items():
            if bpred.notna().any():
                block[f"baseline_{bname}"] = baseline_metrics(bpred, yp, limit)
        report["splits"][name] = block

    model.metrics = report
    # Абляционный прогон не должен затирать рабочую модель и рабочий отчёт: их
    # числа попадают в документацию, и подмена «модели без справочника» на месте
    # рабочей осталась бы незамеченной.
    suffix = args.tag or ("_novak" if args.no_vak else "")
    # Сид — в имя сам, без надежды, что о нём вспомнят в --tag. У сети проверка
    # сидов однажды затёрла основную модель, и прогон по тестовому периоду молча
    # считался на сидовом варианте (docs/GPU_MODELS.md §1.1).
    if args.seed != 42:
        suffix += f"_s{args.seed}"
    base = model.default_path(args.horizon, args.target)
    path = model.save(base.with_name(base.name + suffix) if suffix else None)
    stem = "" if args.target == "sulfur" else f"_{args.target}"
    out = ROOT / "reports" / f"quality_metrics{stem}_h{args.horizon:g}{suffix}.json"
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
                         f">{limit:g}": m.get("n_over_limit"),
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
