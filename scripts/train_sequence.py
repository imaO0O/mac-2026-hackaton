"""Нейросетевой виртуальный анализатор серы на GPU и честное сравнение с бустингом.

    python scripts/train_sequence.py                       # GRU, горизонт 0
    python scripts/train_sequence.py --arch tcn --window 48
    python scripts/train_sequence.py --horizon 2 --cpu

Правила сравнения взяты из `docs/GPU_SETUP.md` §4 и нарушать их нельзя, иначе
результат нечем защищать:

* то же разбиение по времени с эмбарго 48 часов, что у бустинга;
* те же базовые: показание ПАК и предыдущий лабораторный анализ;
* та же конформная калибровка интервала и тот же способ выбрать порог тревоги;
* сиды зафиксированы, отчёт пишется в `reports/`.

Скрипт НЕ подменяет модель в системе автоматически. Он печатает вердикт — побила
сеть бустинг на валидации или нет — и сохраняет модель рядом. Подключение к циклу
делается флагом `python scripts/run_cycle.py --model seq`.
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
from nefte.models.quality_model import baseline_metrics  # noqa: E402
from nefte.models.sequence import SulfurSequenceModel, torch_device  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402


def boosting_reference(horizon: float) -> dict | None:
    """Метрики бустинга того же горизонта — с чем сравниваемся."""
    path = ROOT / "reports" / f"quality_metrics_h{horizon:g}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("splits")


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=0.0, help="горизонт прогноза, часов")
    ap.add_argument("--arch", choices=("gru", "tcn"), default="gru")
    ap.add_argument("--window", type=int, default=24, help="длина окна, часов")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3, help="сколько сидов усреднять")
    ap.add_argument("--cpu", action="store_true", help="учить на CPU, даже если есть GPU")
    ap.add_argument("--pretrain", action="store_true",
                    help="предобучить на показаниях поточного анализатора "
                         "(десятки тысяч меток вместо 983)")
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    args = ap.parse_args()

    cfg = load_config()
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]
    device = torch_device(not args.cpu)
    print(f"[устройство] {device}")
    if device == "cuda":
        import torch
        print(f"             {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    print("[1/5] матрица признаков…")
    feats = build_feature_matrix()
    print(f"      {feats.shape[0]} моментов × {feats.shape[1]} признаков")

    print(f"[2/5] обучающая таблица, горизонт {args.horizon} ч…")
    X, y = build_training_table(horizon_hours=args.horizon, features=feats)
    masks = time_split(X.index, cfg)
    parts = {k: (X[m.to_numpy()], y[m.to_numpy()]) for k, m in masks.items()}
    for name, (xx, _) in parts.items():
        print(f"      {name}: {len(xx)} анализов")

    print(f"[3/5] обучение {args.arch.upper()} на {device}…")
    model = SulfurSequenceModel(
        horizon_hours=args.horizon, arch=args.arch, window=args.window,
        hidden=args.hidden, epochs=args.epochs, limit=limit,
        seeds=tuple(42 + i for i in range(args.seeds)),
    )
    model.fit(feats, parts["train"][1], parts["val"][1], prefer_gpu=not args.cpu,
              pretrain_bounds=tuple(cfg["split"]["train"]) if args.pretrain else None,
              pretrain_epochs=args.pretrain_epochs)
    if args.pretrain:
        print(f"      предобучение на «{ '/'.join(cfg['split']['train']) }», "
              f"канал поточного анализатора из входа убран")
    model.attach(feats)
    for item in model.history:
        pre = (f"предобучение {item['pretrain_epochs']} эпох на "
               f"{item['pretrain_n']} окнах, " if "pretrain_epochs" in item else "")
        print(f"      сид {item['seed']}: {pre}{item['epochs']} эпох дообучения, "
              f"валидационная квантильная потеря {item['val_pinball']:.4f}")

    scale = model.calibrate(feats, parts["val"][1])
    print(f"      конформная поправка σ: ×{scale:.2f}")
    thr = model.select_alarm_threshold(feats, parts["val"][1])
    print(f"      порог тревоги: {thr:.2f}"
          + ("" if model.alarm_reliable else "  (информативной тревоги на val нет)"))

    print("[4/5] оценка…")
    report = {
        "horizon_hours": args.horizon, "arch": args.arch, "window": args.window,
        "hidden": args.hidden, "seeds": list(model.seeds), "device": device,
        "pretrained": model.pretrained,
        "channels": model.channels, "sigma_scale": model.sigma_scale,
        "alarm_threshold": model.alarm_threshold, "history": model.history,
        "splits": {},
    }
    for name in ("train", "val", "test"):
        Xp, yp = parts[name]
        if not len(yp):
            continue
        block = {"model": model.evaluate(feats, yp, limit)}
        for bname, bpred in persistence_baselines(Xp, yp).items():
            if bpred.notna().any():
                block[f"baseline_{bname}"] = baseline_metrics(bpred, yp, limit)
        report["splits"][name] = block

    reference = boosting_reference(args.horizon)
    if reference:
        report["boosting"] = {k: v.get("model") for k, v in reference.items()}

    model.metrics = report
    path = model.save()
    tag = f"{args.arch}{args.window}" + ("_pre" if args.pretrain else "")
    out = ROOT / "reports" / f"sequence_metrics_{tag}_h{args.horizon:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nмодель: {path}\nотчёт:  {out}\n")

    rows = []
    for split, block in report["splits"].items():
        for who, m in block.items():
            def _r(key, source=m):
                value = source.get(key)
                return None if value is None else round(value, 3)
            rows.append({"split": split, "модель": f"seq_{who}" if who == "model" else who,
                         "n": m["n"], ">10": m.get("n_over_limit"),
                         "MAE": round(m["MAE"], 3), "RMSE": round(m["RMSE"], 3),
                         "bias": round(m["bias"], 3), "покрытие80": _r("coverage_80"),
                         "prec@порог": _r("spec_precision"), "rec@порог": _r("spec_recall"),
                         "ROC-AUC": _r("roc_auc")})
        if reference and split in reference:
            m = reference[split]["model"]
            rows.append({"split": split, "модель": "catboost", "n": m["n"],
                         ">10": m.get("n_over_limit"),
                         "MAE": round(m["MAE"], 3), "RMSE": round(m["RMSE"], 3),
                         "bias": round(m["bias"], 3),
                         "покрытие80": None if m.get("coverage_80") is None
                         else round(m["coverage_80"], 3),
                         "prec@порог": None if m.get("spec_precision") is None
                         else round(m["spec_precision"], 3),
                         "rec@порог": None if m.get("spec_recall") is None
                         else round(m["spec_recall"], 3),
                         "ROC-AUC": None if m.get("roc_auc") is None
                         else round(m["roc_auc"], 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- вердикт ------------------------------------------------ #
    print("\nВердикт (решается на ВАЛИДАЦИИ, тест смотрим только как проверку):")
    if not reference or "val" not in reference:
        print("  бустинг того же горизонта не обучен — сравнивать не с чем, "
              "запустите scripts/train_quality.py")
        return 0
    seq_val = report["splits"]["val"]["model"]
    cat_val = reference["val"]["model"]
    better_mae = seq_val["MAE"] < cat_val["MAE"]
    better_auc = (seq_val.get("roc_auc") or 0) > (cat_val.get("roc_auc") or 0)
    print(f"  MAE:      сеть {seq_val['MAE']:.3f} против бустинга {cat_val['MAE']:.3f}"
          f"  → {'сеть' if better_mae else 'бустинг'}")
    if seq_val.get("roc_auc") is not None and cat_val.get("roc_auc") is not None:
        print(f"  ROC-AUC:  сеть {seq_val['roc_auc']:.3f} против "
              f"бустинга {cat_val['roc_auc']:.3f}"
              f"  → {'сеть' if better_auc else 'бустинг'}")
    if better_mae and better_auc:
        print("  Сеть выигрывает по обоим показателям: имеет смысл ставить её в цикл "
              "(scripts/run_cycle.py --model seq).")
    elif better_mae or better_auc:
        print("  Ничья: выигрыш по одному показателю и проигрыш по другому. "
              "Оставляем бустинг — он проще, объясним и не требует torch в демо.")
    else:
        print("  Бустинг лучше по обоим показателям. Отрицательный результат тоже "
              "результат: на 983 анализах сети не хватает данных, и это надо сказать "
              "на защите прямо.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
