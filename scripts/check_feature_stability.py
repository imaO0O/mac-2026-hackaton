"""Насколько устойчив отбор признаков — и можно ли верить выводам из него.

    python scripts/check_feature_stability.py
    python scripts/check_feature_stability.py --horizon 2 --seeds 5

Отбор признаков идёт в два прохода: первый меряет важность, второй учится на
сокращённом наборе. Оба прохода используют один сид, и до сих пор его никто не
менял. Между тем на 983 обучающих строках и 385 признаках бустинг отбирает
неустойчиво — и это важно не само по себе, а потому что **из состава отобранных
признаков мы делаем выводы**:

* «из 13 формул справочника в модель попадает одна»;
* «нормированная температура вытеснила reg_wabt_dev30»;
* «признак плотности стоит на 16-м месте по важности».

Если при другом сиде отбирается другой набор, все три утверждения превращаются в
«так выпало», и говорить их на защите нельзя.

Что меряем
----------
* **ядро** — признаки, отобранные при ВСЕХ сидах. Только про них можно говорить
  «модель на них опирается»;
* **случайные** — отобранные ровно при одном сиде. Их присутствие в конкретной
  сборке ничего не означает;
* **сходство наборов** (Жаккар) — доля общего между любыми двумя прогонами;
* **разброс метрик** — насколько гуляет само качество. Если оно устойчиво, а
  состав признаков нет, значит признаки взаимозаменяемы, и это отдельный вывод.

Результат: reports/feature_stability_h<горизонт>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import (  # noqa: E402
    QUALITY_TARGETS,
    build_feature_matrix,
    build_training_table,
)
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_quality import CONTROL_COLUMNS  # noqa: E402


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=0.0)
    ap.add_argument("--target", default="sulfur", choices=sorted(QUALITY_TARGETS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--top-features", type=int, default=40)
    ap.add_argument("--probe-seeds", type=int, default=None,
                    help="сколько пробных моделей усреднять при отборе; 1 — как было "
                         "до усреднения, нужно чтобы померить эффект самой правки")
    args = ap.parse_args()

    cfg = load_config()
    node = cfg
    for key in QUALITY_TARGETS[args.target]["limit_key"]:
        node = node[key]
    limit = float(node)

    X, y = build_training_table(horizon_hours=args.horizon,
                                features=build_feature_matrix(),
                                train_bounds=tuple(cfg["split"]["train"]),
                                target=args.target)
    masks = time_split(X.index, cfg)
    parts = {k: (X[m.to_numpy()], y[m.to_numpy()]) for k, m in masks.items()}

    selections: dict[int, list[str]] = {}
    metrics: list[dict] = []
    for i in range(args.seeds):
        seed = 42 + i
        model = SulfurModel(horizon_hours=args.horizon, seed=seed, limit=limit,
                            target=args.target)
        extra = ({} if args.probe_seeds is None
                 else {"probe_seeds": args.probe_seeds})
        model.fit(*parts["train"], *parts["val"],
                  top_features=args.top_features, must_keep=CONTROL_COLUMNS, **extra)
        model.calibrate(*parts["val"])
        model.select_risk_source(*parts["val"])
        model.select_alarm_threshold(*parts["val"],
                                     budget=cfg["quality"].get("alarm_budget"))
        selections[seed] = list(model.features)
        block = model.evaluate(*parts["test"], limit)
        metrics.append({"сид": seed, "признаков": len(model.features),
                        "MAE": round(block["MAE"], 3),
                        "ROC-AUC": None if block.get("roc_auc") is None
                        else round(block["roc_auc"], 3),
                        "полнота": round(block.get("spec_recall", float("nan")), 3)})
        print(f"  сид {seed}: {len(model.features)} признаков, "
              f"test MAE {block['MAE']:.3f}")

    sets = {s: set(v) for s, v in selections.items()}
    core = set.intersection(*sets.values())
    union = set.union(*sets.values())
    counts = pd.Series({f: sum(f in v for v in sets.values()) for f in union})
    once = sorted(counts[counts == 1].index)

    jaccard = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
               for a, b in combinations(sets, 2)]

    print(f"\nПо {args.seeds} сидам, горизонт {args.horizon:g} ч\n")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nСостав признаков:")
    print(f"   объединение по всем сидам: {len(union)}")
    print(f"   ЯДРО (во всех сидах):      {len(core)}")
    print(f"   отобраны ровно раз:        {len(once)}")
    print(f"   сходство наборов (Жаккар): медиана {np.median(jaccard):.2f}, "
          f"от {min(jaccard):.2f} до {max(jaccard):.2f}")

    # Отдельно — судьба признаков, из состава которых мы делали выводы.
    watched = [f for f in union
               if f.startswith("vak_") or f.startswith("reg_")
               or f in ("lims_sulfur_prev", "pak_sulfur")]
    if watched:
        print("\nПризнаки, про которые в документации сказано отдельно:")
        table = pd.DataFrame({
            "признак": watched,
            "в скольких сидах": [int(counts[f]) for f in watched],
        }).sort_values("в скольких сидах", ascending=False)
        print(table.to_string(index=False))

    mae = [m["MAE"] for m in metrics]
    auc = [m["ROC-AUC"] for m in metrics if m["ROC-AUC"] is not None]
    print("\nВывод.")
    print(f"  Качество: MAE от {min(mae):.3f} до {max(mae):.3f} "
          f"(разброс {max(mae) - min(mae):.3f})"
          + (f", ROC-AUC от {min(auc):.3f} до {max(auc):.3f}" if auc else "") + ".")
    share = len(core) / len(union) if union else 0.0
    if share < 0.5:
        print(f"  Состав признаков НЕУСТОЙЧИВ: общее ядро — лишь {share:.0%} от "
              f"объединения. Утверждения вида «признак X попал в модель» описывают "
              f"конкретную сборку, а не свойство данных, и на защите их надо "
              f"произносить именно так.")
    else:
        print(f"  Состав устойчив: ядро — {share:.0%} от объединения.")
    if len(mae) > 1 and (max(mae) - min(mae)) < 0.1 and share < 0.5:
        print("  При этом качество почти не меняется. Значит, признаки во многом "
              "ВЗАИМОЗАМЕНЯЕМЫ: они несут общую информацию о состоянии установки, "
              "и какой именно из эквивалентных попадёт в набор — дело случая.")

    tag = "" if args.probe_seeds is None else f"_probe{args.probe_seeds}"
    out = ROOT / "reports" / f"feature_stability_h{args.horizon:g}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        **report_provenance(cfg),
        "горизонт": args.horizon, "показатель": args.target, "сидов": args.seeds,
        "метрики": metrics,
        "ядро": sorted(core), "отобраны_однажды": once,
        "частота": {f: int(counts[f]) for f in sorted(union)},
        "жаккар": {"медиана": float(np.median(jaccard)),
                   "мин": float(min(jaccard)), "макс": float(max(jaccard))},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
