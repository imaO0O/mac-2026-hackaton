"""Модель серы против `Q21` — второго поточного анализатора серы. Только CPU.

    python scripts/check_q21_baseline.py
    python scripts/check_q21_baseline.py --horizons 0 2

Зачем. До ответа организаторов 15.09 модель сравнивалась с ПАК из файла
анализаторов: MAE на тесте 1.29 против 1.48. Но в телеметрии 24-2000 есть `Q21` —
«поточный анализатор серы в г/о ДТ» (так в новой таблице тегов; в листе «КИП» пакета
описания перемешаны, и мы считали, что дубликата анализатора нет). С лабораторией
`Q21` сходится лучше файла ПАК: corr 0.49 против 0.22 на всей истории
(`docs/DATA_NOTES.md` §5б). Честный соперник модели — он.

Сравнение записано до счёта (`docs/PLAN.md`, «Ответы и материалы 15.09»): те же
пробы, что у модели в обучающем отчёте; `Q21` как-of из матрицы признаков, то есть
уже очищенный от заглушки 307; мера — MAE, смещение и ROC-AUC превышения предела
(у `Q21` вероятностью служит само показание). Выбирать нечего: если модель `Q21` не
бьёт, так и записывается.

Результат: reports/q21_baseline.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix, build_training_table  # noqa: E402
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "q21_baseline.json"
COLUMN = "ht_Q21"


def _metrics(pred: pd.Series, y: pd.Series, limit: float, score: pd.Series) -> dict:
    err = pred - y
    over = (y > limit).astype(int)
    auc = float(roc_auc_score(over, score)) if over.nunique() == 2 else None
    return {"MAE": round(float(err.abs().mean()), 3), "bias": round(float(err.mean()), 3),
            "roc_auc": None if auc is None else round(auc, 3)}


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=float, nargs="+", default=[0.0, 2.0])
    args = ap.parse_args()

    cfg = load_config()
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    features = build_feature_matrix()
    out: dict = {}
    for horizon in args.horizons:
        model = SulfurModel.load(SulfurModel.default_path(horizon))
        X, y = build_training_table(horizon_hours=horizon, features=features,
                                    train_bounds=tuple(cfg["split"]["train"]))
        masks = time_split(X.index, cfg)
        block = {}
        for split in ("val", "test"):
            mask = masks[split].to_numpy()
            Xp, yp = X[mask], y[mask]
            # только пробы, где у Q21 есть показание: иначе сравнение нечестно в
            # пользу того, кто отвечает всегда
            common = Xp.index[Xp[COLUMN].notna().to_numpy()] if COLUMN in Xp else Xp.index[:0]
            if not len(common):
                continue
            Xc, yc = Xp.loc[common], yp.loc[common]
            pred = model.predict_frame(Xc)["q50"]
            risk = model.predict_risk(Xc)
            q21 = Xc[COLUMN].astype(float)
            pak = Xc["pak_sulfur"].astype(float) if "pak_sulfur" in Xc else None
            block[split] = {
                "проб": int(len(common)),
                "проб без Q21": int(len(yp) - len(common)),
                "с превышением": int((yc > limit).sum()),
                "модель": _metrics(pred, yc, limit, risk),
                "Q21": _metrics(q21, yc, limit, q21),
            }
            if pak is not None and pak.notna().all():
                block[split]["ПАК из файла"] = _metrics(pak, yc, limit, pak)
            row = block[split]
            print(f"h={horizon:g} {split}: проб {row['проб']} (без Q21 {row['проб без Q21']}), "
                  f"с превышением {row['с превышением']}")
            for name in ("модель", "Q21", "ПАК из файла"):
                if name in row:
                    m = row[name]
                    print(f"    {name:14s} MAE {m['MAE']:.3f}  смещение {m['bias']:+.3f}  "
                          f"ROC-AUC {m['roc_auc']}")
        test = block.get("test") or {}
        if test:
            block["модель лучше Q21 по MAE на тесте"] = bool(test["модель"]["MAE"] < test["Q21"]["MAE"])
            block["модель лучше Q21 по ROC-AUC на тесте"] = bool(
                (test["модель"]["roc_auc"] or 0) > (test["Q21"]["roc_auc"] or 0))
        out[f"h{horizon:g}"] = block

    REPORT.write_text(json.dumps({**report_provenance(cfg), "предел": limit,
                                  "столбец": COLUMN, "горизонты": out},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
