"""Q21 как отдельный источник вероятности риска — по правилу, записанному до счёта.

    python scripts/check_q21_source.py

Участник 2. `Q21` («поточный анализатор серы в г/о ДТ») ранжирует превышения на
тесте лучше модели (ROC-AUC 0.841 против 0.803, `scripts/check_q21_baseline.py`),
хотя по средней ошибке модель точнее. Вопрос участника 1: даёт ли `Q21` что-то как
ОТДЕЛЬНЫЙ источник вероятности нарушения — наравне с интервалом и классификатором.

Правило (docs/PLAN.md, записано до счёта): показание `Q21` как-of из матрицы
признаков переводится в вероятность логистической калибровкой по log(`Q21`) на
обучающем периоде и становится источником, только если на ВАЛИДАЦИИ одновременно:

1. на пробах с показанием `Q21` его PR-AUC не меньше PR-AUC текущего источника
   модели плюс 0.05 (``MIN_SOURCE_MARGIN``, тот же запас, что у классификатора);
2. разброс вероятности p05–p95 не меньше 0.05;
3. показание есть не меньше чем у 90 % проб валидации.

Тест в правиле не участвует — описательно.

Результат: reports/q21_source.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix, build_training_table  # noqa: E402
from nefte.models.quality_model import MIN_SOURCE_MARGIN, SulfurModel, interval_risk  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

COLUMN = "ht_Q21"
MIN_SPREAD = 0.05
MIN_COVERAGE = 0.90
REPORT = ROOT / "reports" / "q21_source.json"


def _log(q: pd.Series) -> np.ndarray:
    return np.log(q.clip(lower=0.1).to_numpy()).reshape(-1, 1)


def scores(over: pd.Series, risk: pd.Series) -> dict:
    if over.nunique() < 2:
        return {"pr_auc": None, "roc_auc": None}
    return {"pr_auc": round(float(average_precision_score(over, risk)), 3),
            "roc_auc": round(float(roc_auc_score(over, risk)), 3)}


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    model = SulfurModel.load(SulfurModel.default_path(0))
    X, y = build_training_table(horizon_hours=0, features=build_feature_matrix(),
                                train_bounds=tuple(cfg["split"]["train"]))
    masks = time_split(X.index, cfg)

    train = masks["train"].to_numpy() & X[COLUMN].notna().to_numpy()
    calib = LogisticRegression(C=1e6, max_iter=1000)
    calib.fit(_log(X.loc[train, COLUMN]), (y[train] > limit).astype(int))

    out: dict = {"источник модели": model.risk_source,
                 "калибровка Q21": {"a": round(float(calib.coef_[0][0]), 4),
                                    "b": round(float(calib.intercept_[0]), 4)}}
    for split in ("val", "test"):
        mask = masks[split].to_numpy()
        Xs, ys = X[mask], y[mask]
        has = Xs[COLUMN].notna()
        Xc, yc = Xs[has], ys[has]
        over = (yc > limit).astype(int)
        q21 = pd.Series(calib.predict_proba(_log(Xc[COLUMN]))[:, 1], index=Xc.index)
        block = {"проб": int(len(Xs)), "с Q21": int(has.sum()),
                 "покрытие": round(float(has.mean()), 3), "с превышением": int(over.sum()),
                 "источник модели": scores(over, model.predict_risk(Xc)),
                 "интервал": scores(over, interval_risk(model.predict_frame(Xc), limit)),
                 "Q21": scores(over, q21),
                 "разброс Q21 p05–p95": round(float(q21.quantile(.95) - q21.quantile(.05)), 3)}
        out[split] = block
        print(f"{split}: проб {block['проб']}, с Q21 {block['с Q21']} ({block['покрытие']:.0%}), "
              f"с превышением {block['с превышением']}")
        for name in ("источник модели", "интервал", "Q21"):
            m = block[name]
            print(f"    {name:16s} PR-AUC {m['pr_auc']}  ROC-AUC {m['roc_auc']}")
        print(f"    разброс вероятности Q21 p05–p95 {block['разброс Q21 p05–p95']}")

    val = out["val"]
    conditions = {
        "1. PR-AUC Q21 ≥ источник модели + 0.05":
            (val["Q21"]["pr_auc"] or 0) >= (val["источник модели"]["pr_auc"] or 0)
            + MIN_SOURCE_MARGIN,
        "2. разброс вероятности ≥ 0.05": val["разброс Q21 p05–p95"] >= MIN_SPREAD,
        "3. показание у ≥ 90 % проб": val["покрытие"] >= MIN_COVERAGE,
    }
    accepted = all(conditions.values())
    print("\nПравило (валидация):")
    for name, ok in conditions.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print("\nВЫВОД: " + ("Q21 становится источником вероятности риска" if accepted else
                         "Q21 источником не становится — измеренный отказ"))
    REPORT.write_text(json.dumps({**report_provenance(cfg), "предел": limit, **out,
                                  "условия": conditions, "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
