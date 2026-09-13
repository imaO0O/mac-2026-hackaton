"""Видит ли верхний край что-нибудь лучше нынешней вероятности превышения? Только CPU.

    python scripts/check_upper_edge_risk.py
    python scripts/check_upper_edge_risk.py models/sulfur_h0 models/sulfur_h0_s100

Модель занижает серу у предела, а вероятность превышения на тесте сжата к
середине (docs/QUALITY_AGENT.md). Поправка ПОСЛЕ модели не прошла ни по
валидации, ни по обучающему периоду. Здесь проверяется другой путь — вероятность,
которая сама смотрит на верхний край:

* **эмпирический хвост** — нормированные остатки валидации вместо нормального
  закона, по верхней полуширине интервала;
* **классификатор** превышения, который модель и так обучает (`risk.cbm`);
* **смесь** интервала и классификатора — средним и по рангам;
* **хвост по квантилям** — отдельные квантильные модели q95 и q97.5 на уже
  отобранных признаках, уровни квантилей откалиброваны по валидации.

Правило принятия записано ДО счёта: вариант принимается, только если на
ВАЛИДАЦИИ при бюджете тревог полнота не хуже нынешней И наклон калибровки не
дальше от единицы. Тест — только проверка. Считается на каждой модели из
аргументов (по умолчанию — рабочая и два сидовых варианта), потому что решение на
43 превышениях валидации шумное, и один сид может принять то, что два других
отклоняют.

Результат: таблица в консоли и reports/upper_edge_risk.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catboost import CatBoostRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix, build_training_table  # noqa: E402
from nefte.models.quality_model import (  # noqa: E402
    Z90, SulfurModel, _normal_cdf, calibration_slope)
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

DEFAULT_MODELS = ("models/sulfur_h0", "models/sulfur_h0_s100", "models/sulfur_h0_s200")
REPORT = ROOT / "reports" / "upper_edge_risk.json"


def to_rank(reference: pd.Series, values: pd.Series) -> pd.Series:
    """Ранг по шкале валидации: тест переводится в неё той же лестницей."""
    ladder = np.sort(reference.to_numpy())
    return pd.Series(np.searchsorted(ladder, values.to_numpy(), side="right") / len(ladder),
                     index=values.index)


def quantile_tail_risk(p: pd.DataFrame, levels: dict[str, float], limit: float) -> pd.Series:
    """P(y > предела) по лестнице квантилей q50/q90/q95/q97.5 с уровнями валидации."""
    xs = p[["q50", "q90", "q95", "q975"]].to_numpy()
    ps = np.array([levels["q50"], levels["q90"], levels["q95"], levels["q975"]])
    out = np.empty(len(p))
    for i, x in enumerate(xs):
        if limit >= x[-1]:
            scale = (x[3] - x[2]) / max(np.log((1 - ps[2]) / max(1 - ps[3], 1e-4)), 1e-3)
            out[i] = (1 - ps[3]) * np.exp(-(limit - x[3]) / max(scale, 1e-3))
        elif limit <= x[0]:
            sigma = max((p["q50"].iloc[i] - p["q10"].iloc[i]) / Z90, 0.05)
            out[i] = 1 - _normal_cdf((limit - x[0]) / sigma)
        else:
            out[i] = 1 - np.interp(limit, x, ps)
    return pd.Series(out, index=p.index)


def variants_for(model: SulfurModel, X: pd.DataFrame, y: pd.Series,
                 masks: dict) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    tr, va, te = (masks[k].to_numpy() for k in ("train", "val", "test"))
    F, limit = model.features, model.limit
    pv, pt = model.predict_frame(X[va]), model.predict_frame(X[te])
    yv = y[va]

    def interval(p):
        return pd.Series(1 - _normal_cdf((limit - p["q50"]) / p["sigma"]), index=p.index)

    up_v = ((pv["q90"] - pv["q50"]) / Z90).clip(lower=0.05)
    ladder = np.sort(((yv - pv["q50"]) / up_v).to_numpy())

    def empirical(p):
        up = ((p["q90"] - p["q50"]) / Z90).clip(lower=0.05)
        t = ((limit - p["q50"]) / up).to_numpy()
        return pd.Series(1 - np.searchsorted(ladder, t, side="right") / len(ladder),
                         index=p.index)

    rv = {"текущий": interval(pv), "эмпирический хвост": empirical(pv)}
    rt = {"текущий": interval(pt), "эмпирический хвост": empirical(pt)}
    if model.clf is not None:
        cv = pd.Series(model.clf.predict_proba(X[va][F])[:, 1], index=pv.index)
        ct = pd.Series(model.clf.predict_proba(X[te][F])[:, 1], index=pt.index)
        rv["классификатор"], rt["классификатор"] = cv, ct
        rv["смесь средним"] = (rv["текущий"] + cv) / 2
        rt["смесь средним"] = (rt["текущий"] + ct) / 2
        rv["смесь рангов"] = (to_rank(rv["текущий"], rv["текущий"]) + to_rank(cv, cv)) / 2
        rt["смесь рангов"] = (to_rank(rv["текущий"], rt["текущий"]) + to_rank(cv, ct)) / 2

    extra = {}
    for name, alpha in (("q95", 0.95), ("q975", 0.975)):
        reg = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}",
                                iterations=model.iterations, learning_rate=model.learning_rate,
                                depth=model.depth, random_seed=model.seed, verbose=False,
                                allow_writing_files=False, task_type="CPU",
                                monotone_constraints=model._monotone_constraints())
        reg.fit(X[tr][F], y[tr] - model.y_offset,
                eval_set=(X[va][F], yv - model.y_offset), use_best_model=True)
        extra[name] = reg
    for p, mask in ((pv, va), (pt, te)):
        p["q95"] = np.maximum(extra["q95"].predict(X[mask][F]) + model.y_offset, p["q90"] + 1e-3)
        p["q975"] = np.maximum(extra["q975"].predict(X[mask][F]) + model.y_offset,
                               p["q95"] + 1e-3)
    levels = {k: float((yv <= pv[k]).mean()) for k in ("q50", "q90", "q95", "q975")}
    rv["хвост по квантилям"] = quantile_tail_risk(pv, levels, limit)
    rt["хвост по квантилям"] = quantile_tail_risk(pt, levels, limit)
    return rv, rt


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    budget = float(cfg["quality"].get("alarm_budget", 0.3))
    paths = sys.argv[1:] or list(DEFAULT_MODELS)
    X, y = build_training_table(horizon_hours=0.0, features=build_feature_matrix(),
                                train_bounds=tuple(cfg["split"]["train"]))
    masks = time_split(X.index, cfg)
    yv, yt = y[masks["val"].to_numpy()], y[masks["test"].to_numpy()]

    rows = []
    for path in paths:
        model = SulfurModel.load(ROOT / path)
        rv, rt = variants_for(model, X, y, masks)
        ov, ot = (yv > model.limit).astype(int), (yt > model.limit).astype(int)
        base = None
        for name in rv:
            threshold = float(np.quantile(rv[name], 1 - budget))
            sv = calibration_slope(rv[name], ov, n_boot=500)
            st = calibration_slope(rt[name], ot, n_boot=500)
            row = {
                "модель": Path(path).name, "вариант": name,
                "val_полнота": round(float((rv[name] >= threshold)[ov.astype(bool)].mean()), 3),
                "val_наклон": round(sv["b"], 3),
                "val_roc_auc": round(float(roc_auc_score(ov, rv[name])), 3),
                "test_доля_тревог": round(float((rt[name] >= threshold).mean()), 3),
                "test_полнота": round(float((rt[name] >= threshold)[ot.astype(bool)].mean()), 3),
                "test_наклон": [round(st["b"], 3), round(st["b_от"], 3), round(st["b_до"], 3)],
                "test_roc_auc": round(float(roc_auc_score(ot, rt[name])), 3),
            }
            if name == "текущий":
                base = row
                row["вывод"] = "база"
            else:
                accepted = (row["val_полнота"] >= base["val_полнота"]
                            and abs(row["val_наклон"] - 1) <= abs(base["val_наклон"] - 1))
                row["вывод"] = "принят" if accepted else "отклонён"
            rows.append(row)

    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(frame.to_string(index=False))

    print("\nВывод по вариантам (правило на валидации, тест — проверка):")
    summary = {}
    for name, g in frame[frame["вариант"] != "текущий"].groupby("вариант", sort=False):
        accepted = int((g["вывод"] == "принят").sum())
        base = frame[frame["вариант"] == "текущий"].set_index("модель")
        worse_test = int(sum(r["test_полнота"] < base.loc[r["модель"], "test_полнота"]
                             or abs(r["test_наклон"][0] - 1)
                             > abs(base.loc[r["модель"], "test_наклон"][0] - 1)
                             for _, r in g.iterrows()))
        summary[name] = {"принят_на_сидах": accepted, "сидов": len(g),
                         "на_тесте_хуже_по_полноте_или_форме": worse_test}
        print(f"  {name:20s} принят на {accepted} из {len(g)}; на тесте хуже текущего "
              f"по полноте или форме на {worse_test} из {len(g)}")
    adopted = [k for k, v in summary.items() if v["принят_на_сидах"] * 2 > v["сидов"]]
    print("  ПРИНЯТ ВАРИАНТ: " + (", ".join(adopted) if adopted else
                                 "ни один — нынешняя вероятность остаётся"))

    REPORT.write_text(json.dumps({**report_provenance(cfg), "бюджет_тревог": budget,
                                  "модели": paths, "строки": rows, "итог": summary,
                                  "принят": adopted}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
