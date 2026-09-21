"""Чем оценивать Т95: прошлым анализом, формулой ВАК или моделью. Только CPU.

    python scripts/check_t95_source.py

Зачем. Т95 — второе жёсткое ограничение, и сейчас его уровень берётся из
лаборатории, а приращение от смены уставок — по формуле ВАК. При этом обученная
модель Т95 даёт MAE около 5.7 °C, а типичный запас до предела 360 °C — три-пять
градусов. То есть оценка грубее запаса, и от того, чем именно оценивать, зависит,
запретит ли оптимизатор вариант.

Сравниваем три оценки на одних и тех же лабораторных анализах:

* **прошлый анализ** — то, что система берёт как уровень (персистенция);
* **формула ВАК** `24-2000:GODT:T95` — официальная запись 15.09;
* **модель** `models/t95_h0` — обученный бустинг.

**Правило записано ДО счёта.** Выбор — на ВАЛИДАЦИИ, по двум условиям вместе:
MAE меньше, чем у нынешней оценки (персистенции), И доля превышений предела,
которые оценка тоже показывает выше предела, не падает. Не выполняется — остаётся
нынешняя, и это записывается как измеренный отказ. Тест считается для отчёта.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix, build_training_table  # noqa: E402
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "t95_source.json"


def metrics(pred: np.ndarray, truth: np.ndarray, limit: float) -> dict:
    good = ~np.isnan(pred) & ~np.isnan(truth)
    pred, truth = pred[good], truth[good]
    if not len(truth):
        return {}
    over, flagged = truth > limit, pred > limit
    return {
        "проб": int(len(truth)),
        "MAE": round(float(np.mean(np.abs(pred - truth))), 2),
        "смещение": round(float(np.mean(pred - truth)), 2),
        "corr": round(float(np.corrcoef(pred, truth)[0, 1]), 3) if len(truth) > 2 else None,
        "превышений в лаборатории": int(over.sum()),
        "полнота превышения": (round(float(flagged[over].mean()), 3) if over.any() else None),
        "ложных превышений": (round(float(flagged[~over].mean()), 3)
                              if (~over).any() else None),
    }


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = float(cfg["spec"]["t95_c"]["max"])
    features = build_feature_matrix()
    X, y = build_training_table(horizon_hours=0.0, features=features,
                                train_bounds=tuple(cfg["split"]["train"]), target="t95")
    masks = time_split(X.index, cfg)
    model = SulfurModel.load(SulfurModel.default_path(0.0, "t95"))

    out: dict = {}
    for split in ("val", "test"):
        mask = masks[split].to_numpy()
        Xs, ys = X[mask], y[mask]
        if not len(ys):
            continue
        truth = ys.to_numpy()
        estimates = {
            "прошлый анализ": ys.shift(1).to_numpy(),
            "формула ВАК": (Xs["vak_24_2000_GODT_T95"].to_numpy()
                            if "vak_24_2000_GODT_T95" in Xs else np.full(len(ys), np.nan)),
            "модель": model.predict_frame(Xs)["q50"].to_numpy(),
        }
        out[split] = {name: metrics(values, truth, limit)
                      for name, values in estimates.items()}
        print(f"\n{split}: анализов {len(ys)}, предел {limit:g} °C")
        for name, row in out[split].items():
            print(f"  {name:16s} {row}")

    val = out.get("val", {})
    base = val.get("прошлый анализ") or {}
    rule = {}
    for name, row in val.items():
        if name == "прошлый анализ" or not row or not base:
            continue
        rule[name] = {
            "MAE меньше, чем у прошлого анализа": bool(row["MAE"] < base["MAE"]),
            "полнота превышения не падает": bool(
                (row["полнота превышения"] or 0) >= (base["полнота превышения"] or 0)),
        }
    winners = [name for name, checks in rule.items() if all(checks.values())]
    choice = (min(winners, key=lambda n: val[n]["MAE"]) if winners else "прошлый анализ")
    print("\nПравило приёмки (валидация):")
    for name, checks in rule.items():
        print(f"  {name}: {checks}")
    print(f"  ВЫБОР оценки Т95: {choice}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "предел": limit,
                                  "оценки": out, "правило": rule, "выбор": choice},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
