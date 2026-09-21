"""Можно ли ловить превышения Т95 комбинацией оценок. Только CPU.

    python scripts/check_t95_combo.py

Зачем. Т95 — второе жёсткое ограничение, и оно самое слабое место решения.
`scripts/check_t95_source.py` уже показал: прошлый анализ ловит 36 % превышений,
формула ВАК и модель — ноль, MAE у всех около 5 °C при запасе до предела 3–5 °C.
Вывод был записан честно: ограничение работает как «не ухудшать», а не как
гарантия. Но из трёх плохих оценок можно собрать одну лучше, если они ошибаются
по-разному: достаточно взять НАИБОЛЬШУЮ — консервативную границу сверху.

Проверяем четыре комбинации против нынешней оценки (прошлый анализ):

* максимум из прошлого анализа и формулы ВАК;
* максимум из всех трёх;
* прошлый анализ плюс его же тренд за последние два анализа (экстраполяция);
* прошлый анализ плюс сигма возраста (верхняя граница доверия).

**Правило записано ДО счёта.** Комбинация принимается, если на ВАЛИДАЦИИ
одновременно:

1. ловит превышений СТРОГО больше, чем прошлый анализ;
2. доля ложных превышений выше не более чем на 5 процентных пунктов.

Ни одна не проходит — измеренный отказ, оценка Т95 остаётся прежней, а
формулировка ограничения («не ухудшать») подтверждается ещё раз.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.quality import t95_sigma  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix, build_training_table  # noqa: E402
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "t95_combo.json"
MAX_EXTRA_FALSE = 0.05


def metrics(pred: np.ndarray, truth: np.ndarray, limit: float) -> dict:
    good = ~np.isnan(pred) & ~np.isnan(truth)
    pred, truth = pred[good], truth[good]
    if not len(truth):
        return {}
    over, flagged = truth > limit, pred > limit
    return {
        "проб": int(len(truth)),
        "MAE": round(float(np.mean(np.abs(pred - truth))), 2),
        "превышений": int(over.sum()),
        "полнота превышения": (round(float(flagged[over].mean()), 3)
                               if over.any() else None),
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
        previous = ys.shift(1).to_numpy()
        # возраст опорного анализа: разница меток времени, в часах
        age_h = pd.Series(ys.index).diff().dt.total_seconds().to_numpy() / 3600.0
        vak = (Xs["vak_24_2000_GODT_T95"].to_numpy()
               if "vak_24_2000_GODT_T95" in Xs else np.full(len(ys), np.nan))
        boost = model.predict_frame(Xs)["q50"].to_numpy()
        trend = ys.shift(1).to_numpy() + (ys.shift(1).to_numpy() - ys.shift(2).to_numpy())
        sigma = np.array([t95_sigma(a if a == a else None) for a in age_h])

        estimates = {
            "прошлый анализ (нынешняя)": previous,
            "максимум: анализ и ВАК": np.fmax(previous, vak),
            "максимум: анализ, ВАК и модель": np.fmax(np.fmax(previous, vak), boost),
            "анализ плюс тренд": trend,
            "анализ плюс сигма возраста": previous + sigma,
        }
        out[split] = {name: metrics(values, truth, limit)
                      for name, values in estimates.items()}
        print(f"\n{split}: анализов {len(ys)}, предел {limit:g} °C")
        for name, row in out[split].items():
            print(f"  {name:30s} {row}")

    val = out.get("val", {})
    base = val.get("прошлый анализ (нынешняя)") or {}
    rule = {}
    for name, row in val.items():
        if name.startswith("прошлый анализ") or not row or not base:
            continue
        rule[name] = {
            "ловит больше превышений": bool(
                (row["полнота превышения"] or 0) > (base["полнота превышения"] or 0)),
            "ложных не больше чем на 5 п.п.": bool(
                (row["ложных превышений"] or 0)
                <= (base["ложных превышений"] or 0) + MAX_EXTRA_FALSE),
        }
    winners = [name for name, checks in rule.items() if all(checks.values())]
    choice = (min(winners, key=lambda n: val[n]["MAE"]) if winners
              else "прошлый анализ (нынешняя)")
    print("\nПравило приёмки (валидация):")
    for name, checks in rule.items():
        print(f"  {name}: {checks}")
    print(f"  ВЫБОР оценки Т95: {choice}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "предел": limit,
                                  "условие: ложных не больше чем на": MAX_EXTRA_FALSE,
                                  "оценки": out, "правило": rule, "выбор": choice},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
