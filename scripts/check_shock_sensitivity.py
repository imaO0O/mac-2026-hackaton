"""Насколько прогноз сдвигается от каждого возмущения. Только CPU.

    python scripts/check_shock_sensitivity.py

Зачем. Сценарии (`scripts/run_scenario.py`) показывают, изменились ли РЕШЕНИЯ. Но
когда решения не меняются, надо честно сказать почему: возмущение двигает прогноз
слишком слабо, чтобы перейти порог вмешательства. Здесь считается сам сдвиг — на
всех моментах теста, по той же модели, что принимает решения.

Мера сравнения одна и та же для всех возмущений: сдвиг медианного прогноза серы в
мг/кг и доля моментов, где риск перешёл рабочий порог тревоги. Второе и есть
«изменились бы решения», только без многочасового прогона.
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
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "shock_sensitivity.json"

# Возмущения ровно те же, что в scripts/run_scenario.py, плюс усиленные варианты:
# надо понять не только «меняется ли», но и «сколько нужно, чтобы поменялось».
SHOCKS = [
    ("сера сырья +3 мг/кг", ("lims_feed_sulfur",), +3.0),
    ("сера сырья +10 мг/кг", ("lims_feed_sulfur",), +10.0),
    ("утяжеление сырья +5 °C", ("avt_T66",), +5.0),
    ("утяжеление сырья +15 °C", ("avt_T66",), +15.0),
    ("температура реактора −4 °C", ("reg_wabt", "ht_T5", "ht_T11"), -4.0),
    ("температура реактора −10 °C", ("reg_wabt", "ht_T5", "ht_T11"), -10.0),
    ("показание анализатора +5 мг/кг", ("ht_Q21",), +5.0),
]


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    model = SulfurModel.load(SulfurModel.default_path(0.0, "sulfur"))
    features = build_feature_matrix()
    masks = time_split(features.index, cfg)
    X = features[masks["test"].to_numpy()][model.features].dropna()
    base_pred = model.predict_frame(X)["q50"]
    base_risk = model.predict_risk(X)
    threshold = float(model.alarm_threshold)
    print(f"тест: строк {len(X)}, медиана прогноза {base_pred.median():.2f} мг/кг, "
          f"порог тревоги {threshold:.2f}, доля тревог {float((base_risk > threshold).mean()):.1%}")

    rows = []
    for name, prefixes, step in SHOCKS:
        shifted = X.copy()
        touched = [c for c in X.columns if c.startswith(prefixes)
                   and not c.endswith(("_slope7", "_dev30", "_std6", "_std36", "_std144"))]
        if not touched:
            continue
        shifted[touched] = shifted[touched] + step
        pred = model.predict_frame(shifted)["q50"]
        risk = model.predict_risk(shifted)
        rows.append({
            "возмущение": name,
            "каналов": len(touched),
            "сдвиг прогноза, мг/кг": round(float((pred - base_pred).median()), 3),
            "доля тревог: было": round(float((base_risk > threshold).mean()), 3),
            "доля тревог: стало": round(float((risk > threshold).mean()), 3),
            "моментов сменили сторону порога": round(
                float(((risk > threshold) != (base_risk > threshold)).mean()), 3),
        })
        print(f"  {rows[-1]}")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "порог тревоги": threshold,
                                  "медиана прогноза, мг/кг": round(float(base_pred.median()), 2),
                                  "возмущения": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
