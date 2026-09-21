"""Раннее предупреждение сетью на два часа: поднимать внимание, не трогая уставки.

    python scripts/check_early_warning.py

Зачем. Главный пробел решения назван в README: система реагирует на идущее
превышение, а не предупреждает. У бустинга на горизонте 2 ч сигнала нет (ROC-AUC на
тесте ниже случайного), у сети на том же горизонте он есть и держится по тройкам
сидов (`docs/GPU_MODELS.md`). Значит, сеть можно поставить ВТОРЫМ каналом: она
считает вероятность превышения через два часа и поднимает внимание оператора, а
решение по уставкам остаётся за рабочей моделью. Уставки этот канал не двигает
вовсе — ошибиться он может только лишней тревогой.

**Правило приёмки записано ДО счёта** (`docs/PLAN.md`, план доработки 20.09).
Считаем на ВАЛИДАЦИИ, по лабораторным пробам:

1. порог канала выбирается как наименьший, при котором доля часов с поднятым
   вниманием не превышает бюджет тревог из конфига (`quality.alarm_budget`);
2. канал принимается, если при этом пороге доля проб с превышением, перед которыми
   внимание поднято в окне −2 … 0 ч, выше, чем у рабочей модели, не меньше чем на
   10 пунктов;
3. решения по уставкам не меняются: канал в ранжировании не участвует — это
   проверяется тем, что он вообще не входит в оптимизатор.

Не выполняется — записываем измеренный отказ, канал не включаем. Тест считается
только для отчёта.

Результат: reports/early_warning.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.models.sequence import SulfurSequenceModel  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "early_warning.json"
WINDOW_HOURS = 2.0          # окно предупреждения: −2 … 0 ч до отбора пробы
MIN_GAIN = 0.10             # на сколько канал обязан поднять полноту предупреждения
GRID = "1h"


def warned(risk: pd.Series, samples: pd.DatetimeIndex, threshold: float) -> np.ndarray:
    """Было ли внимание поднято в окне −2 … 0 ч перед каждой пробой."""
    flags = []
    for ts in samples:
        window = risk.loc[ts - pd.Timedelta(hours=WINDOW_HOURS):ts]
        flags.append(bool((window > threshold).any()))
    return np.array(flags)


def share_of_hours(risk: pd.Series, threshold: float) -> float:
    return float((risk > threshold).mean())


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    budget = float(cfg["quality"]["alarm_budget"])
    features = build_feature_matrix()
    sb = StateBuilder(cfg)

    seq_path = SulfurSequenceModel.default_path(2.0, "tcn", 48, True)
    if not seq_path.exists():
        print(f"нет сети {seq_path.name}: scripts/train_sequence.py")
        return 1
    net = SulfurSequenceModel.load(seq_path).attach(features)
    boost = SulfurModel.load(SulfurModel.default_path(0))

    out: dict = {}
    for split in ("val", "test"):
        lo, hi = cfg["split"][split]
        index = pd.date_range(pd.Timestamp(lo), pd.Timestamp(hi), freq=GRID)
        index = index.intersection(features.index)
        net_risk = net.predict_risk(index, features).dropna()
        rows = features.loc[net_risk.index]
        boost_risk = boost.predict_risk(rows[boost.features])

        lab = sb.lims_sulfur.loc[str(lo):str(hi)].dropna()
        lab = lab[lab.index >= net_risk.index.min()]
        over = (lab > limit).to_numpy()
        if not over.any():
            continue

        grid = np.quantile(net_risk, np.linspace(0.5, 0.999, 60))
        affordable = [t for t in grid if share_of_hours(net_risk, t) <= budget]
        threshold = float(min(affordable)) if affordable else float(max(grid))

        net_flags = warned(net_risk, lab.index, threshold)
        base_flags = warned(boost_risk, lab.index, float(boost.alarm_threshold))
        block = {
            "проб": int(len(lab)),
            "с превышением": int(over.sum()),
            "порог канала": round(threshold, 4),
            "часов с вниманием, доля": round(share_of_hours(net_risk, threshold), 3),
            "бюджет тревог": budget,
            "канал: предупреждений перед превышением": round(float(net_flags[over].mean()), 3),
            "рабочая модель: то же": round(float(base_flags[over].mean()), 3),
            "канал: ложные предупреждения": round(float(net_flags[~over].mean()), 3),
            "рабочая модель: ложные": round(float(base_flags[~over].mean()), 3),
        }
        block["прирост полноты"] = round(
            block["канал: предупреждений перед превышением"]
            - block["рабочая модель: то же"], 3)
        out[split] = block
        print(f"\n{split}: {json.dumps(block, ensure_ascii=False)}")

    val = out.get("val", {})
    rule = {
        "1. доля часов с вниманием не выше бюджета тревог": bool(
            val.get("часов с вниманием, доля", 1.0) <= budget + 1e-9),
        "2. полнота предупреждения выше рабочей модели на 10 пунктов": bool(
            val.get("прирост полноты", 0.0) >= MIN_GAIN),
    }
    accepted = bool(val) and all(rule.values())
    print("\nПравило приёмки (валидация):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: канал раннего предупреждения "
          f"{'ПРИНЯТ' if accepted else 'НЕ принят'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "окно, ч": WINDOW_HOURS,
                                  "модель канала": seq_path.name,
                                  "выборки": out, "правило": rule,
                                  "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
