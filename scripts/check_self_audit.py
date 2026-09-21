"""Сбываются ли обещания системы: самоконтроль по приходящим анализам. Только CPU.

    python scripts/check_self_audit.py

Зачем. Все наши числа посчитаны на всей выборке разом: покрытие интервала 77 %,
MAE 1.3 мг/кг. Оператору это не помогает: он видит одну карточку и не знает, врала
ли система на прошлой неделе. А модель может испортиться молча — сменился режим,
поменяли катализатор, уехал прибор, — и заметить это должна сама система, а не
разбор через полгода.

Считаем скользящую сбываемость: по последним N лабораторным анализам — какая доля
попала в интервал, который система обещала на момент отбора пробы. Это ровно то,
что можно показывать на дашборде и в карточке одной строкой.

**Правило записано ДО счёта.** Самоконтроль имеет смысл, если на ВАЛИДАЦИИ:

1. он чувствителен: скользящее покрытие хотя бы раз опускается ниже 50 % —
   значит, есть что ловить, а не ровная линия около среднего;
2. он не кричит постоянно: доля окон ниже 50 % меньше четверти — иначе тревога
   станет фоном.

Не выполняется — измеренный отказ: сбываемость остаётся общим числом в отчёте и в
карточку не идёт.
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
from nefte.models.quality_model import Z90, SulfurModel  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "self_audit.json"
WINDOW = 14                      # две недели анализов при суточном ритме
LOW = 0.50
MAX_ALARM_SHARE = 0.25


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    model = SulfurModel.load(SulfurModel.default_path(0.0, "sulfur"))
    features = build_feature_matrix()
    masks = time_split(features.index, cfg)
    sb = StateBuilder(cfg)
    lab = sb.lims_sulfur.dropna().sort_index()

    out: dict = {"окно, анализов": WINDOW}
    for split in ("val", "test"):
        frame = features[masks[split].to_numpy()]
        if frame.empty:
            continue
        lo, hi = cfg["split"][split]
        sample = lab.loc[str(lo):str(hi)]
        if sample.empty:
            continue
        # прогноз на МОМЕНТ ОТБОРА пробы: сверяем то, что система говорила тогда
        index = pd.DatetimeIndex(sample.index)
        pos = frame.index.searchsorted(index, side="right") - 1
        ok = pos >= 0
        rows = frame.iloc[np.clip(pos, 0, len(frame) - 1)][model.features]
        # выравнивание по одной и той же маске: dropna выкидывает произвольные
        # строки, и брать «последние столько же» анализов значило бы сверять
        # прогноз с чужой пробой
        alive = ok & rows.notna().all(axis=1).to_numpy()
        if not alive.any():
            continue
        pred = model.predict_frame(rows[alive])
        truth = sample.to_numpy()[alive]
        # интервал 80 % — тот же, каким он определён во всём проекте
        # (interval_metrics): q50 +- Z90*sigma с конформной поправкой, а не сырые
        # квантили q10/q90
        half = Z90 * pred["sigma"].to_numpy()
        middle = pred["q50"].to_numpy()
        inside = ((truth >= middle - half) & (truth <= middle + half)).astype(float)
        rolling = pd.Series(inside).rolling(WINDOW).mean().dropna()
        out[split] = {
            "анализов": int(len(truth)),
            "покрытие целиком": round(float(inside.mean()), 3),
            "скользящее покрытие: минимум": round(float(rolling.min()), 3),
            "скользящее покрытие: медиана": round(float(rolling.median()), 3),
            "доля окон ниже 50 %": round(float((rolling < LOW).mean()), 3),
            "окон": int(len(rolling)),
        }
        print(f"\n{split}:")
        for key, value in out[split].items():
            print(f"  {key}: {value}")

    val = out.get("val", {})
    rule = {
        "1. чувствителен (минимум окна ниже 50 %)": bool(
            float(val.get("скользящее покрытие: минимум", 1.0)) < LOW),
        "2. не кричит постоянно (таких окон меньше четверти)": bool(
            float(val.get("доля окон ниже 50 %", 1.0)) < MAX_ALARM_SHARE),
    }
    accepted = all(rule.values())
    print("\nПравило приёмки (валидация):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: самоконтроль сбываемости {'ПРИНЯТ' if accepted else 'НЕ принят'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), **out,
                                  "условие: низкое покрытие": LOW,
                                  "условие: доля тревожных окон меньше": MAX_ALARM_SHARE,
                                  "правило": rule, "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
