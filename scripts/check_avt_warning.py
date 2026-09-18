"""Даёт ли режим АВТ предупреждение по сере за 2 часа (участник 2). Только CPU.

    python scripts/check_avt_warning.py

Зачем. У бустинга на горизонте 2 ч сигнала нет — это главная слабость системы.
Запаздывание АВТ → гидроочистка меньше шести часов (`docs/AVT_TO_HT_LAG.md`), и
изменение дизельного погона на АВТ может дойти до серы гидроочищенного ДТ позже,
чем через два часа. Если так, режим АВТ предупреждает заранее.

Правило записано ДО счёта (docs/PLAN.md, коммит 13006e7) и исполняется как есть:

* цель — новое превышение по `Q21` на часовой сетке: медиана за последний час не
  выше 10 мг/кг, а за час [t+2 ч, t+3 ч) — выше; остановы и заглушка 307 исключены;
* база — только гидроочистка: уровень `Q21` и его изменение за 2 и 6 ч, WABT и
  нагрузка `F26`, их изменения за 6 ч;
* АВТ — `T55`, `T33`, `F30`, `F32`, `F65`, названные заранее; у каждого изменение за
  6 ч и отклонение от медианы за 7 суток по прошлому;
* логистическая регрессия со стандартизацией, обучение на обучающем периоде, мера —
  PR-AUC на валидации; АВТ предупреждает, если прибавляет не меньше 0.05. Тест —
  описательно.

Результат: reports/avt_warning.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.loaders import load_telemetry  # noqa: E402
from nefte.models.regime import FEED, REACTOR_TEMPS, outage_mask  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

AVT_TAGS = ("T55", "T33", "F30", "F32", "F65")
LIMIT = 10.0
STUB = 307.0
MIN_GAIN = 0.05
REPORT = ROOT / "reports" / "avt_warning.json"


def hourly(series: pd.Series, how: str = "mean") -> pd.Series:
    """Значение на момент t — по прошедшему часу (t−1 ч, t]."""
    return getattr(series.resample("1h", label="right", closed="right"), how)()


def build_table(cfg: dict) -> pd.DataFrame:
    ht = load_telemetry("ht")
    avt = load_telemetry("avt")
    q21 = ht["Q21"].where((ht["Q21"] > 0) & (ht["Q21"].round(1) != STUB))
    down = hourly(outage_mask(ht[FEED], retrospective=True).astype(float), "max") > 0

    q = hourly(q21, "median")
    wabt = hourly(ht[[t for t in REACTOR_TEMPS if t in ht.columns]].mean(axis=1))
    feed = hourly(ht[FEED])
    table = pd.DataFrame({
        "q21": q, "q21_d2": q - q.shift(2), "q21_d6": q - q.shift(6),
        "wabt": wabt, "feed": feed,
        "wabt_d6": wabt - wabt.shift(6), "feed_d6": feed - feed.shift(6),
    })
    for tag in AVT_TAGS:
        a = hourly(avt[tag]).reindex(table.index)
        table[f"avt_{tag}_d6"] = a - a.shift(6)
        table[f"avt_{tag}_dev7"] = a - a.rolling("7D", min_periods=24).median()
    future = q.shift(-3)            # медиана за (t+2 ч, t+3 ч]
    table["future"] = future
    stopped = (down.reindex(table.index, fill_value=True)
               | down.shift(-3, fill_value=True).reindex(table.index, fill_value=True))
    table = table[~stopped & (table["q21"] <= LIMIT)].dropna()
    table["event"] = (table["future"] > LIMIT).astype(int)
    return table


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    table = build_table(cfg)
    base = ["q21", "q21_d2", "q21_d6", "wabt", "feed", "wabt_d6", "feed_d6"]
    with_avt = base + [c for c in table.columns if c.startswith("avt_")]

    splits = {name: table.loc[lo:hi] for name, (lo, hi) in
              ((s, cfg["split"][s]) for s in ("train", "val", "test"))}
    out: dict = {"часов": {s: len(v) for s, v in splits.items()},
                 "новых превышений": {s: int(v["event"].sum()) for s, v in splits.items()}}
    scores: dict = {}
    for name, columns in (("база", base), ("база + АВТ", with_avt)):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(splits["train"][columns], splits["train"]["event"])
        scores[name] = {}
        for split in ("val", "test"):
            part = splits[split]
            risk = model.predict_proba(part[columns])[:, 1]
            scores[name][split] = {
                "pr_auc": round(float(average_precision_score(part["event"], risk)), 3),
                "roc_auc": round(float(roc_auc_score(part["event"], risk)), 3),
                "частота": round(float(part["event"].mean()), 4)}
    gain = scores["база + АВТ"]["val"]["pr_auc"] - scores["база"]["val"]["pr_auc"]
    accepted = gain >= MIN_GAIN

    print(f"Часов в работе при сере в норме: {out['часов']}; новых превышений через 2 ч: "
          f"{out['новых превышений']}")
    for name, block in scores.items():
        print(f"  {name:12s} валидация {block['val']}  тест {block['test']}")
    print(f"\nПрибавка PR-AUC от АВТ на валидации: {gain:+.3f} (нужно не меньше {MIN_GAIN})")
    print("ВЫВОД: " + ("режим АВТ даёт предупреждение за 2 ч" if accepted else
                       "режим АВТ предупреждения за 2 ч не даёт — измеренный отказ"))

    REPORT.write_text(json.dumps({**report_provenance(cfg), "теги АВТ": list(AVT_TAGS),
                                  **out, "качество": scores,
                                  "прибавка_pr_auc_валидация": round(gain, 3),
                                  "принято": accepted}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
