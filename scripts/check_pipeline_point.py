"""Какая лабораторная точка стоит за «Pipeline» в формулах ВАК гидроочистки. Только CPU.

    python scripts/check_pipeline_point.py

Зачем. Исправленные организаторами формулы `24-2000:GODT:T95` и `24-2000:GODT:D15`
берут на вход лабораторное значение `LIMS:24-2000.Pipeline.*`. Такой точки в
выгрузке нет; есть «Гидроочистка, точка 1» (сырьё, ФРАКЦ_ДИЗ) и «точка 2»
(товарное ДТ). В коде с самого начала стоит точка 2 (`models/vak.py`,
`LIMS_TOKENS`). На сессии вопросов 11.09 на прямой вопрос «Pipeline — это точка 1,
то есть сырьё?» прозвучало «да, всё верно… точки отбора перед установкой и после
установки» — ответ можно прочитать как подтверждение точки 1, но прямо это не
сказано.

Проверка. Обе формулы считаются с каждым прочтением на всех лабораторных
анализах товарного ДТ, и результат сравнивается с этим анализом:

* **А** — точка 2, последний анализ, опубликованный до отбора (как в коде);
* **Б** — точка 1, последний анализ, опубликованный до отбора (честный вариант
  для работы в реальном времени);
* **Б'** — точка 1, отобранная не позже целевой пробы, без задержки публикации.
  В реальном времени недоступна; показана, чтобы увидеть, сколько физики теряется
  на задержке.

Эталоны — сами последние анализы точек 2 и 1 без всякой формулы.

Результат: таблица в консоли и reports/pipeline_point.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import known_from  # noqa: E402
from nefte.data.loaders import lims_series, load_lims  # noqa: E402
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.models.vak import compile_formulas  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "pipeline_point.json"
# цель, лабораторный вход формулы и физический диапазон для отсева выбросов
CASES = {
    "24-2000:GODT:T95": ("95%.T", "LIMS_T95", (250.0, 420.0)),
    "24-2000:GODT:D15": ("D15", "LIMS_D15", (780.0, 900.0)),
}


def _asof(index: pd.DatetimeIndex, series: pd.Series, exact: bool) -> np.ndarray:
    s = series.dropna().sort_index()
    left = pd.DataFrame({"ts": index})
    right = pd.DataFrame({"ts": s.index, "v": s.to_numpy()})
    out = pd.merge_asof(left, right, on="ts", allow_exact_matches=exact)
    return out["v"].to_numpy(dtype=float)


def _stats(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    e = pred[mask] - truth[mask]
    return {"пар": int(mask.sum()), "смещение": round(float(e.mean()), 2),
            "MAE": round(float(np.abs(e).mean()), 2),
            "corr": round(float(np.corrcoef(pred[mask], truth[mask])[0, 1]), 3)}


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    delay = float(cfg["quality"].get("lims_publication_delay_hours", 0.0))
    lims = load_lims()
    matrix = build_feature_matrix()
    usable, _ = compile_formulas()
    result = {}

    for target, (param, token, (lo, hi)) in CASES.items():
        item = next(i for i in usable if i["target"] == target)
        truth_series = lims_series(f"Гидроочистка|2|{param}", lims)
        truth_series = truth_series[(truth_series > lo) & (truth_series < hi)].sort_index()
        feed = lims_series(f"Гидроочистка|1|{param}", lims)
        feed = feed[(feed > lo) & (feed < hi)]
        ts = truth_series.index
        truth = truth_series.to_numpy(dtype=float)

        columns = ["ht_" + t for t in item["tags"]]
        tel = matrix[columns].dropna().sort_index()
        joined = pd.merge_asof(pd.DataFrame({"ts": ts}),
                               tel.rename_axis("ts").reset_index(), on="ts",
                               direction="backward", tolerance=pd.Timedelta("2h"))

        def formula(lab: np.ndarray, item=item, joined=joined, token=token) -> np.ndarray:
            ns = {t: joined["ht_" + t].to_numpy(dtype=float) for t in item["tags"]}
            ns[token] = lab
            return np.asarray(eval(item["code"], {"__builtins__": {}, "np": np}, ns),  # noqa: S307
                              dtype=float)

        readings = {
            "А: точка 2, опубликованный до отбора": _asof(ts, known_from(truth_series, delay), False),
            "Б: точка 1, опубликованный до отбора": _asof(ts, known_from(feed, delay), False),
            "Б': точка 1 без задержки публикации": _asof(ts, feed, True),
        }
        preds = {name: formula(lab) for name, lab in readings.items()}
        common = ~np.isnan(truth)
        for p in preds.values():
            common &= ~np.isnan(p)

        rows = {name: _stats(p, truth, common) for name, p in preds.items()}
        rows["эталон: последний анализ точки 2 без формулы"] = _stats(
            readings["А: точка 2, опубликованный до отбора"], truth, common)
        rows["эталон: последний анализ точки 1 без формулы"] = _stats(
            readings["Б: точка 1, опубликованный до отбора"], truth, common)

        pairs = pd.merge_asof(
            pd.DataFrame({"ts": ts, "p2": truth}),
            pd.DataFrame({"ts": feed.sort_index().index, "p1": feed.sort_index().to_numpy()}),
            on="ts", direction="nearest", tolerance=pd.Timedelta("12h")).dropna()
        gap = {"пар в пределах 12 ч": int(len(pairs)),
               "медиана точка 1 − точка 2": round(float((pairs["p1"] - pairs["p2"]).median()), 2)}

        print(f"\n{target}: анализов товарного ДТ {len(ts)}, сырья {len(feed)}; "
              f"сырьё выше продукта на {gap['медиана точка 1 − точка 2']:+.2f} "
              f"({gap['пар в пределах 12 ч']} пар)")
        print(pd.DataFrame(rows).T.to_string())
        result[target] = {"формула": item["expr"], "анализов_точки_2": int(len(ts)),
                          "анализов_точки_1": int(len(feed)), "сырьё_против_продукта": gap,
                          "прочтения": rows}

    print("\nЧитать так. Прочтение важно только для признаков vak_*: ограничение по Т95 "
          "в оптимизаторе берёт уровень из лаборатории, а из формулы — приращение от "
          "уставок, и лабораторный вход в приращении сокращается при любом прочтении.")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "задержка_публикации_ч": delay,
                                  "формулы": result}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
