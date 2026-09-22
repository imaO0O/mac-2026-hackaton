"""Реагирует ли система на превышение — по ЛАБОРАТОРНЫМ ПРОБАМ, а не по моментам. Только CPU.

    python scripts/check_event_response.py
    python scripts/check_event_response.py reports/test_period_step1h.json

Зачем. Прогон по тестовому периоду считает «долю пропусков» по моментам решения:
момент с превышением в ближайшие 24 часа, где система держала режим. У этой меры
два дефекта, и оба вскрылись, когда прогон впервые сделали на реальной частоте —
раз в час вместо раза в 12 часов:

* **она зависит от шага.** На рабочей точке пропусков при шаге 1 ч заметно больше,
  чем при шаге 12 ч, — у одной и той же системы. При часовом шаге после каждого
  действия четыре часа действует запрет на новое вмешательство, и эти часы при
  превышении засчитываются как пропуск, хотя рекомендация уже выдана и в силе.
  Сколько таких — `docs/HARD_CHECKS.md` §3, число там сверяется с прогоном тестом
  (`tests/test_readme_matches_reports.py`, `hourly_misses`);
* **она требует предвидения на сутки**, а виртуальный анализатор работает на
  нулевом горизонте и видит превышение, которое уже идёт.

Здесь мера другая. Единица — лабораторная проба теста. Для каждого окна
относительно момента отбора считается, была ли в окне хотя бы одна рекомендация
«меняем уставки» — отдельно перед пробами с превышением и перед нормальными.
Разница этих двух долей — сколько действия системы говорят о превышении. Моменты
отказа в окне не считаются (решение не принималось); окно, где отказ сплошной,
выпадает.

Интервал разницы — 90 %, блочный бутстрэп по неделям проб.

Результат: таблица в консоли и reports/event_response.json.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

DEFAULT_RUNS = ("reports/test_period.json", "reports/test_period_step1h.json")
# окна в часах относительно отбора пробы; публикация результата — через 4 ч
WINDOWS = ((-2, 0), (0, 4), (-2, 4), (-6, 0), (-24, 0))
BOOTSTRAP = 2000
REPORT = ROOT / "reports" / "event_response.json"


def window_table(rows: pd.DataFrame, lab: pd.Series, limit: float, lo: int, hi: int) -> pd.DataFrame:
    out = []
    for t, value in lab.items():
        w = rows.loc[t + pd.Timedelta(hours=lo): t + pd.Timedelta(hours=hi) - pd.Timedelta("1s")]
        w = w[~w["исход"].eq("отказ")]
        if w.empty or w["риск"].isna().all():
            continue
        out.append({"ts": t, "over": bool(value > limit),
                    "acted": bool(w["исход"].eq("меняем уставки").any()),
                    "risk": float(w["риск"].max())})
    return pd.DataFrame(out)


def bootstrap_difference(table: pd.DataFrame, key: str) -> list[float]:
    weeks = pd.Index(table["ts"].dt.to_period("W").astype(str))
    codes, uniques = pd.factorize(weeks)
    members = [np.flatnonzero(codes == w) for w in range(len(uniques))]
    over, acted = table["over"].to_numpy(), table["acted"].to_numpy()
    rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
    diffs = []
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([members[w] for w in rng.integers(0, len(members), len(members))])
        o, a = over[idx], acted[idx]
        if o.any() and (~o).any():
            diffs.append(a[o].mean() - a[~o].mean())
    lo, hi = np.percentile(diffs, [5, 95])
    return [round(float(lo), 3), round(float(hi), 3)]


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    all_lab = StateBuilder(cfg).lims_sulfur

    result = {}
    # --out <файл>: отчёт по другим прогонам (например, подбору на валидации) не
    # должен затирать рабочий event_response.json — так однажды и случилось
    args = sys.argv[1:]
    report_path = REPORT
    if "--out" in args:
        at = args.index("--out")
        report_path = ROOT / args[at + 1]
        args = args[:at] + args[at + 2:]
    for run in (args or DEFAULT_RUNS):
        path = ROOT / run
        if not path.exists():
            print(f"[пропуск] нет {run}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        # пробы — из периода САМОГО прогона: прогоны бывают и по валидации
        period_lo, period_hi = data["summary"].get("период", cfg["split"]["test"])
        lab = all_lab.loc[str(period_lo):str(period_hi)]
        print(f"\n{path.name}: проб серы в периоде {len(lab)}, "
              f"из них с превышением {int((lab > limit).sum())}")
        rows = pd.DataFrame(data["rows"])
        rows["ts"] = pd.to_datetime(rows["ts"])
        rows = rows.set_index("ts").sort_index()
        step = data["summary"]["шаг"]
        days = max((rows.index[-1] - rows.index[0]).total_seconds() / 86400, 1)
        block = {"шаг": step,
                 "вмешательств_в_сутки": round(float(rows["исход"].eq("меняем уставки").sum() / days), 2),
                 "окна": {}}
        print(f"\n=== {path.name}: шаг {step}, вмешательств в сутки {block['вмешательств_в_сутки']}")
        print(f"{'окно, ч':>10s} {'перед превыш.':>14s} {'перед нормой':>13s} {'разница':>8s} "
              f"{'90 % интервал':>16s} {'ROC-AUC риска':>14s}")
        for lo, hi in WINDOWS:
            # окно короче шага — моментов в нём почти нет, мерить нечего
            if (hi - lo) < pd.Timedelta(step).total_seconds() / 3600:
                continue
            table = window_table(rows, lab, limit, lo, hi)
            if table.empty or table["over"].nunique() < 2:
                continue
            hit = float(table.loc[table["over"], "acted"].mean())
            base = float(table.loc[~table["over"], "acted"].mean())
            ci = bootstrap_difference(table, f"{path.name}/{lo}/{hi}")
            auc = float(roc_auc_score(table["over"], table["risk"]))
            key = f"[{lo:+d}, {hi:+d})"
            block["окна"][key] = {"перед_превышением": round(hit, 3),
                                  "перед_нормой": round(base, 3),
                                  "разница": round(hit - base, 3), "90% интервал": ci,
                                  "roc_auc_макс_риска": round(auc, 3),
                                  "проб_с_превышением": int(table["over"].sum()),
                                  "проб_без": int((~table["over"]).sum())}
            print(f"{key:>10s} {hit:>13.0%} {base:>13.0%} {hit - base:>+8.0%} "
                  f"{f'{ci[0]:+.0%}…{ci[1]:+.0%}':>16s} {auc:>14.3f}")
        result[path.name] = block

    report_path.write_text(json.dumps({**report_provenance(cfg), "предел": limit,
                                       "бутстрэп": {"блок": "неделя", "выборок": BOOTSTRAP},
                                       "прогоны": result}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nОтчёт: {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
