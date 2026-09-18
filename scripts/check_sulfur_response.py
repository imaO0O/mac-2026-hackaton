"""Отклик серы на уставки по эпизодам «не в ответ на серу» (участник 2). Только CPU.

    python scripts/check_sulfur_response.py

Зачем. Отклик серы на уставки в системе принят, а не измерен: кинетика первого
порядка даёт около −20 % серы на градус, порядок 1.5 — около −6 %, второй — около
−3 % (`scripts/check_kinetic_order.py`). По истории его в лоб не выделить: оператор
двигает режим, увидев серу, и обратная связь гасит видимую связь. Чистые эпизоды
(`scripts/find_regime_episodes.py`) — ступеньки режима, причиной которых сера не
была, — единственное место, где отклик можно измерить.

Правило записано ДО счёта (docs/PLAN.md) и исполняется как есть:

* эпизоды — чистые ступеньки; основная выборка — все три переменные, вторичная — WABT;
* сера — `Q21` без заглушки 307: медиана за 6 ч до ступеньки и за 12–24 ч после; эпизод
  годен, если в обоих окнах не меньше 60 % отсчётов и сдвиг переменной держится (тот
  же знак, не меньше половины ступеньки);
* предсказание — кинетика системы (`models/kinetics`) на фактических значениях до и
  после, при порядке 1, 1.5 и 2; сера сырья — лаборатория на момент эпизода;
* мера — наклон β измеренного Δln S на предсказанный при первом порядке, без
  свободного члена; 90 % интервал — бутстрэп по эпизодам. Порядок согласуется, если
  отношение его предсказания к первому порядку при медианных условиях лежит в
  интервале β обучения. Валидация проверяет, тест — описательно.

Результат: reports/sulfur_response.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.loaders import lims_series, load_telemetry  # noqa: E402
from nefte.models.kinetics import H2_ORDER, arrhenius_factor  # noqa: E402
from nefte.models.regime import FEED, PRESSURE, REACTOR_TEMPS  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

BEFORE = pd.Timedelta(hours=6)
AFTER = (pd.Timedelta(hours=12), pd.Timedelta(hours=24))
MIN_VALID = 0.6
HOLD = 0.5
STUB = 307.0
ORDERS = (1.0, 1.5, 2.0)
BOOTSTRAP = 2000
FEED_SULFUR = "Гидроочистка|1|Mass.Sulfur"      # % масс.
REPORT = ROOT / "reports" / "sulfur_response.json"


def predicted_dln(s_out: float, s_in: float, t: tuple, f: tuple, p: tuple,
                  order: float) -> float:
    """Δln S по кинетике системы (models/kinetics.make_kinetic_surrogate)."""
    factor = arrhenius_factor(t[0], t[1]) * (f[0] / f[1]) * (p[1] / p[0]) ** H2_ORDER
    tau = math.log(s_in / s_out)
    if abs(order - 1.0) < 1e-9:
        return -tau * (factor - 1.0)
    q = 1.0 - order
    complex_now = (s_out ** q - s_in ** q) / (order - 1.0)
    value = (s_in ** q + (order - 1.0) * complex_now * factor) ** (1.0 / q)
    return math.log(value / s_out)


def window(series: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[float, float]:
    part = series.loc[lo:hi - pd.Timedelta("1s")]
    expected = max(int((hi - lo) / pd.Timedelta(minutes=10)), 1)
    return float(part.median()), float(part.notna().sum() / expected)


def beta(measured: np.ndarray, first: np.ndarray, rng) -> tuple[float, list[float]]:
    point = float((measured * first).sum() / (first ** 2).sum())
    draws = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(measured), len(measured))
        den = (first[i] ** 2).sum()
        if den > 0:
            draws.append(float((measured[i] * first[i]).sum() / den))
    return round(point, 3), [round(float(x), 3) for x in np.percentile(draws, [5, 95])]


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    episodes = json.loads((ROOT / "reports" / "regime_episodes.json")
                          .read_text(encoding="utf-8"))
    clean = [e for e in episodes["эпизоды"] if e.get("чистый")]

    ht = load_telemetry("ht")
    q21 = ht["Q21"].where((ht["Q21"] > 0) & (ht["Q21"].round(1) != STUB))
    wabt = ht[[t for t in REACTOR_TEMPS if t in ht.columns]].mean(axis=1)
    columns = {"wabt": wabt, "feed": ht[FEED], "pressure": ht[PRESSURE]}
    feed_sulfur = lims_series(FEED_SULFUR).dropna()
    feed_sulfur = feed_sulfur[feed_sulfur > 0] * 1e4          # в ppm

    rows = []
    for e in clean:
        t = pd.Timestamp(e["момент"])
        s_b, valid_b = window(q21, t - BEFORE, t)
        s_a, valid_a = window(q21, t + AFTER[0], t + AFTER[1])
        known = feed_sulfur.loc[:t]
        if valid_b < MIN_VALID or valid_a < MIN_VALID or known.empty:
            continue
        if not (s_b > 0 and s_a > 0):
            continue
        means = {k: (window(v, t - BEFORE, t)[0], window(v, t + AFTER[0], t + AFTER[1])[0])
                 for k, v in columns.items()}
        shift = means[e["переменная"]][1] - means[e["переменная"]][0]
        step = float(e["изменение"])
        if np.sign(shift) != np.sign(step) or abs(shift) < HOLD * abs(step):
            continue
        s_in = float(known.iloc[-1])
        if not s_in > s_b:
            continue
        pred = {n: predicted_dln(s_b, s_in, means["wabt"], means["feed"], means["pressure"], n)
                for n in ORDERS}
        rows.append({"момент": str(t), "переменная": e["переменная"], "ступенька": step,
                     "S до": round(s_b, 2), "S после": round(s_a, 2),
                     "измерено Δln S": round(math.log(s_a / s_b), 4),
                     **{f"предсказано n={n:g}": round(v, 4) for n, v in pred.items()}})
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["момент"])

    # отношение предсказания порядка n к первому — при медианных условиях и +1 °C
    med = {"s_out": float(frame["S до"].median()), "s_in": float(feed_sulfur.median()),
           "t": float(wabt.median())}
    step_1c = (med["t"], med["t"] + 1.0)
    first_1c = predicted_dln(med["s_out"], med["s_in"], step_1c, (1.0, 1.0), (1.0, 1.0), 1.0)
    ratio = {n: round(predicted_dln(med["s_out"], med["s_in"], step_1c, (1.0, 1.0),
                                    (1.0, 1.0), n) / first_1c, 3) for n in ORDERS}
    per_degree = {n: round(100 * (math.exp(ratio[n] * first_1c) - 1), 1) for n in ORDERS}

    rng = np.random.default_rng(20260918)
    result = {}
    for sample, cond in (("все чистые", None), ("только WABT", "wabt")):
        part = frame if cond is None else frame[frame["переменная"] == cond]
        block = {}
        for split in ("train", "val", "test"):
            lo, hi = cfg["split"][split]
            s = part[(part["ts"] >= lo) & (part["ts"] <= pd.Timestamp(hi) + pd.Timedelta(days=1))]
            if len(s) < 3:
                block[split] = {"эпизодов": len(s)}
                continue
            point, ci = beta(s["измерено Δln S"].to_numpy(), s["предсказано n=1"].to_numpy(),
                             rng)
            block[split] = {"эпизодов": len(s), "β": point, "90% интервал": ci}
        train_ci = block["train"].get("90% интервал")
        block["согласуются порядки"] = ([f"{n:g}" for n in ORDERS
                                         if train_ci[0] <= ratio[n] <= train_ci[1]]
                                        if train_ci else [])
        val_beta = block["val"].get("β")
        block["валидация в интервале обучения"] = (
            None if val_beta is None or not train_ci
            else bool(train_ci[0] <= val_beta <= train_ci[1]))
        result[sample] = block

    print(f"Чистых эпизодов {len(clean)}, годных по правилу {len(frame)} "
          f"({frame['переменная'].value_counts().to_dict()})")
    print(f"Отношение предсказания к первому порядку при медианных условиях: {ratio}")
    print(f"Отклик на +1 °C по кинетике системы, %: {per_degree}")
    for sample, block in result.items():
        print(f"\n{sample}:")
        for split in ("train", "val", "test"):
            print(f"  {split:5s} {block[split]}")
        print(f"  согласуются порядки: {block['согласуются порядки'] or 'ни один'}; "
              f"валидация в интервале обучения: {block['валидация в интервале обучения']}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "условия": med,
                                  "отношение_к_первому_порядку": ratio,
                                  "отклик_на_градус_%": per_degree, "итог": result,
                                  "эпизоды": rows}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
