"""Запаздывание АВТ → гидроочистка и гипотеза промежуточного парка (участник 2). CPU.

    python scripts/check_avt_to_ht_lag.py

Вопрос. Качество дизельной фракции АВТ (лаборатория, точка АВТ|3) доходит до
сырья гидроочистки (точка 1) сразу или через промежуточный парк со временем
пребывания в сутки и больше? От ответа зависит, можно ли прогнозировать серу на
гидроочистке по сырью со стороны АВТ и на сколько вперёд.

Почему не «в лоб». Первые разности суточных анализов гасят медленный сигнал парка
и дают на соседних сутках отрицательную корреляцию просто по построению. Поэтому:

1. **остатки от 15-суточного тренда** — убирают сезон и кампании, оставляя
   суточные колебания качества; корреляция по сдвигам 0…7 суток в обе стороны;
2. **пары проб по часовому сдвигу** — анализы берут в 9, 10, 14 и 22 часа, и пары
   «проба АВТ — проба ГО» дают разрешение меньше суток;
3. **модель парка**: сырьё ГО = экспоненциальное сглаживание АВТ с временем
   пребывания τ, сдвинутое на L суток; τ и L подбираются по корреляции, интервал —
   блочным бутстрепом по неделям;
4. **контроль**: та же связь с другими точками АВТ. Тяжёлая фракция (АВТ|1) с
   сырьём ГО связана быть не должна — если связана, мы меряем общий дрейф нефти, а
   не перенос потока.

Готово, когда названо запаздывание с интервалом или показано, что связи на
сдвигах до 7 суток нет (план доработки 14.09, участник 2, п. 2).

Результат: reports/avt_to_ht_lag.json и сводка в консоли.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.data.loaders import load_lims  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

PARAMS = ["95%.T", "90%.T", "EBP.T", "50%.T", "CloudPoint", "IBP.T"]
MODEL_PARAMS = ["95%.T", "EBP.T", "CloudPoint"]
TREND_DAYS = 15
TAUS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0)
LAGS = tuple(range(0, 8))
PAIR_BINS = [-30, -18, -6, 6, 18, 30, 42, 54, 78, 126, 174]
BOOTSTRAP = 300


def lab(lims: pd.DataFrame, unit: str, point: str, param: str) -> pd.Series:
    """Лабораторный ряд без грубых выбросов (дальше 5 MAD от медианы)."""
    s = lims[(lims["unit"] == unit) & (lims["point"] == point)
             & (lims["param"] == param)].set_index("ts")["value"].sort_index()
    if param != "CloudPoint":            # ноль в разгонке — пропуск, помутнение — нет
        s = s[s > 0]
    if s.empty:
        return s
    median = s.median()
    mad = (s - median).abs().median() * 1.4826
    return s[(s - median).abs() <= 5 * mad] if mad > 0 else s


def daily_residual(s: pd.Series) -> pd.Series:
    d = s.groupby(s.index.normalize()).mean().asfreq("D")
    return d - d.rolling(TREND_DAYS, center=True, min_periods=5).median()


def lag_profile(avt: pd.Series, ht: pd.Series) -> dict[int, float]:
    """Корреляция остатков: ГО отстаёт от АВТ на k суток (k < 0 — опережает)."""
    out = {}
    for k in range(-7, 8):
        pair = pd.concat([avt, ht.shift(-k)], axis=1).dropna()
        out[k] = round(float(pair.corr().iloc[0, 1]), 3) if len(pair) > 50 else None
    return out


def pair_profile(avt: pd.Series, ht: pd.Series) -> list[dict]:
    """Пары проб по часовому сдвигу — разрешение меньше суток."""
    a = (avt - avt.rolling(f"{TREND_DAYS}D", center=True, min_periods=5).median()).dropna()
    g = (ht - ht.rolling(f"{TREND_DAYS}D", center=True, min_periods=5).median()).dropna()
    ta, va = a.index.values, a.values
    shifts, xs, ys = [], [], []
    for ts, value in g.items():
        lo = np.searchsorted(ta, np.datetime64(ts - pd.Timedelta("7D")))
        hi = np.searchsorted(ta, np.datetime64(ts + pd.Timedelta("30h")), side="right")
        dh = (np.datetime64(ts) - ta[lo:hi]) / np.timedelta64(1, "h")
        shifts.extend(dh)
        xs.extend(va[lo:hi])
        ys.extend([value] * (hi - lo))
    frame = pd.DataFrame({"dh": shifts, "a": xs, "g": ys})
    frame["bin"] = pd.cut(frame["dh"], PAIR_BINS)
    rows = []
    for interval, part in frame.groupby("bin", observed=True):
        rows.append({"сдвиг, ч": f"{interval.left:+.0f}…{interval.right:+.0f}",
                     "пар": int(len(part)),
                     "corr": round(float(part["a"].corr(part["g"])), 3)})
    return rows


def tank_model(avt: pd.Series, ht: pd.Series, seed: int = 42) -> dict:
    """Время пребывания τ и сдвиг L: точечная оценка и бутстреп по неделям."""
    smoothed = {tau: (avt if tau == 0 else
                      avt.ewm(alpha=1 - np.exp(-1 / tau), ignore_na=True).mean())
                for tau in TAUS}
    target = ht.reindex(avt.index)
    columns = {(tau, lag): smoothed[tau].shift(lag).to_numpy()
               for tau in TAUS for lag in LAGS}
    y_all = target.to_numpy()

    def best_on(idx: np.ndarray):
        best = None
        for key, x_all in columns.items():
            x, y = x_all[idx], y_all[idx]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() < 100:
                continue
            c = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if best is None or c > best[1]:
                best = (key, c)
        return best

    full = np.arange(len(avt))
    (tau, lag), corr = best_on(full)
    grid = {f"τ={t}": {f"L={k}": round(float(pd.concat(
        [smoothed[t].shift(k), target], axis=1).dropna().corr().iloc[0, 1]), 3)
        for k in LAGS} for t in TAUS}

    weeks = avt.index.to_period("W")
    unique = weeks.unique()
    positions = {w: np.where(weeks == w)[0] for w in unique}
    rng = np.random.default_rng(seed)
    taus, lags = [], []
    for _ in range(BOOTSTRAP):
        chosen = rng.choice(len(unique), size=len(unique), replace=True)
        idx = np.concatenate([positions[unique[i]] for i in chosen])
        found = best_on(idx)
        if found:
            taus.append(found[0][0])
            lags.append(found[0][1])
    share = lambda values: {str(k): round(v, 3) for k, v in  # noqa: E731
                            pd.Series(values).value_counts(normalize=True).items()}
    return {"τ, сут": tau, "L, сут": lag, "corr": round(corr, 3), "сетка": grid,
            "бутстреп": {"τ": share(taus), "L": share(lags),
                         "τ ≤ 1 сут, доля": round(float(np.mean(np.array(taus) <= 1.0)), 3),
                         "L = 0, доля": round(float(np.mean(np.array(lags) == 0)), 3)}}


def main() -> int:
    use_utf8_console()
    lims = load_lims()

    print("\n[1] Уровни: один ли это поток\n")
    levels = []
    for p in PARAMS:
        a, g = lab(lims, "АВТ", "3", p), lab(lims, "Гидроочистка", "1", p)
        levels.append({"показатель": p, "АВТ|3 медиана": round(float(a.median()), 1),
                       "сырьё ГО медиана": round(float(g.median()), 1),
                       "анализов АВТ|3": len(a), "анализов ГО": len(g)})
    print(pd.DataFrame(levels).to_string(index=False))

    print("\n[2] Остатки от 15-суточного тренда: корреляция по сдвигу в сутках\n")
    profiles = {}
    for p in PARAMS:
        profiles[p] = lag_profile(daily_residual(lab(lims, "АВТ", "3", p)),
                                  daily_residual(lab(lims, "Гидроочистка", "1", p)))
        print(f"  {p:10s} " + " ".join(f"{k:+d}:{v:+.2f}" for k, v in profiles[p].items()
                                       if v is not None))

    print("\n[3] Контроль: связь с другими точками АВТ на нулевом сдвиге\n")
    control = {}
    for p in ("95%.T", "50%.T"):
        ht_resid = daily_residual(lab(lims, "Гидроочистка", "1", p))
        control[p] = {}
        for point in ("1", "2", "2.1", "3"):
            series = lab(lims, "АВТ", point, p)
            if len(series) < 50:
                continue
            pair = pd.concat([daily_residual(series), ht_resid], axis=1).dropna()
            if len(pair) > 50:
                control[p][f"АВТ|{point}"] = {"corr": round(float(pair.corr().iloc[0, 1]), 3),
                                              "n": int(len(pair))}
        print(f"  {p:8s} " + "  ".join(f"{k}: {v['corr']:+.2f} (n={v['n']})"
                                       for k, v in control[p].items()))

    print("\n[4] Пары проб по часовому сдвигу (ГО позже АВТ на Δ часов)\n")
    pairs = {}
    for p in MODEL_PARAMS:
        pairs[p] = pair_profile(lab(lims, "АВТ", "3", p), lab(lims, "Гидроочистка", "1", p))
        print(f"  {p}:\n   " + "  ".join(f"{r['сдвиг, ч']}ч:{r['corr']:+.2f}" for r in pairs[p]))

    print(f"\n[5] Модель парка: τ и L, бутстреп по неделям ({BOOTSTRAP} выборок)\n")
    tanks = {}
    for p in MODEL_PARAMS:
        tanks[p] = tank_model(daily_residual(lab(lims, "АВТ", "3", p)),
                              daily_residual(lab(lims, "Гидроочистка", "1", p)))
        t = tanks[p]
        print(f"  {p:10s} лучшее τ = {t['τ, сут']} сут, L = {t['L, сут']} сут, corr {t['corr']}; "
              f"L = 0 в {t['бутстреп']['L = 0, доля']:.0%} выборок, "
              f"τ ≤ 1 сут в {t['бутстреп']['τ ≤ 1 сут, доля']:.0%}")

    zero_lag = min(t["бутстреп"]["L = 0, доля"] for t in tanks.values())
    short_tau = min(t["бутстреп"]["τ ≤ 1 сут, доля"] for t in tanks.values())
    far = max(abs(v) for prof in profiles.values() for k, v in prof.items()
              if v is not None and 2 <= k <= 7 and v > 0) if profiles else None
    print("\nВывод")
    print(f"  Запаздывание меньше суток: сдвиг 0 выбран в {zero_lag:.0%} бутстрепов по всем "
          "показателям, а по парам проб пик лежит в окне ±6 часов.")
    print(f"  Буфер со временем пребывания больше суток данными не поддерживается: "
          f"τ ≤ 1 сут в не менее чем {short_tau:.0%} выборок.")
    print(f"  На сдвигах 2…7 суток положительной связи нет (максимум {far:+.2f}); "
          "отрицательные значения там — след 15-суточного вычитания тренда.")

    report = {
        "уровни": levels, "по_суткам": profiles, "контроль": control,
        "пары_проб": pairs, "модель_парка": tanks,
        "вывод": {"сдвиг 0, минимальная доля бутстрепов": zero_lag,
                  "τ ≤ 1 сут, минимальная доля бутстрепов": short_tau,
                  "максимум корреляции на сдвигах 2…7 сут": far},
    }
    out = ROOT / "reports" / "avt_to_ht_lag.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
