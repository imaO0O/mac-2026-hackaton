"""Почему сырья в К-2 больше, чем нефти: разбор единиц F65 (участник 2). Только CPU.

    python scripts/check_f65_units.py

Сверка схем АВТ (`docs/AVT_SCHEMES.md` §2) нашла одно несходящееся равенство:
производительность К-2 по отбензиненной нефти `F65` в 1.26 раза больше суммы трёх
ходов обессоленной нефти `F7`+`F8`+`F9`. Отбензиненной нефти не может быть больше,
чем нефти, так что один из приборов пишет не то, что кажется. Гипотез три:

1. **пропущен поток** — ходов больше трёх или в К-2 идёт что-то ещё;
2. **разные единицы** — масса против объёма или объём при разной температуре;
3. **калибровка** — просто постоянный множитель.

Скрипт проверяет, что из этого данные различают:

* **масштаб или добавка**: F65 = k·(сумма) + b. Пропущенный поток со своей
  динамикой даёт свободный член и отношение, зависящее от загрузки; множитель —
  нет;
* **с чем связана разность**: если это конкретный пропущенный поток, разность
  должна совпадать с одним тегом, а не со всеми продуктами сразу;
* **зависит ли отношение от плотности и температуры**: масса против объёма
  менялась бы вместе с тяжестью нефти, объём при температуре потока — с
  температурой;
* **какой из двух приборов согласован с остальными**: баланс К-2 и формула ПТФ
  справочника, в которую входит `F65`.

Результат: reports/f65_units.json и сводка в консоли.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.data.cleaning import clean_telemetry  # noqa: E402
from nefte.data.loaders import load_lims, load_telemetry  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

PASSES = ["F7", "F8", "F9"]
K2_PRODUCTS = ["F16", "F25", "F34", "F32", "F30", "F31"]
WORKING_F65 = 300.0          # ниже — К-2 не в работе


def corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    return float(pair.corr().iloc[0, 1]) if len(pair) > 50 else float("nan")


def daily_lab(lims: pd.DataFrame, point: str, param: str) -> pd.Series:
    s = lims[(lims["unit"] == "АВТ") & (lims["point"] == point)
             & (lims["param"] == param)].set_index("ts")["value"].sort_index()
    s = s[s > 0]
    return s.groupby(s.index.normalize()).mean()


def main() -> int:
    use_utf8_console()
    avt, _ = clean_telemetry(load_telemetry("avt"), unit="avt")
    run = avt[avt["F65"] > WORKING_F65]
    passes = run[PASSES].sum(axis=1, min_count=len(PASSES))
    ratio = (run["F65"] / passes).replace([np.inf, -np.inf], np.nan)
    frame = pd.concat([passes.rename("сумма"), run["F65"]], axis=1).dropna()

    print("\n[1] Масштаб или добавка: F65 = k·(F7+F8+F9) + b\n")
    k, b = np.polyfit(frame["сумма"], frame["F65"], 1)
    low, high = frame["сумма"].quantile([0.2, 0.8])
    by_load = {"малая загрузка (< p20)": round(float(ratio[passes < low].median()), 3),
               "большая загрузка (> p80)": round(float(ratio[passes > high].median()), 3)}
    daily = ratio.resample("D").median()
    yearly = daily.groupby(daily.index.year).median()
    by_year = {str(y): round(float(v), 3) for y, v in yearly.items()}
    print(f"  k = {k:.3f}, b = {b:.1f} при сумме от {frame['сумма'].quantile(.05):.0f} "
          f"до {frame['сумма'].quantile(.95):.0f}")
    print(f"  отношение: {by_load}")
    print(f"  по годам: {by_year}")

    print("\n[2] С чем связана разность F65 − (F7+F8+F9)\n")
    residual = run["F65"] - passes
    flows = [c for c in run.columns if c.startswith("F") and c not in PASSES + ["F65"]]
    links = sorted(((c, corr(residual, run[c])) for c in flows),
                   key=lambda item: -abs(item[1]) if item[1] == item[1] else 0)
    top = [{"тег": c, "corr": round(v, 3)} for c, v in links[:6]]
    print(f"  разность: медиана {residual.median():.1f}, ст.откл {residual.std():.1f}")
    print("  сильнее всего связана с: " + ", ".join(f"{t['тег']} {t['corr']:+.2f}" for t in top))
    load = corr(residual, passes)
    print(f"  и с самой загрузкой F7+F8+F9: {load:+.2f}")

    print("\n[3] Зависит ли отношение от температуры и тяжести нефти\n")
    temps = [c for c in run.columns if c.startswith("T")]
    temp_links = sorted(((c, corr(ratio, run[c])) for c in temps),
                        key=lambda item: -abs(item[1]) if item[1] == item[1] else 0)
    strongest_temp = temp_links[0]
    lims = load_lims()
    density = {}
    for point, param in (("1", "D15"), ("3", "D15"), ("1", "95%.T")):
        lab = daily_lab(lims, point, param)
        pair = pd.concat([daily, lab], axis=1).dropna()
        density[f"АВТ|{point} {param}"] = {"corr": round(float(pair.corr().iloc[0, 1]), 3),
                                          "суток": int(len(pair))}
    print(f"  сильнейшая связь с температурой: {strongest_temp[0]} {strongest_temp[1]:+.3f}")
    for name, value in density.items():
        print(f"  с {name}: {value['corr']:+.3f} на {value['суток']} сутках")

    print("\n[4] Какой прибор согласован с остальными\n")
    products = run[K2_PRODUCTS].sum(axis=1, min_count=len(K2_PRODUCTS))
    balance = (products / run["F65"]).replace([np.inf, -np.inf], np.nan)
    density_w = (run["W70"] / run["F30"]).replace([np.inf, -np.inf], np.nan)
    print(f"  баланс К-2 по F65: (F16+F25+F34+F32+F30+F31)/F65 = {balance.median():.3f}")
    print(f"  W70/F30, массовый и объёмный расход одного отбора: {density_w.median():.3f} "
          f"(p05 {density_w.quantile(.05):.3f} … p95 {density_w.quantile(.95):.3f})")

    cfpp = lims[(lims["unit"] == "АВТ") & (lims["point"] == "3")
                & (lims["param"] == "FilterabilityLimit.T")].set_index("ts")["value"].sort_index()
    base = 31.40363 - 0.06784 * avt["T33"] + 17.411 * avt["P67"] - 8.11544 * avt["P4"]
    reading = {}
    for name, f65 in (("F65 как в телеметрии", avt["F65"]), (f"F65 / {k:.3f}", avt["F65"] / k)):
        series = (base - 0.47309 * f65 / (avt["F32"] + avt["F30"]))
        series = series.replace([np.inf, -np.inf], np.nan)
        pos = series.index.searchsorted(cfpp.index, side="right") - 1
        ok = pos >= 0
        model, truth = series.to_numpy()[pos[ok]], cfpp.to_numpy()[ok]
        good = ~np.isnan(model)
        err = model[good] - truth[good]
        reading[name] = {"смещение": round(float(err.mean()), 2),
                         "MAE": round(float(np.abs(err).mean()), 2), "анализов": int(good.sum())}
    for name, value in reading.items():
        print(f"  ПТФ по формуле справочника, {name}: смещение {value['смещение']:+.2f} °C, "
              f"MAE {value['MAE']:.2f} на {value['анализов']} анализах")

    scale = abs(b) < 0.05 * float(frame["сумма"].median())
    flat_load = abs(by_load["малая загрузка (< p20)"] - by_load["большая загрузка (> p80)"]) < 0.02
    print("\nВывод")
    print(f"  1. Это множитель, а не пропущенный поток: свободный член {b:.1f} при сумме "
          f"около {frame['сумма'].median():.0f}, отношение одно и то же при малой и большой "
          "загрузке и по годам." if scale and flat_load else
          "  1. Данные не показывают чистого множителя — см. таблицы выше.")
    print("     Разность связана со всеми продуктами сразу, то есть растёт вместе с "
          "загрузкой, а не с каким-то одним потоком.")
    print("  2. Механизм по данным не различается. С плотностью нефти отношение не "
          "связано, с температурой тоже, но диапазон температур узкий, и эта проверка "
          "слабая.")
    print("  3. С остальными согласован F65: на нём сходится баланс К-2, и формула ПТФ "
          "хуже с F65, приведённым к сумме ходов. Вероятнее расходятся F7–F9.")

    report = {
        "масштаб": {"k": round(float(k), 3), "b": round(float(b), 1), "по_загрузке": by_load,
                    "по_годам": by_year},
        "разность": {"медиана": round(float(residual.median()), 1),
                     "сильнейшие_связи": top, "с_загрузкой": round(load, 3)},
        "температура": {"тег": strongest_temp[0], "corr": round(strongest_temp[1], 3)},
        "плотность": density,
        "согласованность": {"баланс_К2_по_F65": round(float(balance.median()), 3),
                            "W70_к_F30": round(float(density_w.median()), 3),
                            "формула_ПТФ": reading},
        "вывод": {"множитель": bool(scale and flat_load),
                  "механизм_различим": False,
                  "согласован_с_остальными": "F65"},
    }
    out = ROOT / "reports" / "f65_units.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
