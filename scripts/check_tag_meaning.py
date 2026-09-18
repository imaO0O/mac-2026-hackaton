"""Новая таблица тегов 24-2000 против данных: все 26 тегов (участник 2).

    python scripts/check_tag_meaning.py

История. Лист «КИП» из выданного пакета на 24-2000 оказался перемешан со строками:
по смыслу на своём коде стоят 6 описаний из 26. Прежняя версия этого скрипта
разбирала 14 тегов «под вопросом» и нашла, что буквы кодов верны, а описания нет
(`T11` — температура, `P8` — не температура ГСС, `F19` — масса потока ДТ). 15.09
организаторы прислали новую таблицу (`configs/tags_2026-09-15.csv`, читается по
умолчанию), и вопрос сменился: **сходится ли с данными уже она**.

На каждый тег — три проверки, все на работающей установке:

1. **диапазон под заявленную величину**: температура, давление, перепад, расход
   жидкости или газа, сера. Не меньше 95 % значений — в физическом диапазоне;
2. **заглушки**: доля самого частого значения. На этих данных уже встречались
   307, 313, 240 и 252 — значения, которые прибор держит, когда не мерит;
3. **двойники**: все пары тегов, где одна величина записана под двумя кодами
   (corr ≥ 0.97 и устойчивое отношение). У двойника описания обязаны говорить об
   одном потоке, а отношение массы к объёму — быть плотностью.

И четыре проверки выполнимости, которые описания должны проходить вместе:
масса сырья не меньше массы продукта, объём сырья порядка объёма продукта,
плотность газа поддува не меньше, чем у водорода, и уровень поточных анализаторов
серы — как у лаборатории на своём потоке.

Результат: reports/tag_meaning.json — вердикт по каждому тегу и вопросы
организаторам по тому, что не сошлось.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import clean_lims_sulfur  # noqa: E402
from nefte.data.loaders import lims_series, load_tag_dictionary, load_telemetry  # noqa: E402
from nefte.models.regime import FEED, outage_mask  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

TABLE = "configs/tags_2026-09-15.csv"

# Физический диапазон под заявленную величину (на работающей установке 24-2000).
# Объёмный расход жидкости ограничен сверху мощностью установки: сырья около
# 260 м³/ч, так что тысяча — уже не жидкость.
RANGES = {
    "температура, °C": (-40.0, 500.0), "°C": (-40.0, 500.0),
    "давление, МПа": (0.0, 10.0), "перепад, МПа": (0.0, 1.0),
    "расход, м³/ч": (0.0, 1000.0), "расход, т/ч": (0.0, 1000.0),
    "масс.расход, т/ч": (0.0, 1000.0), "масс.расход газа, т/ч": (0.0, 100.0),
    "расход газа, нм³/ч": (0.0, 500000.0), "расход ВСГ, нм³/ч": (0.0, 500000.0),
    "сера, ppm": (0.0, 30000.0),
}
IN_RANGE_MIN = 0.95
STUB_SHARE = 0.20            # самое частое значение чаще — прибор не мерит
TWIN_CORR = 0.97
TWIN_WIDTH = 0.10            # ширина p05–p95 отношения в долях медианы
LIQUID_DENSITY = (0.60, 0.95)
H2_DENSITY_KG_NM3 = 0.0899   # легче водорода газа не бывает

# Поток по описанию — для сверки двойников: у одной величины он один.
STREAMS = (("сырь", "сырьё"), ("гидроочищ", "ГО ДТ"), ("бензин", "бензин"),
           ("газ поддува", "газ поддува"), ("всг", "ВСГ"), ("квенч", "квенч"))
REPORT = ROOT / "reports" / "tag_meaning.json"


def stream(description: str) -> str | None:
    text = description.lower()
    return next((name for key, name in STREAMS if key in text), None)


def per_tag(series: pd.Series, quantity: str) -> dict:
    lo, hi = RANGES.get(quantity, (-np.inf, np.inf))
    s = series.dropna()
    top_share = float(s.round(3).value_counts(normalize=True).iloc[0]) if len(s) else 0.0
    return {"p01": round(float(s.quantile(.01)), 3), "медиана": round(float(s.median()), 3),
            "p99": round(float(s.quantile(.99)), 3),
            "в диапазоне": round(float(s.between(lo, hi).mean()), 3),
            "доля самого частого значения": round(top_share, 3)}


def find_twins(hourly: pd.DataFrame, tags: pd.DataFrame) -> list[dict]:
    described = dict(zip(tags["code"], tags["description"]))
    codes = [c for c in tags["code"] if c in hourly.columns]
    twins = []
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            pair = hourly[[a, b]].dropna()
            pair = pair[(pair[a] > 0) & (pair[b] > 0)]
            if len(pair) < 1000:
                continue
            corr = float(pair[a].corr(pair[b]))
            if corr < TWIN_CORR:
                continue
            ratio = pair[a] / pair[b]
            width = float((ratio.quantile(.95) - ratio.quantile(.05)) / ratio.median())
            if width > TWIN_WIDTH:
                continue
            twins.append({"пара": f"{a}~{b}", "отношение": round(float(ratio.median()), 4),
                          "ширина": round(width, 4), "corr": round(corr, 3),
                          "описания": [str(described[a]), str(described[b])]})
    return twins


def twin_problems(twin: dict, quantity: dict) -> list[tuple[str, list[str]]]:
    a, b = twin["пара"].split("~")
    da, db = twin["описания"]
    sa, sb = stream(da), stream(db)
    problems = []
    if sa and sb and sa != sb:
        problems.append((f"двойник {twin['пара']} (отношение {twin['отношение']}), а "
                         f"описания называют разные потоки: «{sa}» и «{sb}»", [a, b]))
    mass = {t for t in (a, b) if "масс" in quantity[t] or "т/ч" in quantity[t]}
    volume = {t for t in (a, b) if "м³/ч" in quantity[t] or "м3/ч" in quantity[t]}
    if len(mass) == 1 and len(volume) == 1 and sa == sb:
        m, v = mass.pop(), volume.pop()
        density = twin["отношение"] if m == a else 1.0 / twin["отношение"]
        if not LIQUID_DENSITY[0] <= density <= LIQUID_DENSITY[1]:
            problems.append((f"{m} (масса) и {v} (объём) одного потока, а отношение "
                             f"{density:.3f} — не плотность жидкости", [m, v]))
    if abs(twin["отношение"] - round(twin["отношение"], 2)) < 1e-4 and twin["ширина"] < 0.01:
        problems.append((f"{twin['пара']}: отношение {twin['отношение']:.4f} постоянно до "
                         "четвёртого знака — один тег вычислен из другого, это не два прибора",
                         [a, b]))
    return problems


def feasibility(work: pd.DataFrame, cfg: dict) -> list[dict]:
    med = work.median()
    checks = [
        {"проверка": "масса сырья F9 не меньше массы ГО ДТ F17",
         "значения": {"F9": round(float(med["F9"]), 1), "F17": round(float(med["F17"]), 1)},
         "сходится": bool(med["F9"] >= 0.98 * med["F17"]), "теги": ["F9", "F17"]},
        {"проверка": "объём сырья F15 порядка объёма ГО ДТ F26",
         "значения": {"F15": round(float(med["F15"]), 1), "F26": round(float(med["F26"]), 1)},
         "сходится": bool(0.8 <= med["F15"] / med["F26"] <= 1.3), "теги": ["F15"]},
    ]
    gas = 1000.0 * work["W7"] / work["F22"]
    checks.append({"проверка": "плотность газа поддува W7/F22 не меньше, чем у водорода",
                   "значения": {"кг/нм³": round(float(gas.median()), 4),
                                "водород": H2_DENSITY_KG_NM3},
                   "сходится": bool(gas.median() >= H2_DENSITY_KG_NM3),
                   "теги": ["W7", "F22"]})
    product = clean_lims_sulfur(lims_series(cfg["quality"]["target"]["lims_source"]))
    checks.append({"проверка": "Q21 (сера в г/о ДТ) на уровне лабораторной серы продукта",
                   "значения": {"Q21": round(float(med["Q21"]), 2),
                                "ЛИМС": round(float(product.median()), 2)},
                   "сходится": bool(0.5 <= med["Q21"] / product.median() <= 2.0),
                   "теги": ["Q21"]})
    feed = lims_series("Гидроочистка|1|Mass.Sulfur").dropna()
    feed = feed[feed > 0]
    if len(feed):
        # в ЛИМС сера сырья в % масс., у анализатора ppm: 1 % = 10 000 ppm
        checks.append({"проверка": "Q20 на уровне лабораторной серы сырья",
                       "значения": {"Q20, ppm": round(float(med["Q20"]), 0),
                                    "ЛИМС сырьё, ppm": round(float(feed.median() * 1e4), 0)},
                       "сходится": bool(0.5 <= med["Q20"] / (feed.median() * 1e4) <= 2.0),
                       "теги": ["Q20"]})
    return checks


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    tags = load_tag_dictionary()
    tags = tags[tags["unit"] == "ht"].reset_index(drop=True)
    quantity = dict(zip(tags["code"], tags["quantity"]))

    ht = load_telemetry("ht")
    down = outage_mask(ht[FEED], retrospective=True)
    work = ht[~down & (ht[FEED] > 0.5 * ht[FEED].median())]
    hourly = work.resample("1h").mean()

    rows = {}
    for _, row in tags.iterrows():
        code = row["code"]
        if code not in ht.columns:
            rows[code] = {"описание": row["description"], "величина": row["quantity"],
                          "проблемы": ["тега нет в телеметрии"]}
            continue
        stats = per_tag(work[code], row["quantity"])
        problems = []
        if stats["в диапазоне"] < IN_RANGE_MIN:
            lo, hi = RANGES.get(row["quantity"], (None, None))
            problems.append(f"в диапазоне «{row['quantity']}» ({lo}…{hi}) только "
                            f"{stats['в диапазоне']:.0%} значений, медиана {stats['медиана']}")
        if stats["доля самого частого значения"] > STUB_SHARE:
            problems.append(f"одно значение в {stats['доля самого частого значения']:.0%} "
                            "отсчётов — прибор не мерит")
        rows[code] = {"описание": row["description"], "величина": row["quantity"], **stats,
                      "проблемы": problems}

    twins = find_twins(hourly, tags)
    for twin in twins:
        for problem, codes in twin_problems(twin, quantity):
            for code in codes:
                rows[code]["проблемы"].append(problem)
    checks = feasibility(work, cfg)
    for check in checks:
        if not check["сходится"]:
            for code in check["теги"]:
                rows[code]["проблемы"].append(f"не проходит: {check['проверка']} "
                                              f"({check['значения']})")

    bad = {code: r["проблемы"] for code, r in rows.items() if r["проблемы"]}
    print(f"Новая таблица тегов 24-2000 ({TABLE}) против данных: {len(rows)} тегов\n")
    for code, r in rows.items():
        mark = "НЕ СХОДИТСЯ" if r["проблемы"] else "сходится"
        print(f"  {code:4s} {mark:12s} {r.get('медиана', ''):>10}  {r['величина']:22s} "
              f"{str(r['описание'])[:55]}")
    print("\nДвойники (corr ≥ 0.97, устойчивое отношение):")
    for twin in twins:
        print(f"  {twin['пара']:9s} отношение {twin['отношение']:.4f}, ширина "
              f"{twin['ширина']:.3f}, corr {twin['corr']}")
    print("\nПроверки выполнимости:")
    for check in checks:
        print(f"  {'да ' if check['сходится'] else 'НЕТ'} {check['проверка']}: {check['значения']}")
    print(f"\nСходится {len(rows) - len(bad)} из {len(rows)}. Не сходится:")
    for code, problems in bad.items():
        for problem in dict.fromkeys(problems):
            print(f"  {code}: {problem}")

    REPORT.write_text(json.dumps({
        "таблица": TABLE, "тегов": len(rows), "сходится": len(rows) - len(bad),
        "не_сходится": {code: list(dict.fromkeys(p)) for code, p in bad.items()},
        "теги": rows, "двойники": twins, "выполнимость": checks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
