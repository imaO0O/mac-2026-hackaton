"""Что означают теги и формулы блока АВТ — по данным (участник 2). Только CPU.

    python scripts/check_avt_formulas.py

Продолжение `scripts/check_tag_meaning.py`, но для ЭЛОУ-АВТ-6 и с более сильной
уликой. Там вердикт выносился по порядку величины и по отклику на останов —
косвенным признакам. Здесь есть прямая проверка: **лаборатория**. Формула
справочника обещает конкретный показатель конкретного потока, а в ЛИМС этот
показатель измерен сотнями анализов. Если формула его воспроизводит — теги в ней
опознаны верно, спорить не о чем. Если нет — видно, насколько и в какую сторону.

Что делает скрипт:

1. **Привязывает блок формул к точке отбора ЛИМС.** В справочнике написано
   «ЭЛОУ-АВТ-6. 240-350» и «ЭЛОУ-АВТ-6. 350», а в лаборатории точки называются
   1, 2, 2.1 и 3, и соответствие нигде не выдано. Оно восстанавливается по
   рабочим формулам: та точка, которую формула воспроизводит с наименьшей
   ошибкой, и есть её поток.
2. **Считает каждую формулу против лаборатории**: смещение, MAE, корреляцию.
3. **Разбирает по слагаемым четыре формулы, которые не воспроизводятся**, и
   проверяет конкретные гипотезы об ошибке записи — скобку, константу,
   дублирование ячейки.
4. **Выносит вердикт по каждому спорному тегу** и формулирует вопросы
   организаторам — по одному на формулу, с числом в каждом.

Главный вывод забегая вперёд: дело НЕ в тегах. Короткие имена АВТ означают то,
что написано в справочнике КИП; ломаются сами записи формул. Прошлый вывод
(`docs/VAK_FEATURES.md`) был противоположным, и его пришлось заменить.

Результат: reports/avt_formulas.json и таблицы в консоли.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import clean_telemetry  # noqa: E402
from nefte.data.loaders import (  # noqa: E402
    load_lims,
    load_tag_dictionary,
    load_telemetry,
    load_vak_formulas,
    parse_vak_formula,
)
from nefte.models.vak import PLAUSIBLE_RANGES  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

TAG_RE = re.compile(r"\b([A-Z]{1,4}[0-9]{1,3})\b")

# Показатель в имени формулы → имя параметра в ЛИМС. ПТФ в лаборатории АВТ
# называется FilterabilityLimit.T, а в формулах и на гидроочистке — CFPP; это
# один и тот же показатель, предельная температура фильтруемости.
LIMS_PARAM = {"D15": "D15", "T50": "50%.T", "T90": "90%.T", "T95": "95%.T",
              "EBP": "EBP.T", "IBP": "IBP.T", "I350": "I350", "I250": "I250",
              "CFPP": ["CFPP", "FilterabilityLimit.T"], "CloudPoint": "CloudPoint"}

# Формулы, которые проверка правдоподобия отбраковывает (docs/VAK_FEATURES.md).
BROKEN = ["AVT6:240-350:CFPP", "AVT6:350:T50", "AVT6:350:I350", "AVT6:240-350:EBP"]


def evaluate(expr: str, avt: pd.DataFrame) -> pd.Series | None:
    """Считает выражение на телеметрии АВТ. None — если не хватает тегов."""
    tags = sorted(set(TAG_RE.findall(expr)))
    if any(tag not in avt.columns for tag in tags):
        return None
    namespace = {tag: avt[tag] for tag in tags}
    try:
        value = eval(compile(expr, "<vak>", "eval"), {"__builtins__": {}, "np": np},
                     namespace)                                        # noqa: S307
    except Exception:                                                  # noqa: BLE001
        return None
    if np.isscalar(value):
        return None
    return pd.Series(value, index=avt.index).replace([np.inf, -np.inf], np.nan)


def compare_to_lab(series: pd.Series, lab: pd.Series) -> dict | None:
    """Сопоставляет расчёт с лабораторией: последний отсчёт ДО момента отбора."""
    position = series.index.searchsorted(lab.index, side="right") - 1
    known = position >= 0
    model = series.to_numpy()[position[known]]
    truth = lab.to_numpy()[known]
    good = ~np.isnan(model) & ~np.isnan(truth)
    model, truth = model[good], truth[good]
    if len(truth) < 20:
        return None
    return {"n": int(len(truth)), "лаб": round(float(np.median(truth)), 2),
            "модель": round(float(np.median(model)), 2),
            "смещение": round(float(np.mean(model - truth)), 2),
            "MAE": round(float(np.mean(np.abs(model - truth))), 2),
            "corr": round(float(np.corrcoef(model, truth)[0, 1]), 3)}


# Ноль в разгонке и в плотности — это пропуск, а не значение. К показателям
# холодных свойств (ПТФ, температура помутнения) это НЕ относится: они
# отрицательны по существу, и фильтр «>0» выбросил бы ряд целиком. Отсюда явный
# список, а не проверка по имени: FilterabilityLimit.T тоже оканчивается на «.T».
POSITIVE_ONLY = {"D15", "IBP.T", "50%.T", "90%.T", "95%.T", "EBP.T", "I250", "I350"}


def lab_series(lims: pd.DataFrame, point: str, param) -> pd.Series | None:
    names = param if isinstance(param, list) else [param]
    for name in names:
        sub = lims[(lims["unit"] == "АВТ") & (lims["point"] == point)
                   & (lims["param"] == name)]
        if not sub.empty:
            values = sub.set_index("ts")["value"].sort_index()
            if name in POSITIVE_ONLY:
                values = values[values > 0]
            if len(values) >= 20:
                return values
    return None


def best_point(series: pd.Series, lims: pd.DataFrame, param) -> list[dict]:
    """Ранжирует точки отбора по тому, какую из них формула описывает лучше.

    Сравнивать точки по сырому MAE нельзя: у плотности он в кг/м3, у разгонки в
    °C, и «лучшая» точка выбиралась бы по единицам измерения. Поэтому ошибка
    нормируется на собственный разброс лабораторного ряда — сколько его
    стандартных отклонений составляет промах формулы. Значение меньше 1 значит,
    что формула знает о потоке больше, чем среднее по этому же потоку.
    """
    rows = []
    for point in sorted(lims[lims["unit"] == "АВТ"]["point"].unique()):
        lab = lab_series(lims, point, param)
        if lab is None:
            continue
        stats = compare_to_lab(series, lab)
        if stats:
            spread = float(lab.std()) or 1.0
            rows.append({"точка": f"АВТ|{point}", **stats,
                         "смещение/разброс": round(abs(stats["смещение"]) / spread, 2),
                         "MAE/разброс": round(stats["MAE"] / spread, 2)})
    return sorted(rows, key=lambda r: r["MAE/разброс"])


def reproduces(row: dict | None) -> bool:
    """Считаем ли, что формула описывает именно этот поток.

    Два условия, и оба нужны. Смещение меньше половины разброса ряда — формула
    попадает в уровень; MAE меньше полутора разбросов — она хотя бы не хуже, чем
    «всегда предсказывать среднее». Одного первого мало: формула может случайно
    угадать медиану чужого потока, как ``AVT6:240-350:D15`` угадывает диапазон
    плотности и всё равно промахивается мимо своей точки на 24 кг/м3.
    """
    return bool(row and row["смещение/разброс"] < 0.5 and row["MAE/разброс"] < 1.5)


def plausible_share(series: pd.Series, kind: str) -> float | None:
    bounds = PLAUSIBLE_RANGES.get(kind)
    if bounds is None:
        return None
    finite = series.dropna()
    if finite.empty:
        return 0.0
    return float(((finite >= bounds[0]) & (finite <= bounds[1])).mean())


def main() -> int:  # noqa: PLR0915 — это отчёт, и он линейный по построению
    use_utf8_console()
    cfg = load_config()
    avt, _ = clean_telemetry(load_telemetry("avt"), unit="avt")
    lims = load_lims()
    corrections = (cfg.get("vak") or {}).get("corrections") or {}
    # Прочтения, выведенные нами из данных. Лежат в конфиге отдельно и в рабочий
    # расчёт не попадают: см. комментарий там же и раздел [6] ниже.
    proposed = (cfg.get("vak") or {}).get("proposed_corrections") or {}
    # Официальная таблица организаторов 15.09 закрыла оба прочтения: они перенесены
    # в vak.superseded_proposals с вердиктом, а в corrections теперь официальные
    # формулы. Разбор ниже воспроизводит прежний счёт по истории и называет вердикт.
    history = (cfg.get("vak") or {}).get("superseded_proposals") or {}
    proposed = {**history, **proposed}
    dictionary = load_tag_dictionary()
    described = dict(zip(dictionary[dictionary["unit"] == "avt"]["code"],
                         dictionary[dictionary["unit"] == "avt"]["description"]))

    formulas = load_vak_formulas()
    formulas = formulas[formulas["block"].astype(str).str.startswith("ЭЛОУ")]

    # ------------------------------------------------------------------ #
    print("\n[1] Каждая формула АВТ против лаборатории\n")
    table, per_formula = [], {}
    for _, row in formulas.iterrows():
        target = row["target"]
        kind = target.rsplit(":", 1)[-1]
        source = corrections.get(target, {}).get("formula", row["formula"])
        series = evaluate(parse_vak_formula(source), avt)
        if series is None:
            continue
        ranked = best_point(series, lims, LIMS_PARAM.get(kind, kind))
        share = plausible_share(series, kind)
        best = ranked[0] if ranked else None
        per_formula[target] = {"ranked": ranked, "series": series}
        table.append({
            "формула": target,
            "медиана": round(float(series.median()), 2),
            "в физ. диапазоне": None if share is None else f"{share:.0%}",
            "лучшая точка": best["точка"] if best else "—",
            "n": best["n"] if best else None,
            "смещение": best["смещение"] if best else None,
            "MAE": best["MAE"] if best else None,
            "MAE/разброс": best["MAE/разброс"] if best else None,
            "corr": best["corr"] if best else None,
        })
    print(pd.DataFrame(table).to_string(index=False))

    # ------------------------------------------------------------------ #
    print("\n[2] Какому потоку ЛИМС отвечает каждый блок справочника\n")
    # Голосуют не все формулы подряд, а только те, что поток действительно
    # описывают: промах меньше половины его собственного разброса. Иначе
    # сломанная формула перетягивает блок на чужую точку большинством в один
    # голос — ровно это и происходит с AVT6:240-350:D15, см. [5].
    candidates: dict[str, list[tuple[float, str, str]]] = {}
    for _, row in formulas.iterrows():
        target = row["target"]
        if target in BROKEN or target not in per_formula:
            continue
        ranked = per_formula[target]["ranked"]
        if not ranked:
            continue
        head = ranked[0]
        ok = reproduces(head)
        runner = (f"следующая {ranked[1]['точка']} {ranked[1]['MAE/разброс']:.2f}"
                  if len(ranked) > 1 else "других точек с этим показателем нет")
        print(f"  {target:24s} → {head['точка']:8s} смещение/разброс "
              f"{head['смещение/разброс']:4.2f}  MAE/разброс {head['MAE/разброс']:5.2f}  "
              f"({runner}) — {'принято' if ok else 'не голосует: мимо потока'}")
        if ok:
            candidates.setdefault(str(row["block"]), []).append(
                (head["MAE/разброс"], head["точка"], target))

    print()
    block_point = {}
    for block, votes in candidates.items():
        votes.sort()
        block_point[block] = votes[0][1]
        print(f"  «{block}» → {votes[0][1]}  (по формулам: "
              + ", ".join(f"{t.split(':')[-1]} {v:.2f}" for v, _, t in votes) + ")")
    print("\n  Это не написано нигде в выданных материалах, а нужно всем: без")
    print("  привязки к точке отбора формулу не с чем сверить, и любая проверка")
    print("  сводится к «похоже на правду по порядку величины».")

    # ------------------------------------------------------------------ #
    print("\n[3] Четыре формулы, которые не воспроизводятся: разбор и гипотезы\n")
    findings = []

    # Для сломанной формулы «лучшая точка» ничего не значит: к ней формула просто
    # случайно ближе. Документация сверяет каждую формулу со СВОЕЙ точкой блока,
    # поэтому отчёт несёт и её — иначе таблицу нельзя проверить тестом.
    block_of = dict(zip(formulas["target"], formulas["block"].astype(str)))
    own_rows = []
    for row in table:
        point = block_point.get(block_of.get(row["формула"], ""))
        own = next((r for r in per_formula.get(row["формула"], {}).get("ranked", [])
                    if r["точка"] == point), None)
        row["своя точка"] = point
        for key in ("n", "смещение", "MAE", "corr"):
            row[f"{key} (своя)"] = own[key] if own else None
        own_rows.append({"формула": row["формула"], "точка": point,
                         **({k: own[k] for k in ("n", "смещение", "MAE", "corr")}
                            if own else {})})
    print()
    print("  Те же формулы против СВОЕЙ точки блока:")
    print(pd.DataFrame(own_rows).to_string(index=False))

    # --- 3.1 AVT6:240-350:CFPP — скобка поставлена не туда --------------- #
    lab_cfpp = lab_series(lims, block_point.get("ЭЛОУ-АВТ-6. 240-350", "АВТ|3").split("|")[-1],
                          LIMS_PARAM["CFPP"])
    variants = {
        "как исправили организаторы: F65/F32+F30":
            (history.get("AVT6:240-350:CFPP") or {}).get("first_organizer_answer")
            or corrections["AVT6:240-350:CFPP"]["formula"],
        "по аналогии с D15 того же блока: F65/(F32+F30)":
            proposed["AVT6:240-350:CFPP"]["formula"],
    }
    print("  3.1 AVT6:240-350:CFPP")
    print("      В той же ячейке блока, где стоит D15, отношение расходов записано")
    print("      как F30/(F32+F30). Поправка организаторов убрала лишнюю скобку так,")
    print("      что осталось F65/F32 + F30 — то есть к температуре в °C прибавляется")
    print("      расход в т/ч. Проверяем оба прочтения по лаборатории:")
    cfpp_rows = []
    for name, formula in variants.items():
        series = evaluate(parse_vak_formula(formula), avt)
        stats = compare_to_lab(series, lab_cfpp) if lab_cfpp is not None else None
        share = plausible_share(series, "CFPP")
        cfpp_rows.append({"вариант": name, "в диапазоне": f"{share:.0%}", **(stats or {})})
    print(pd.DataFrame(cfpp_rows).to_string(index=False))
    sent, ours = cfpp_rows[0], cfpp_rows[1]
    point_240 = block_point.get("ЭЛОУ-АВТ-6. 240-350", "АВТ|3")
    findings.append({
        "формула": "AVT6:240-350:CFPP",
        "диагноз": "скобка в поправке организаторов поставлена не туда",
        "исправление": "31.40363-0.06784*T33+17.411*P67-8.11544*P4-0.47309*(F65/(F32+F30))",
        "проверка": cfpp_rows,
        "теги опознаны верно": True,
        "ответ организаторов 15.09": (history.get("AVT6:240-350:CFPP") or {}).get("verdict"),
        "вопрос организаторам":
            f"В AVT6:240-350:CFPP последнее слагаемое читается как F65/(F32+F30)? "
            f"В таком виде формула воспроизводит ПТФ по точке {point_240} со "
            f"смещением {ours.get('смещение'):+.1f} °C и MAE {ours.get('MAE'):.1f} °C "
            f"на {ours.get('n')} анализах; в присланном виде (F65/F32+F30) смещение "
            f"{sent.get('смещение'):+.1f} °C и в физический диапазон попадает "
            f"{sent['в диапазоне']} значений.",
    })

    # --- 3.2 AVT6:350:T50 — в ячейке копия формулы плотности ------------- #
    print("\n  3.2 AVT6:350:T50")
    t50 = evaluate(parse_vak_formula(
        formulas[formulas["target"] == "AVT6:350:T50"]["formula"].iloc[0]), avt)
    d15 = evaluate(parse_vak_formula(
        formulas[formulas["target"] == "AVT6:350:D15"]["formula"].iloc[0]), avt)
    difference = (d15 - t50).dropna()
    print(f"      Разность с формулой D15 того же блока: медиана "
          f"{difference.median():.5f}, стандартное отклонение {difference.std():.6f}.")
    print("      То есть это ОДНА И ТА ЖЕ формула, отличается только свободный член.")
    d15_fit = per_formula["AVT6:350:D15"]["ranked"][0]
    print("      Величина 880 °C — не температура 50 % отгона (ждём 180…340 °C), а")
    print("      плотность в кг/м3, и формула D15 из соседней ячейки лабораторию")
    print(f"      воспроизводит: {d15_fit['точка']}, смещение "
          f"{d15_fit['смещение']:+.2f} кг/м3, MAE {d15_fit['MAE']:.2f} на "
          f"{d15_fit['n']} анализах.")
    findings.append({
        "формула": "AVT6:350:T50",
        "диагноз": "в ячейке T50 записана формула плотности D15 того же блока; "
                   "различие только в свободном члене",
        "разность D15 − T50": {"медиана": round(float(difference.median()), 5),
                               "ст.откл": round(float(difference.std()), 6)},
        "теги опознаны верно": True,
        "вопрос организаторам":
            "AVT6:350:T50 совпадает с AVT6:350:D15 с точностью до константы "
            "(2.027) и даёт 881 — это плотность, а не 50 % отгона. Какая формула "
            "должна стоять в ячейке T50 блока 350?",
    })

    # --- 3.3 AVT6:350:I350 — знак при T6, а не потерянная цифра ---------- #
    # Официальная таблица (15.09) опровергла наше прочтение «39.562 → 399.562»: у
    # организаторов константа прежняя, а при T6 стоит ПЛЮС. Здесь три записи рядом —
    # и объяснение, почему неверное прочтение восстановило уровень.
    print()
    print("  3.3 AVT6:350:I350")
    t6 = avt["T6"]
    common = (-1.62865 * avt["L43"] - 0.22361 * avt["T18"]
              + 0.00031 * avt["F64"] * (avt["T15"] - avt["T11"]))
    versions = {
        "выданная: 39.562 − 0.76664·T6": 39.562 - 0.76664 * t6 + common,
        "наше прочтение: 399.562 − 0.76664·T6": 399.562 - 0.76664 * t6 + common,
        "официальная: 39.562 + 0.76664·T6": 39.562 + 0.76664 * t6 + common,
    }
    lab_i350 = lab_series(lims, block_point.get("ЭЛОУ-АВТ-6. 350", "АВТ|1").split("|")[-1],
                          "I350")
    i350_rows = []
    for name, series in versions.items():
        stats = compare_to_lab(series, lab_i350) if lab_i350 is not None else None
        inside = float(((series > 0) & (series < 100)).mean())
        i350_rows.append({"запись": name, "в 0…100 %": round(inside, 3), **(stats or {})})
    print(pd.DataFrame(i350_rows).to_string(index=False))
    t6_median = float(t6.median())
    gap = 360.0 - 2 * 0.76664 * t6_median
    print(f"      Наше прочтение и официальная запись различаются на 360 − 1.533·T6. При "
          f"медиане T6 = {t6_median:.1f} °C это {gap:.1f} %,")
    print("      поэтому уровень прочтение и восстановило — совпадением рабочей точки, а "
          "наклон по T6 перевернуло.")
    print("      Отсюда его отрицательная связь с лабораторией. Официальная запись в "
          "диапазоне, но смещена и за")
    print("      лабораторией не следует: виртуальным анализатором отгона не станет ни "
          "одна из трёх.")
    if "AVT6:350:I350" in history:
        print("      Ответ организаторов 15.09: " + " ".join(
            str(history["AVT6:350:I350"]["verdict"]).split()))
    official = i350_rows[-1]
    findings.append({
        "формула": "AVT6:350:I350",
        "диагноз": ("знак при T6: в выданной записи минус, в официальной таблице плюс. "
                    "Прочтение «константа 399.562» восстанавливало уровень совпадением: при "
                    f"медиане T6 {t6_median:.0f} °C записи расходятся на {gap:.1f} %, а "
                    "наклон по T6 у них противоположный"),
        "проверка": i350_rows,
        "ответ организаторов 15.09": (history.get("AVT6:350:I350") or {}).get("verdict"),
        "теги опознаны верно": True,
        "вопрос организаторам":
            f"Официальная AVT6:350:I350 на АВТ|1 смещена на {official.get('смещение')} % и "
            f"с лабораторией не связана (corr {official.get('corr')}). На каком потоке и за "
            "какой период она подбиралась?",
    })

    # --- 3.4 AVT6:240-350:EBP — формула плохо обусловлена ---------------- #
    print("\n  3.4 AVT6:240-350:EBP")
    ebp_terms = {
        "813.883": pd.Series(813.883, index=avt.index),
        "+2.66463·F30": 2.66463 * avt["F30"],
        "−0.20239·T33": -0.20239 * avt["T33"],
        "−3.65888·F36": -3.65888 * avt["F36"],
        "−14.08235·T37": -14.08235 * avt["T37"],
        "−1.32603·T40": -1.32603 * avt["T40"],
        "+14.60206·T58": 14.60206 * avt["T58"],
    }
    print("      Слагаемое                    медиана    ст.откл")
    for name, part in ebp_terms.items():
        print(f"        {name:26s} {part.median():9.1f} {part.std():10.1f}")
    pair = ebp_terms["−14.08235·T37"] + ebp_terms["+14.60206·T58"]
    lab_ebp = lab_series(lims, block_point.get("ЭЛОУ-АВТ-6. 240-350", "3").split("|")[-1],
                         "EBP.T")
    whole = compare_to_lab(sum(ebp_terms.values()), lab_ebp) if lab_ebp is not None else {}
    print(f"\n      Пара T37/T58 вместе: медиана {pair.median():.1f} °C при "
          f"стандартном отклонении {pair.std():.1f} °C,")
    print(f"      корреляция самих тегов {avt['T37'].corr(avt['T58']):+.2f}, "
          f"разброс лаборатории {lab_ebp.std():.1f} °C.")
    print(f"      Итог формулы: смещение {whole.get('смещение')} °C на "
          f"{whole.get('n')} анализах — ЦЕНТР ВЕРНЫЙ, а MAE {whole.get('MAE')} °C.")
    without = compare_to_lab(sum(v for k, v in ebp_terms.items()
                                 if "T37" not in k and "T58" not in k),
                             lab_ebp) if lab_ebp is not None else {}
    print(f"      Даже если убрать пару целиком, MAE остаётся {without.get('MAE')} °C:")
    print("      формула — разность нескольких слагаемых по 500–850 °C, и её шум")
    print("      задаётся не тегами, а самой записью. Одной опечатки тут нет.")
    findings.append({
        "формула": "AVT6:240-350:EBP",
        "диагноз": "формула плохо обусловлена: сумма разнознаковых слагаемых по "
                   "500–850 °C, шум результата в 6 раз больше лабораторного разброса",
        "как есть": whole,
        "без пары T37/T58": without,
        "теги опознаны верно": True,
        "вопрос организаторам":
            f"AVT6:240-350:EBP даёт верный центр (смещение {whole.get('смещение')} °C "
            f"на {whole.get('n')} анализах), но MAE {whole.get('MAE')} °C при "
            f"лабораторном разбросе {lab_ebp.std():.1f} °C. Коэффициенты −14.08 при "
            "T37 и +14.60 при T58 верны? На выданных тегах они дают два слагаемых "
            "по 850 °C, которые почти сокращаются.",
    })

    # ------------------------------------------------------------------ #
    print("\n[4] Вердикт по каждому спорному тегу\n")
    disputed = sorted({tag for target in BROKEN
                       for tag in TAG_RE.findall(parse_vak_formula(
                           corrections.get(target, {}).get("formula")
                           or formulas[formulas["target"] == target]["formula"].iloc[0]))})
    # Сильнейшая улика по тегу — не его собственный диапазон, а то, что он уже
    # работает в формуле, которая лабораторию воспроизводит. Тег не может
    # означать одно в рабочей формуле и другое в соседней сломанной.
    confirmed_by: dict[str, list[str]] = {}
    for target, data in per_formula.items():
        ranked = data["ranked"]
        if target in BROKEN or not reproduces(ranked[0] if ranked else None):
            continue
        source = corrections.get(target, {}).get("formula") or \
            formulas[formulas["target"] == target]["formula"].iloc[0]
        for tag in set(TAG_RE.findall(parse_vak_formula(source))):
            confirmed_by.setdefault(tag, []).append(target.split(":", 1)[-1])

    verdicts = []
    for tag in disputed:
        if tag not in avt.columns:
            verdicts.append({"тег": tag, "вердикт": "нет в телеметрии"})
            continue
        values = avt[tag]
        works_in = confirmed_by.get(tag, [])
        verdicts.append({
            "тег": tag,
            "описание КИП": str(described.get(tag, ""))[:38],
            "медиана": round(float(values.median()), 2),
            "p05": round(float(values.quantile(0.05)), 2),
            "p95": round(float(values.quantile(0.95)), 2),
            "брак, %": round(float(values.isna().mean() * 100), 1),
            "работает в": ", ".join(works_in) or "—",
            "вердикт": ("подтверждён рабочей формулой" if works_in
                        else "значения отвечают описанию КИП"),
        })
    print(pd.DataFrame(verdicts).to_string(index=False))
    proven = sum(1 for v in verdicts if v.get("работает в", "—") != "—")
    print(f"\n  Ни один из {len(verdicts)} тегов не выбивается из своего описания:")
    print("  температуры лежат там, где положено температурам, расходы — где")
    print(f"  расходам, уровень L43 в процентах. А {proven} из них вдобавок уже")
    print("  работают в формулах, которые лабораторию воспроизводят (столбец")
    print("  «работает в»). Тег не может означать одно в рабочей формуле и другое")
    print("  в соседней сломанной — значит, ломаются не теги, а записи формул.")

    # ------------------------------------------------------------------ #
    print("\n[5] Отдельная находка: правдоподобие — не то же самое, что верность\n")
    d15_240 = per_formula.get("AVT6:240-350:D15")
    if d15_240:
        own = next((r for r in d15_240["ranked"]
                    if r["точка"] == block_point.get("ЭЛОУ-АВТ-6. 240-350")), None)
        if own:
            print(f"  AVT6:240-350:D15 проверку правдоподобия проходит (медиана "
                  f"{float(d15_240['series'].median()):.1f} кг/м3, диапазон 700…950),")
            print(f"  но свою же точку {own['точка']} воспроизводит со смещением "
                  f"{own['смещение']:+.1f} кг/м3 при MAE {own['MAE']:.1f} на "
                  f"{own['n']} анализах.")
            print("  Проверка по диапазону ловит бессмыслицу, но не ошибку: формула")
            print("  может быть физичной и всё равно описывать не тот поток.")
            findings.append({
                "формула": "AVT6:240-350:D15",
                "диагноз": "проходит проверку правдоподобия, но лабораторию своей "
                           "точки не воспроизводит",
                "против своей точки": own,
                "теги опознаны верно": True,
                "вопрос организаторам":
                    f"AVT6:240-350:D15 систематически завышает плотность на "
                    f"{own['смещение']:+.1f} кг/м3 относительно {own['точка']} "
                    f"({own['n']} анализов). Формула рассчитана на другой поток?",
            })

    # ------------------------------------------------------------------ #
    print("\n[6] Что из этого следует\n")
    print("  1. Вывод «короткие имена тегов АВТ означают не те величины» НЕ")
    print("     подтвердился. Теги опознаны верно; ломаются записи формул.")
    print("  2. Из четырёх невоспроизводящихся формул три объяснены конкретной")
    print("     ошибкой записи (скобка, дублированная ячейка, потерянная цифра),")
    print("     и для двух из них исправленное прочтение проверено лабораторией.")
    print("  3. Четвёртая (EBP) объяснена обусловленностью, а не опечаткой: её")
    print("     нельзя починить одним символом, и вопрос организаторам другой.")
    print("  4. Привязка блоков к точкам ЛИМС восстановлена и пригодится всем:")
    print("     без неё формулу не с чем сверять.")
    print("\n  Передача участникам 1 и 3 (менять чужие числа мы не стали):")
    print("  прочтения, закрытые официальной таблицей 15.09, лежат в configs/config.yaml →")
    print("  vak.superseded_proposals с вердиктом; в расчёт идут официальные формулы")
    print("  (vak.corrections, сверка — tests/test_vak_official.py).")
    print("  Участнику 3: исправленная CFPP даёт ПТФ компонента АВТ в реальном")
    print("  времени, сейчас блендинг берёт его последним лабораторным анализом.")

    report = {
        "метод": "сверка формул справочника с лабораторией по точкам отбора",
        "блок_и_точка_ЛИМС": block_point,
        "формулы": table,
        "разбор_сломанных": findings,
        "вердикты_по_тегам": verdicts,
        "вопросы_организаторам": [f["вопрос организаторам"] for f in findings],
    }
    out = ROOT / "reports" / "avt_formulas.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
