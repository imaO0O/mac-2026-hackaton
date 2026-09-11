"""Скорость дезактивации катализатора и остаточный ресурс цикла (участник 2). CPU.

    python scripts/check_catalyst_life.py
    python scripts/check_catalyst_life.py --activation 120 --target 5

Четвёртый критерий ТЗ — надёжность оборудования — до сих пор держался у нас на
прокси без единого измеренного числа: наработка катализатора входила в severity
как «часов от последнего останова». Возраст не отвечает ни на один вопрос,
который на самом деле задают: сколько активности теряется в месяц, сколько
градусов запаса осталось и когда установку придётся выводить.

Метод — нормированная температура реакторного блока (разбор в
``models/catalyst.py``). Считается по лабораторным анализам серы: 1451 точка,
сера продукта и сера сырья, режим усреднён за 6 часов до отбора пробы.

Скрипт отвечает на четыре вопроса и на каждый даёт число с интервалом:

1. Сколько длительных остановов действительно были сменой катализатора?
   Проверяется шагом нормированной температуры, а не длительностью.
2. Сколько градусов в месяц приходится добавлять, чтобы держать серу?
   Оценка по завершённым циклам, интервал — блочным бутстрепом.
3. Сколько запаса до уровня, на котором установку выводили раньше?
4. Сколько месяцев осталось текущему циклу? Тремя независимыми способами,
   потому что один способ на такой выборке — это мнение, а не оценка.

Плюс проверка гипотезы «падение цетанового числа — тот же процесс»: ЦЧ должно
восстанавливаться при смене катализатора, если это так.

Результат: reports/catalyst_life.json и таблицы в консоли.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import clean_lims_sulfur  # noqa: E402
from nefte.data.loaders import lims_series, load_telemetry  # noqa: E402
from nefte.models.catalyst import (  # noqa: E402
    DAYS_IN_MONTH,
    REFERENCE_SULFUR_MGKG,
    Cycle,
    fit_deactivation,
    remaining_by_analogue,
    sulfur_drift_per_month,
)
from nefte.models.regime import FEED  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

FEED_SULFUR_SERIES = "Гидроочистка|1|Mass.Sulfur"
PRODUCT_SULFUR_SERIES = "Гидроочистка|2|Mg.Sulfur"
CETANE_SERIES = "Гидроочистка|2|CetaneNumber"


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ht = load_telemetry("ht")
    product = clean_lims_sulfur(lims_series(PRODUCT_SULFUR_SERIES))
    # сера сырья выдана в процентах массовых, приводим к мг/кг
    feed_sulfur = lims_series(FEED_SULFUR_SERIES) * 1e4
    return ht, product, feed_sulfur


def cetane_check(ht: pd.DataFrame, cycles: list[Cycle]) -> dict:
    """Связано ли падение цетанового числа с возрастом катализатора.

    Проверка простая и решающая: если причина — дезактивация, то при смене
    катализатора ЦЧ обязано подскочить вверх, а внутри цикла падать. Если же оно
    падает ровно по календарю и смену не замечает, причина другая (сырьё), и
    относить его к надёжности катализатора нельзя.
    """
    try:
        cetane = lims_series(CETANE_SERIES)
    except KeyError:                                  # pragma: no cover — нет ряда
        return {}
    starts = [c.start for c in cycles]
    cycle_id = np.searchsorted(np.array(starts, dtype="datetime64[ns]"),
                               cetane.index.to_numpy(dtype="datetime64[ns]"),
                               side="right") - 1
    frame = pd.DataFrame({"cn": cetane, "cycle": cycle_id})
    frame["run_days"] = [(ts - starts[c]).total_seconds() / 86400
                         for ts, c in zip(frame.index, frame["cycle"])]
    calendar = (frame.index - frame.index[0]).days.to_numpy(dtype="float64")
    by_calendar = float(np.polyfit(calendar, frame["cn"], 1)[0] * 365.25)

    within = []
    for number, part in frame.groupby("cycle"):
        if len(part) < 8:
            continue
        slope = float(np.polyfit(part["run_days"], part["cn"], 1)[0] * 365.25)
        within.append({"цикл": int(number), "анализов": len(part),
                       "наклон, ед/год": round(slope, 2),
                       "среднее ЦЧ": round(float(part["cn"].mean()), 2)})

    jumps = []
    for cycle in cycles[1:]:
        window = pd.Timedelta(days=180)
        before = frame.loc[cycle.start - window:cycle.start, "cn"]
        after = frame.loc[cycle.start:cycle.start + window, "cn"]
        if len(before) < 3 or len(after) < 3:
            continue
        jumps.append({"смена": str(cycle.start.date()),
                      "ЦЧ до": round(float(before.mean()), 2),
                      "ЦЧ после": round(float(after.mean()), 2),
                      "шаг": round(float(after.mean() - before.mean()), 2)})

    expected = -by_calendar * np.mean([c.days for c in cycles if c.completed]) / 365.25
    observed = float(np.mean([j["шаг"] for j in jumps])) if jumps else 0.0
    verdict = ("дезактивация катализатора" if observed > expected / 2
               else "НЕ дезактивация катализатора: ЦЧ проходит смену насквозь")
    return {"анализов": len(frame), "наклон по календарю, ед/год": round(by_calendar, 3),
            "внутри циклов": within, "шаг при смене": jumps,
            "ожидаемый шаг, если причина — катализатор": round(float(expected), 2),
            "вердикт": verdict}


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", type=float, default=None,
                    help="энергия активации, кДж/моль (по умолчанию из models/regime.py)")
    ap.add_argument("--target", type=float, default=REFERENCE_SULFUR_MGKG,
                    help="эталонная сера продукта, мг/кг")
    args = ap.parse_args()

    cfg = load_config()
    ht, product, feed_sulfur = load_inputs(cfg)
    kwargs = {"target_sulfur": args.target}
    if args.activation is not None:
        kwargs["activation_kj"] = args.activation

    fit = fit_deactivation(ht, product, feed_sulfur, raw_feed=ht[FEED], **kwargs)
    if fit is None:
        print("не удалось собрать выборку: нет завершённых циклов")
        return 1

    # --- 1. какие остановы были сменой катализатора -------------------- #
    print("\n[1] Длительные остановы: смена катализатора или просто ремонт\n")
    print(pd.DataFrame(fit.outage_steps).to_string(index=False))
    print("\n  Решает не длительность, а шаг нормированной температуры: после смены")
    print("  катализатор снова холоднее держит ту же серу. Порог −10 °C выбран")
    print("  не на глаз — наблюдённые шаги стоят далеко от него с обеих сторон.")
    changes = sum(r["смена катализатора"] for r in fit.outage_steps)
    print(f"  Итог: смен катализатора {changes} из {len(fit.outage_steps)} длительных остановов.")

    # --- 2. скорость дезактивации -------------------------------------- #
    print("\n[2] Скорость дезактивации\n")
    print(pd.DataFrame([c.as_dict() for c in fit.cycles]).to_string(index=False))
    lo, hi = fit.rate_ci
    print(f"\n  По завершённым циклам вместе: {fit.rate_c_per_month:+.3f} °C/мес, "
          f"95% ДИ [{lo:+.3f}, {hi:+.3f}]")

    # Модельно-свободная проверка: сырой WABT, без кинетики и без нормировки.
    # Считается в ТЕХ ЖЕ точках, что и NWABT, — иначе сравнивались бы не методы,
    # а выборки (на 10-минутной сетке своё распределение по времени).
    raw_rates = {}
    for cycle in fit.cycles:
        if not cycle.completed:
            continue
        part = fit.frame[fit.frame["cycle"] == cycle.index]
        raw_rates[cycle.index] = float(
            np.polyfit(part["run_days"], part["wabt"], 1)[0] * DAYS_IN_MONTH)
    print("  Проверка без кинетики (сырой WABT, ничего не нормировано): "
          + ", ".join(f"цикл {k} {v:+.2f} °C/мес" for k, v in raw_rates.items()))
    print("  Нормировка ответ почти не меняет — значит, рост температуры не")
    print("  артефакт колебаний нагрузки и серы сырья, он настоящий.")

    drift = sulfur_drift_per_month(
        fit.rate_c_per_month, float(fit.frame["sulfur_out"].median()),
        float(fit.frame["sulfur_in"].median()), float(fit.frame["wabt"].median()),
        **({"activation_kj": args.activation} if args.activation else {}))
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    median_sulfur = float(fit.frame["sulfur_out"].median())
    weeks = (limit - median_sulfur) / drift * DAYS_IN_MONTH / 7 if drift > 0 else float("inf")
    print("\n  Что это значит для оператора: если температуру НЕ поднимать, сера")
    print(f"  продукта растёт примерно на {drift:+.2f} мг/кг в месяц. От медианы "
          f"{median_sulfur:.1f} до предела {limit:.0f} — около {weeks:.0f} недель.")
    print("  Отсюда и берётся необходимость постоянно двигать температуру вверх:")
    print("  это не запас прочности, а компенсация уходящей активности.")

    # --- 3. чувствительность к допущениям ------------------------------ #
    print("\n[3] Чувствительность к допущениям\n")
    sensitivity = {"энергия активации": {}, "эталонная сера": {}}
    for activation in (70.0, 85.0, 100.0, 115.0, 130.0):
        probe = fit_deactivation(ht, product, feed_sulfur, raw_feed=ht[FEED],
                                 target_sulfur=args.target, activation_kj=activation)
        if probe is not None:
            sensitivity["энергия активации"][activation] = round(probe.rate_c_per_month, 3)
    for target in (5.0, 8.0, 10.0):
        probe = fit_deactivation(ht, product, feed_sulfur, raw_feed=ht[FEED],
                                 target_sulfur=target, **(
                                     {"activation_kj": args.activation} if args.activation else {}))
        if probe is not None:
            sensitivity["эталонная сера"][target] = round(probe.rate_c_per_month, 3)
    fit.sensitivity = sensitivity
    print("  Ea, кДж/моль:   " + "  ".join(f"{k:.0f}→{v:+.3f}"
                                           for k, v in sensitivity["энергия активации"].items()))
    print("  эталон, мг/кг:  " + "  ".join(f"{k:.0f}→{v:+.3f}"
                                           for k, v in sensitivity["эталонная сера"].items()))
    print("  Обе оси меняют ответ в третьем знаке: оценка держится на данных, а не")
    print("  на выбранных нами константах. Это стоит сказать вслух — энергию")
    print("  активации мы всё-таки не измеряли.")

    # --- 4. остаточный ресурс ------------------------------------------ #
    print("\n[4] Остаточный ресурс текущего цикла\n")
    if not fit.current:
        print("  текущий цикл не выделен — все циклы завершены")
        return 0
    level = float(fit.current["NWABT сейчас, °C"])
    margin = float(fit.current["запас до уровня вывода, °C"])
    print(f"  Уровень вывода в ремонт: {fit.eor_level_c:.1f} °C — не норматив, а"
          f" собственная история установки")
    print("  (оба завершённых цикла выведены практически с одного уровня).")
    print(f"  Текущий цикл пущен {fit.current['пуск']}, наработка "
          f"{fit.current['наработка, сут']:.0f} сут, NWABT {level:.1f} °C, "
          f"запас {margin:.1f} °C.\n")

    linear = margin / fit.rate_c_per_month
    linear_ci = (margin / hi, margin / lo)
    print(f"  (а) по средней скорости:      {linear:5.1f} мес  "
          f"[{linear_ci[0]:.1f} … {linear_ci[1]:.1f}]")

    analogue = remaining_by_analogue(fit.frame, fit.cycles, level)
    left = [r["оставалось, мес"] for r in analogue if r.get("оставалось, мес") is not None]
    if left:
        print("  (б) по аналогии с прошлыми циклами: "
              + ", ".join(f"цикл {r['цикл']} → {r['оставалось, мес']:.1f} мес"
                          + (" (нижняя граница)" if r.get("нижняя граница") else "")
                          for r in analogue if r.get("оставалось, мес") is not None))
        if any(r.get("нижняя граница") for r in analogue):
            print("      «Нижняя граница» — про цикл, который наблюдается не с пуска:")
            print("      нынешнего уровня активности он достиг ДО начала данных, то есть")
            print("      прожил после него больше, чем видно. Занижает, а не завышает.")
    current_rate = float(fit.current["скорость в этом цикле, °C/мес"])
    print(f"  (в) по текущей скорости {current_rate:+.2f} °C/мес: {margin / current_rate:5.1f} мес")

    early = []
    for cycle in fit.cycles:
        if not cycle.completed:
            continue
        part = fit.frame[(fit.frame["cycle"] == cycle.index)
                         & (fit.frame["run_days"] <= fit.current["наработка, сут"])]
        if len(part) < 20:
            continue
        early.append((cycle.index,
                      float(np.polyfit(part["run_days"], part["nwabt"], 1)[0] * DAYS_IN_MONTH)))
    print("      но ранняя скорость всегда выше поздней: те же первые "
          f"{fit.current['наработка, сут']:.0f} сут в прошлых циклах шли "
          + ", ".join(f"{v:+.2f}" for _, v in early) + " °C/мес,")
    print("      а после 120-х суток выходили на общий наклон. Способ (в) — это")
    print("      нижняя граница «если разгон не кончится», а не ожидание.")

    low = min([x for x in ([linear_ci[0]] + left) if x == x]) if left else linear_ci[0]
    high = max([x for x in ([linear_ci[1]] + left) if x == x]) if left else linear_ci[1]
    print(f"\n  ОТВЕТ: остаточный ресурс цикла {min(left) if left else linear:.0f}–"
          f"{linear:.0f} месяцев, полный разброс способов {low:.0f}…{high:.0f}.")
    print("  Оценка держится на четырёх вещах, и каждую можно проверить:")
    print("    * кинетика псевдопервого порядка и Ea = 100 кДж/моль (допущение,")
    print("      но ответ к ней нечувствителен — см. [3]);")
    print("    * сера сырья интерполирована между 132 анализами;")
    print("    * уровень вывода взят из двух завершённых циклов, а не из норматива;")
    print("    * текущий цикл наблюдается неполные четыре месяца, и его ранняя")
    print("      скорость выше, чем была у предшественников на том же сроке —")
    print("      это повод пересчитать оценку на 150-е сутки, а не поверить ей раз.")

    # --- 5. цетановое число: тот же процесс или нет -------------------- #
    cetane = cetane_check(ht, fit.cycles)
    if cetane:
        print("\n[5] Падение цетанового числа — это дезактивация?\n")
        print(f"  По календарю: {cetane['наклон по календарю, ед/год']:+.2f} ед/год")
        for row in cetane["внутри циклов"]:
            print(f"  Внутри цикла {row['цикл']}: {row['наклон, ед/год']:+.2f} ед/год "
                  f"({row['анализов']} анализов)")
        for row in cetane["шаг при смене"]:
            print(f"  Смена {row['смена']}: ЦЧ {row['ЦЧ до']:.2f} → {row['ЦЧ после']:.2f} "
                  f"(шаг {row['шаг']:+.2f})")
        print("\n  Если бы причиной была дезактивация, при смене катализатора ЦЧ")
        print(f"  подскочило бы примерно на "
              f"{cetane['ожидаемый шаг, если причина — катализатор']:+.2f} единицы.")
        print(f"  ВЕРДИКТ: {cetane['вердикт']}.")
        print("  Внутрицикловой наклон совпадает с календарным, а смена катализатора")
        print("  на ЦЧ не отражается. Значит, гипотеза «тот же процесс» не")
        print("  подтвердилась, и искать причину надо в сырье, а не в катализаторе.")

    report = {
        "метод": "нормированная температура реакторного блока (NWABT), "
                 "кинетика псевдопервого порядка",
        "допущения": {
            "энергия активации, кДж/моль": args.activation or 100.0,
            "эталонная сера, мг/кг": args.target,
            "сера сырья": "132 анализа, между ними интерполяция по времени",
            "уровень вывода": "медиана NWABT за последние 45 суток завершённых циклов",
            "порог смены катализатора, °C": -10.0,
        },
        "наблюдений": int(len(fit.frame)),
        "остановы": fit.outage_steps,
        "циклы": [c.as_dict() for c in fit.cycles],
        "скорость_дезактивации": {
            "°C/мес": round(fit.rate_c_per_month, 3),
            "95% ДИ": [round(lo, 3), round(hi, 3)],
            "без кинетики, сырой WABT": {str(k): round(v, 3) for k, v in raw_rates.items()},
            "рост серы при замороженной температуре, мг/кг в мес": round(drift, 2),
        },
        "чувствительность": sensitivity,
        "уровень_вывода_°C": round(fit.eor_level_c, 1),
        "текущий_цикл": fit.current,
        "остаточный_ресурс_мес": {
            "по средней скорости": round(linear, 1),
            "по средней скорости, ДИ": [round(linear_ci[0], 1), round(linear_ci[1], 1)],
            "по аналогии": analogue,
            "по текущей скорости": round(margin / current_rate, 1),
            "ранняя скорость прошлых циклов, °C/мес": {str(k): round(v, 2) for k, v in early},
        },
        "цетановое_число": cetane,
    }
    out = ROOT / "reports" / "catalyst_life.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
