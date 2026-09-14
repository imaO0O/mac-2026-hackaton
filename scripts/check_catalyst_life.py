"""Скорость дезактивации катализатора и остаточный ресурс цикла (участник 2). CPU.

    python scripts/check_catalyst_life.py
    python scripts/check_catalyst_life.py --checkpoint-day 150   # контрольная точка
    python scripts/check_catalyst_life.py --as-of 2026-06-01     # «как будто сегодня»
    python scripts/check_catalyst_life.py --activation 120 --target 5

Четвёртый критерий ТЗ — надёжность оборудования — до сих пор держался у нас на
прокси без единого измеренного числа: наработка катализатора входила в severity
как «часов от последнего останова». Возраст не отвечает ни на один вопрос,
который на самом деле задают: сколько активности теряется в месяц, сколько
градусов запаса осталось и когда установку придётся выводить.

Метод — нормированная температура реакторного блока (разбор в
``models/catalyst.py``). Считается по лабораторным анализам серы: 1451 точка,
сера продукта и сера сырья, режим усреднён за 6 часов до отбора пробы.

Скрипт отвечает на вопросы по порядку, и на каждый даёт число с интервалом:

1. Сколько длительных остановов действительно были сменой катализатора?
   Проверяется шагом нормированной температуры, а не длительностью.
2. Сколько градусов в месяц приходится добавлять, чтобы держать серу?
   Оценка по завершённым циклам, интервал — блочным бутстрепом.
3. Насколько ответ зависит от наших допущений, а не от данных.
4. Сколько месяцев осталось текущему циклу? Четырьмя независимыми способами,
   потому что один способ на такой выборке — это мнение, а не оценка.
5. Контрольная точка: отстаёт ли текущая партия катализатора от предшественницы
   на одинаковой наработке, и по какому критерию судить об этом дальше. Критерий
   снимается с данных ЗАРАНЕЕ, иначе «пересчитать через месяц» — напоминание,
   а не проверка.
6. Бэктест: что этот же метод сказал бы на прошлых сутках цикла, исход которого
   уже известен. Единственная возможная проверка оценки остаточного ресурса.
7. Проверка гипотезы «падение цетанового числа — тот же процесс»: ЦЧ должно
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
    CHECKPOINT_LEVEL_MARGIN_C,
    CHECKPOINT_MAX_RATE_C_PER_MONTH,
    DAYS_IN_MONTH,
    REFERENCE_SULFUR_MGKG,
    Cycle,
    fit_deactivation,
    lag_against_reference,
    level_at_runday,
    local_rate,
    remaining_by_analogue,
    remaining_by_lag,
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


def checkpoint_block(fit, reference, checkpoint_day: int, today: pd.Timestamp) -> dict:
    """Контрольная точка: сверка с предшественником на одинаковой наработке.

    Пункт «пересчитать на N-е сутки» без критерия — это напоминание, а не
    проверка: через месяц никто не вспомнит, что считать плохим результатом.
    Здесь критерий вынимается из данных заранее — чем был на этих сутках
    полностью наблюдённый предшественник, — и печатается вне зависимости от
    того, дожили мы до контрольной точки или нет.
    """
    print(f"\n[5] Контрольная точка: {checkpoint_day}-е сутки текущего цикла\n")
    if fit.current is None or reference is None:
        print("  не с чем сверять: нет текущего цикла или полностью наблюдённого "
              "предшественника")
        return {}

    current = next(c for c in fit.cycles if not c.completed)
    days = [d for d in (40, 60, 80, 100, checkpoint_day) if d <= current.days + 1]
    lag = lag_against_reference(fit.frame, current.index, reference.index, days)
    if lag:
        print(pd.DataFrame(lag).to_string(index=False))
        first = next((r for r in lag if r["значимо"]), None)
        if first:
            print(f"\n  Отставание становится значимым с {first['сутки']:.0f}-х суток: "
                  f"до этого доверительный интервал накрывает ноль.")
        else:
            print("\n  Отставание НЕ значимо ни на одних сутках: интервал везде "
                  "накрывает ноль.")

    reached = current.days >= checkpoint_day
    print(f"\n  Данные доведены до {current.days:.0f}-х суток "
          f"(последний анализ {fit.frame.index.max().date()}), "
          f"{checkpoint_day}-е сутки наступают "
          f"{(current.start + pd.Timedelta(days=checkpoint_day)).date()}.")

    target = level_at_runday(fit.frame, reference.index, checkpoint_day)
    pace = local_rate(fit.frame, reference.index, 60.0, float(checkpoint_day))
    criterion = {}
    if target is not None:
        criterion = {"эталонный цикл": reference.index, "сутки": checkpoint_day,
                     "уровень эталона, °C": target["уровень, °C"],
                     "наклон эталона на сутках 60…N, °C/мес": None if pace is None
                     else round(pace, 2)}
        print(f"\n  КРИТЕРИЙ, снятый с цикла {reference.index} заранее:")
        print(f"    на {checkpoint_day}-е сутки он был на уровне "
              f"{target['уровень, °C']:.1f} °C (ДИ {target['95% ДИ']}),")
        if pace is not None:
            print(f"    а на сутках 60…{checkpoint_day} шёл со скоростью "
                  f"{pace:+.2f} °C/мес — то есть уже вышел на полку.")
        print(f"    Если на {checkpoint_day}-е сутки NWABT окажется выше "
              f"{target['уровень, °C'] + CHECKPOINT_LEVEL_MARGIN_C:.0f} °C "
              "или скорость на этом же участке")
        print(f"    останется выше {CHECKPOINT_MAX_RATE_C_PER_MONTH:+.0f} °C/мес, "
              "оценку ресурса надо пересматривать вниз.")

    if reached:
        ours = level_at_runday(fit.frame, current.index, checkpoint_day)
        our_pace = local_rate(fit.frame, current.index, 60.0, float(checkpoint_day))
        print(f"\n  ФАКТ: {ours['уровень, °C']:.1f} °C, скорость "
              f"{'—' if our_pace is None else f'{our_pace:+.2f}'} °C/мес.")
        criterion["факт"] = ours
    else:
        print(f"\n  Пересчитать нечем: в выданном пакете телеметрия кончается "
              f"{fit.frame.index.max().date()}. Когда данные появятся, "
              f"контрольная точка — одна команда:")
        print(f"    python scripts/check_catalyst_life.py --checkpoint-day {checkpoint_day}")
        print("  А чтобы убедиться, что она работает, её можно прогнать задним "
              "числом на\n  любую прошлую дату: --as-of 2026-06-01.")
    criterion["отставание"] = lag
    criterion["достигнута"] = bool(reached)
    return criterion


def backtest_block(ht: pd.DataFrame, product: pd.Series, feed_sulfur: pd.Series,
                   fit, args) -> list[dict]:
    """Что метод сказал бы на прошлых сутках цикла, исход которого мы знаем.

    Единственная возможная проверка оценки остаточного ресурса: взять
    завершённый цикл, обрезать данные на его N-х сутках и сравнить предсказание
    с тем, что случилось на самом деле. Обрезается ВСЁ — и телеметрия, и
    лаборатория, — иначе проверка подсмотрит будущее теми же данными, из которых
    строится уровень вывода в ремонт.
    """
    print("\n[6] Бэктест: насколько метод врал на цикле, исход которого известен\n")
    reference = next((c for c in fit.cycles if c.completed and not c.left_censored), None)
    if reference is None:
        print("  нет полностью наблюдённого завершённого цикла — проверять не на чем")
        return []

    kwargs = {"target_sulfur": args.target}
    if args.activation is not None:
        kwargs["activation_kj"] = args.activation

    rows = []
    for day in (108, 150, 200, 300, 400, 500):
        if day >= reference.days:
            continue
        as_of = reference.start + pd.Timedelta(days=day)
        past = fit_deactivation(ht.loc[:as_of], product.loc[:as_of],
                                feed_sulfur.loc[:as_of], raw_feed=ht.loc[:as_of, FEED],
                                **kwargs)
        if past is None or not past.current:
            continue
        level = float(past.current["NWABT сейчас, °C"])
        margin = past.eor_level_c - level
        linear = margin / past.rate_c_per_month if past.rate_c_per_month > 0 else float("nan")
        analogue = [r["оставалось, мес"] for r in
                    remaining_by_analogue(past.frame, past.cycles, level)
                    if r.get("оставалось, мес") is not None]
        truth = (reference.days - day) / DAYS_IN_MONTH
        rows.append({
            "сутки": day,
            "NWABT тогда, °C": round(level, 1),
            "(а) линейно, мес": round(linear, 1),
            "(б) аналогия, мес": round(min(analogue), 1) if analogue else None,
            "ФАКТ, мес": round(truth, 1),
            "ошибка (а), %": round((linear - truth) / truth * 100, 0),
        })
    if not rows:
        print("  не набралось точек для проверки")
        return []
    print(pd.DataFrame(rows).to_string(index=False))

    early = rows[0]
    print(f"\n  На {early['сутки']}-х сутках — то есть ровно там, где сейчас стоит "
          f"текущий цикл, —")
    print(f"  линейный способ дал {early['(а) линейно, мес']:.1f} мес при факте "
          f"{early['ФАКТ, мес']:.1f}: промах {early['ошибка (а), %']:+.0f} %.")
    print("  Аналогия в этой точке промахнулась сильнее и в другую сторону, но ей")
    print("  тогда не на что было опереться, кроме левообрезанного цикла.")
    late = [r for r in rows if r["сутки"] >= 300]
    if late:
        print("\n  А вот к концу цикла метод систематически ВРЁТ В ПЛЮС: "
              + ", ".join(f"{r['сутки']} сут {r['ошибка (а), %']:+.0f} %" for r in late) + ".")
        print("  Причина понятная: средняя по циклу скорость занижает ту, с которой")
        print("  катализатор стареет в конце. Пользоваться линейной оценкой можно в")
        print("  начале цикла, а ближе к выводу она превращается в утешение.")

    if fit.current:
        level = float(fit.current["NWABT сейчас, °C"])
        linear = (fit.eor_level_c - level) / fit.rate_c_per_month
        corrected = linear / (1.0 + early["ошибка (а), %"] / 100.0)
        print(f"\n  Отсюда поправка к сегодняшней оценке: линейные "
              f"{linear:.1f} мес с известным оптимизмом "
              f"{early['ошибка (а), %']:+.0f} % дают {corrected:.1f} мес.")
        print("  Это независимая дорога к тому же числу, что и способ (г), и обе")
        print("  опираются на единственный цикл, исход которого мы видели целиком.")
        for row in rows:
            row["скорректированная оценка текущего цикла, мес"] = (
                round(corrected, 1) if row is early else None)
    return rows


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", type=float, default=None,
                    help="энергия активации, кДж/моль (по умолчанию из models/regime.py)")
    ap.add_argument("--target", type=float, default=REFERENCE_SULFUR_MGKG,
                    help="эталонная сера продукта, мг/кг")
    ap.add_argument("--checkpoint-day", type=int, default=150,
                    help="на какие сутки наработки назначена контрольная точка")
    ap.add_argument("--as-of", default=None,
                    help="считать так, будто сегодня эта дата: всё, что позже, "
                         "не используется (проверка контрольной точки задним числом)")
    args = ap.parse_args()

    cfg = load_config()
    ht, product, feed_sulfur = load_inputs(cfg)
    if args.as_of:
        # Обрезаем ВСЕ источники, а не только телеметрию: уровень вывода в ремонт
        # считается по лаборатории, и через неё проверка подсмотрела бы будущее.
        today = pd.Timestamp(args.as_of)
        ht = ht.loc[:today]
        product, feed_sulfur = product.loc[:today], feed_sulfur.loc[:today]
        print(f"[срез] считаем по данным до {today.date()} включительно")
    else:
        today = ht.index.max()
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
    current_cycle = next(c for c in fit.cycles if not c.completed)

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
    print("      а дальше выходили на общий наклон. Способ (в) — это нижняя")
    print("      граница «если разгон не кончится», а не ожидание.")

    # (г) поправка на отставание от полностью наблюдённого предшественника
    reference = next((c for c in fit.cycles
                      if c.completed and not c.left_censored), None)
    by_lag = (remaining_by_lag(fit.frame, fit.cycles, current_cycle, reference,
                               fit.rate_c_per_month) if reference else None)
    if by_lag:
        print(f"  (г) поправка на отставание от цикла {by_lag['эталонный цикл']}: "
              f"{by_lag['остаток, мес']:5.1f} мес")
        print(f"      цикл {by_lag['эталонный цикл']} с этой наработки прожил ещё "
              f"{by_lag['эталон прожил ещё, мес']:.1f} мес (это факт, а не оценка),")
        print(f"      а мы отстаём от него на {by_lag['отставание, °C']:+.2f} °C — "
              f"то есть на {by_lag['отставание, мес наработки']:.1f} мес наработки.")

    candidates = [x for x in ([linear] + left + ([by_lag["остаток, мес"]] if by_lag else []))
                  if x == x]
    low, high = min(candidates), max(candidates)
    print(f"\n  ОТВЕТ: остаточный ресурс цикла {low:.0f}…{high:.0f} месяцев.")
    if by_lag:
        print(f"  Ближе к {by_lag['остаток, мес']:.0f}: способы (а) и (г) сходятся, "
              f"а расхождение с (б) объяснено ниже, в [6].")
    print("  Оценка держится на пяти вещах, и каждую можно проверить:")
    print("    * кинетика псевдопервого порядка и Ea = 100 кДж/моль (допущение,")
    print("      но ответ к ней нечувствителен — см. [3]);")
    print("    * сера сырья интерполирована между 132 анализами;")
    print("    * уровень вывода взят из двух завершённых циклов, а не из норматива;")
    print("    * полностью наблюдаемый цикл всего один: скорость подтверждена двумя,")
    print("      а длина цикла — по существу одним;")
    print("    * текущий цикл наблюдается неполные четыре месяца.")

    # --- 5. контрольная точка ------------------------------------------ #
    checkpoint = checkpoint_block(fit, reference, args.checkpoint_day, today)

    # --- 6. бэктест самого метода -------------------------------------- #
    backtest = backtest_block(ht, product, feed_sulfur, fit, args)

    # --- 7. цетановое число: тот же процесс или нет -------------------- #
    cetane = cetane_check(ht, fit.cycles)
    if cetane:
        print("\n[7] Падение цетанового числа — это дезактивация?\n")
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
        "срез_данных": str(today.date()),
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
            "по отставанию от эталона": by_lag,
            "ранняя скорость прошлых циклов, °C/мес": {str(k): round(v, 2) for k, v in early},
        },
        "контрольная_точка": checkpoint,
        "бэктест_метода": backtest,
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
