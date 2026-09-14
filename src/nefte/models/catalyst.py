"""Дезактивация катализатора гидроочистки в числах.

Зачем понадобилось. В severity наработка катализатора входит прокси-признаком
«часов от последнего длительного останова» — то есть возрастом, а не состоянием.
Возраст ничего не говорит ни о том, сколько запаса осталось, ни о том, когда
установку придётся выводить. ТЗ же требует оценивать надёжность оборудования, и
это единственный из четырёх критериев, где у нас до сих пор не было ни одного
измеренного числа.

Измерить дезактивацию по выданным данным можно, и стандартным промышленным
способом: **нормированная температура реакторного блока (NWABT)**. Идея в том,
что сама по себе WABT ничего не говорит — её поднимают и из-за тяжёлого сырья, и
из-за роста нагрузки. Нормировка убирает и то и другое, оставляя только
активность катализатора.

Для реакции псевдопервого порядка по сере::

    ln(S_вход / S_выход) = k(T) / LHSV,     k(T) = A·exp(−Ea / R·T)

Логарифмируя и разрешая относительно температуры, получаем температуру, при
которой ТЕКУЩИЙ катализатор дал бы эталонную серу при эталонной нагрузке::

    1/T_норм = 1/T_факт + (R/Ea)·[ ln ln(S_вх/S_вых) − ln ln(S_вх/S_эталон)
                                   + ln(расход) − ln(расход_эталон) ]

Предэкспонента A в разности сокращается — её знать не нужно, и это важно: именно
её падение и есть дезактивация. Остаётся один внешний параметр, энергия активации,
и она у нас уже объявлена допущением в ``models/regime.py`` (100 кДж/моль).
Чувствительность к ней проверяется отдельно и оказывается пренебрежимой: в
диапазоне 70…130 кДж/моль оценка скорости меняется на 3 %.

ДОПУЩЕНИЯ, которые надо назвать вслух:

* реакция псевдопервого порядка по сере, влияние водорода не выделено;
* объём катализатора неизвестен, поэтому LHSV входит пропорционально расходу сырья;
* сера сырья меряется примерно раз в месяц (132 анализа) и между анализами
  интерполируется по времени;
* уровень вывода установки в ремонт берётся не из нормативов (их в пакете нет),
  а из её собственной истории — по двум завершённым циклам.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nefte.models.regime import (
    ACTIVATION_ENERGY_KJ,
    FEED,
    GAS_CONSTANT_KJ,
    REACTOR_TEMPS,
    outage_mask,
)

DAYS_IN_MONTH = 30.44

# Эталонная сера продукта, к которой приводится температура. Значение роли не
# играет — проверено: при 5, 8 и 10 мг/кг скорость дезактивации совпадает до
# третьего знака. Взято близким к медиане лаборатории (8.6 мг/кг).
REFERENCE_SULFUR_MGKG = 8.0

# Режим усредняется за предшествующие 6 часов: постоянная времени канала серы
# 4.6 ч (reports/delays.json), мгновенное значение в момент отбора пробы шумит.
REGIME_WINDOW = "6h"

# Останов длиннее этого срока — кандидат в смену катализатора. Тот же порог, что
# в ``ReliabilityAgent.CATALYST_RESET_HOURS``.
LONG_OUTAGE_HOURS = 48.0

# Насколько должна упасть нормированная температура через останов, чтобы считать
# его СМЕНОЙ катализатора, а не просто ремонтом. В истории шаги через три
# длительных останова равны −23.8, −24.0 и +5.9 °C: порог −10 °C разделяет их с
# запасом, и ни одно значение рядом с ним не лежит.
RESET_STEP_C = -10.0

# Окно, по которому берётся уровень NWABT до и после останова.
STEP_WINDOW_DAYS = 45


@dataclass
class Cycle:
    """Один цикл катализатора, восстановленный по данным."""

    index: int
    start: pd.Timestamp
    end: pd.Timestamp | None          # None — цикл не завершён
    n_points: int
    days: float
    nwabt_start: float
    nwabt_end: float
    rate_c_per_month: float
    rate_ci: tuple[float, float]
    completed: bool
    # Первый цикл начинается не со смены катализатора, а с первой строки данных:
    # катализатор в нём уже какого-то возраста, и какого — неизвестно. Всё, что
    # про его длину и остаток, — нижние границы.
    left_censored: bool = False

    def as_dict(self) -> dict:
        return {
            "цикл": self.index,
            "наблюдается с начала": not self.left_censored,
            "пуск": str(self.start.date()),
            "вывод": str(self.end.date()) if self.end is not None else None,
            "завершён": self.completed,
            "длительность, сут": round(self.days, 0),
            "анализов": self.n_points,
            "NWABT в начале, °C": round(self.nwabt_start, 1),
            "NWABT в конце, °C": round(self.nwabt_end, 1),
            "скорость, °C/мес": round(self.rate_c_per_month, 3),
            "95% ДИ": [round(self.rate_ci[0], 3), round(self.rate_ci[1], 3)],
        }


@dataclass
class DeactivationFit:
    """Результат разбора: циклы, скорость, остаточный ресурс."""

    frame: pd.DataFrame
    cycles: list[Cycle]
    outage_steps: list[dict]
    rate_c_per_month: float
    rate_ci: tuple[float, float]
    eor_level_c: float
    current: dict
    sensitivity: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# нормированная температура
# --------------------------------------------------------------------------- #

def normalized_wabt(wabt: pd.Series, feed: pd.Series, sulfur_out: pd.Series,
                    sulfur_in: pd.Series, feed_reference: float,
                    target_sulfur: float = REFERENCE_SULFUR_MGKG,
                    activation_kj: float = ACTIVATION_ENERGY_KJ) -> pd.Series:
    """Температура, при которой текущий катализатор дал бы эталонную серу.

    Все ряды должны быть выровнены по одному индексу. Возвращает NWABT в °C.
    """
    conversion_now = np.log(sulfur_in / sulfur_out)
    conversion_ref = np.log(sulfur_in / target_sulfur)
    shift = (np.log(conversion_now) - np.log(conversion_ref)
             + np.log(feed) - np.log(feed_reference))
    inverse = 1.0 / (wabt + 273.15) + (GAS_CONSTANT_KJ / activation_kj) * shift
    return 1.0 / inverse.where(inverse > 0) - 273.15


def build_observations(ht: pd.DataFrame, sulfur_out: pd.Series, sulfur_in: pd.Series,
                       target_sulfur: float = REFERENCE_SULFUR_MGKG,
                       activation_kj: float = ACTIVATION_ENERGY_KJ) -> pd.DataFrame:
    """Таблица наблюдений: на каждый лабораторный анализ серы — режим и NWABT.

    Сера сырья меряется примерно раз в месяц, поэтому между анализами она
    интерполируется по времени. Это допущение, и оно смещает отдельную точку,
    но не тренд: интерполяция не создаёт систематического дрейфа.
    """
    temps = [t for t in REACTOR_TEMPS if t in ht.columns]
    if not temps or FEED not in ht.columns:
        return pd.DataFrame()

    # retrospective=True: здесь разбор ИСТОРИИ, а не срез на момент t. Отсеять
    # надо весь эпизод останова целиком, включая первые часы, — те отсчёты и
    # правда были остановом. Причинная маска (по умолчанию) объявляет останов
    # только после набора порога, и первые часы простоя попали бы в режим.
    working = ~outage_mask(ht[FEED], min_outage_hours=6.0, retrospective=True)
    wabt = ht[temps].mean(axis=1).where(working).rolling(REGIME_WINDOW).mean()
    feed = ht[FEED].where(working).rolling(REGIME_WINDOW).mean()

    grid = sulfur_in.index.union(sulfur_out.index)
    feed_sulfur = (sulfur_in.reindex(grid).interpolate("time").reindex(sulfur_out.index))

    out = pd.DataFrame({"sulfur_out": sulfur_out, "sulfur_in": feed_sulfur})
    position = ht.index.searchsorted(out.index, side="right") - 1
    known = position >= 0
    out, position = out[known], position[known]
    out["wabt"] = wabt.to_numpy()[position]
    out["feed"] = feed.to_numpy()[position]
    out = out.dropna()

    # Отсечки — про физику, а не про красоту: сера продукта выше 40 мг/кг это не
    # товарное топливо, сера сырья ниже 0.3 % — не то сырьё, расход ниже 50 —
    # не рабочий режим. Каждая из них убирает единицы точек.
    out = out[(out["sulfur_out"] > 0.5) & (out["sulfur_out"] < 40.0)
              & (out["sulfur_in"] > 3000.0) & (out["feed"] > 50.0)]
    if out.empty:
        return out

    out["nwabt"] = normalized_wabt(out["wabt"], out["feed"], out["sulfur_out"],
                                   out["sulfur_in"], float(out["feed"].median()),
                                   target_sulfur, activation_kj)
    return out.dropna(subset=["nwabt"])


# --------------------------------------------------------------------------- #
# циклы катализатора
# --------------------------------------------------------------------------- #

def long_outages(feed: pd.Series, min_hours: float = LONG_OUTAGE_HOURS,
                 steps_per_hour: int = 6) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Интервалы остановов длиннее ``min_hours`` по СЫРОМУ расходу сырья."""
    # Тот же случай: перечисляем эпизоды истории целиком, поэтому retrospective.
    # От границ эпизода зависит шаг NWABT через него, то есть вывод о смене
    # катализатора, — причинная маска сдвинула бы начало на длительность порога.
    down = outage_mask(feed, min_outage_hours=min_hours, steps_per_hour=steps_per_hour,
                       retrospective=True)
    block = (down != down.shift()).cumsum()
    out = []
    for _, part in feed.groupby(block):
        if down.loc[part.index[0]]:
            out.append((part.index[0], part.index[-1]))
    return out


def outage_steps(observations: pd.DataFrame, outages: list[tuple],
                 window_days: int = STEP_WINDOW_DAYS) -> list[dict]:
    """Шаг нормированной температуры через каждый длительный останов.

    Это и есть проверка допущения «длительный останов = смена катализатора».
    Раньше оно принималось на веру и давало четыре цикла; шаг NWABT показывает,
    что смен было две, а третий останов активность не вернул.
    """
    window = pd.Timedelta(days=window_days)
    rows = []
    for start, end in outages:
        before = observations.loc[start - window:start, "nwabt"]
        after = observations.loc[end:end + window, "nwabt"]
        level_before = float(before.median()) if len(before) >= 5 else float("nan")
        level_after = float(after.median()) if len(after) >= 5 else float("nan")
        step = level_after - level_before
        rows.append({
            "начало": str(start.date()),
            "конец": str(end.date()),
            "длительность, ч": round((end - start).total_seconds() / 3600, 0),
            "NWABT до, °C": None if level_before != level_before else round(level_before, 1),
            "NWABT после, °C": None if level_after != level_after else round(level_after, 1),
            "шаг, °C": None if step != step else round(step, 1),
            "смена катализатора": bool(step == step and step <= RESET_STEP_C),
        })
    return rows


def cycle_starts(observations: pd.DataFrame, steps: list[dict]) -> list[pd.Timestamp]:
    """Моменты пуска после смены катализатора плюс начало наблюдений."""
    starts = [observations.index[0].normalize()]
    for row in steps:
        if row["смена катализатора"]:
            starts.append(pd.Timestamp(row["конец"]))
    return sorted(set(starts))


# --------------------------------------------------------------------------- #
# скорость дезактивации
# --------------------------------------------------------------------------- #

def _block_bootstrap_slope(days: np.ndarray, values: np.ndarray, seed: int = 42,
                           n: int = 2000, block_days: float = 30.0
                           ) -> tuple[float, float]:
    """Доверительный интервал наклона блочным бутстрепом.

    Обычный бутстреп по точкам здесь запрещён: соседние анализы серы связаны,
    независимых наблюдений заметно меньше, чем строк, и интервал вышел бы вдвое
    у́же правды. Блок в 30 суток длиннее любой автокорреляции режима.
    """
    if len(days) < 30:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    blocks = (days // block_days).astype(int)
    unique = np.unique(blocks)
    if len(unique) < 4:
        return float("nan"), float("nan")
    index_by_block = {b: np.where(blocks == b)[0] for b in unique}
    slopes = []
    for _ in range(n):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_block[b] for b in picked])
        if len(idx) < 20:
            continue
        slopes.append(np.polyfit(days[idx], values[idx], 1)[0] * DAYS_IN_MONTH)
    if not slopes:
        return float("nan"), float("nan")
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))


def _pooled_rate(frame: pd.DataFrame, seed: int = 42, n: int = 2000,
                 block_days: float = 30.0) -> tuple[float, tuple[float, float]]:
    """Общая скорость по всем циклам: наклон общий, уровень у каждого свой.

    Циклы начинаются с разной активности (после смены −24 °C), поэтому просто
    сложить их нельзя: разница уровней притворится трендом. Фиксированный эффект
    цикла убирает уровень и оставляет только наклон.
    """
    codes, _ = pd.factorize(frame["cycle"])
    dummies = (np.eye(codes.max() + 1)[codes][:, 1:] if codes.max() > 0
               else np.empty((len(frame), 0)))
    days = frame["run_days"].to_numpy(dtype="float64")
    values = frame["nwabt"].to_numpy(dtype="float64")
    design = np.column_stack([days, dummies, np.ones(len(frame))])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    rate = float(beta[0] * DAYS_IN_MONTH)

    rng = np.random.default_rng(seed)
    keys = (frame["cycle"].astype(str) + "_"
            + (frame["run_days"] // block_days).astype(int).astype(str)).to_numpy()
    unique = np.unique(keys)
    index_by_key = {k: np.where(keys == k)[0] for k in unique}
    slopes = []
    for _ in range(n):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_key[k] for k in picked])
        try:
            coef, *_ = np.linalg.lstsq(design[idx], values[idx], rcond=None)
        except np.linalg.LinAlgError:          # pragma: no cover — вырожденная выборка
            continue
        slopes.append(float(coef[0] * DAYS_IN_MONTH))
    ci = ((float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
          if slopes else (float("nan"), float("nan")))
    return rate, ci


def fit_deactivation(ht: pd.DataFrame, sulfur_out: pd.Series, sulfur_in: pd.Series,
                     raw_feed: pd.Series | None = None,
                     target_sulfur: float = REFERENCE_SULFUR_MGKG,
                     activation_kj: float = ACTIVATION_ENERGY_KJ,
                     seed: int = 42) -> DeactivationFit | None:
    """Полный разбор: циклы, скорость дезактивации, остаточный ресурс.

    ``raw_feed`` — расход сырья ДО очистки: маска останова считается по нему, как
    и везде в проекте (замороженный на нуле расход и есть факт останова).
    """
    observations = build_observations(ht, sulfur_out, sulfur_in, target_sulfur,
                                      activation_kj)
    if observations.empty or FEED not in ht.columns:
        return None

    feed_raw = raw_feed if raw_feed is not None else ht[FEED]
    steps = outage_steps(observations, long_outages(feed_raw))
    starts = cycle_starts(observations, steps)

    cycle_id = np.searchsorted(np.array(starts, dtype="datetime64[ns]"),
                               observations.index.to_numpy(dtype="datetime64[ns]"),
                               side="right") - 1
    observations = observations.assign(cycle=cycle_id)
    observations["run_days"] = [
        (ts - starts[c]).total_seconds() / 86400
        for ts, c in zip(observations.index, observations["cycle"])
    ]

    resets = [pd.Timestamp(r["начало"]) for r in steps if r["смена катализатора"]]
    reset_starts = {pd.Timestamp(r["конец"]) for r in steps if r["смена катализатора"]}
    cycles: list[Cycle] = []
    for number, part in observations.groupby("cycle"):
        if len(part) < 30:
            continue
        days = part["run_days"].to_numpy(dtype="float64")
        values = part["nwabt"].to_numpy(dtype="float64")
        slope, intercept = np.polyfit(days, values, 1)
        end = next((r for r in resets if r > part.index[0]), None)
        tail = part[part["run_days"] > days.max() - STEP_WINDOW_DAYS]["nwabt"]
        cycles.append(Cycle(
            index=int(number), start=starts[int(number)], end=end,
            n_points=len(part), days=float(days.max()),
            nwabt_start=float(intercept),
            nwabt_end=float(tail.median()),
            rate_c_per_month=float(slope * DAYS_IN_MONTH),
            rate_ci=_block_bootstrap_slope(days, values, seed=seed),
            completed=end is not None,
            left_censored=starts[int(number)] not in reset_starts,
        ))

    completed = [c for c in cycles if c.completed]
    if not completed:
        return None
    pooled_frame = observations[observations["cycle"].isin([c.index for c in completed])]
    rate, rate_ci = _pooled_rate(pooled_frame, seed=seed)
    eor = float(np.mean([c.nwabt_end for c in completed]))

    current = cycles[-1] if cycles and not cycles[-1].completed else None
    current_info: dict = {}
    if current is not None:
        part = observations[observations["cycle"] == current.index]
        recent = part[part["run_days"] > current.days - STEP_WINDOW_DAYS]["nwabt"]
        level = float(recent.median())
        margin = eor - level
        current_info = {
            "пуск": str(current.start.date()),
            "наработка, сут": round(current.days, 0),
            "NWABT сейчас, °C": round(level, 1),
            "запас до уровня вывода, °C": round(margin, 1),
            "скорость в этом цикле, °C/мес": round(current.rate_c_per_month, 2),
        }

    return DeactivationFit(frame=observations, cycles=cycles, outage_steps=steps,
                           rate_c_per_month=rate, rate_ci=rate_ci, eor_level_c=eor,
                           current=current_info)


def sulfur_drift_per_month(rate_c_per_month: float, sulfur_out: float,
                           sulfur_in: float,
                           wabt_c: float = 365.0,
                           activation_kj: float = ACTIVATION_ENERGY_KJ) -> float:
    """Во что обходится дезактивация, если температуру НЕ поднимать, мг/кг в месяц.

    Из ``S = S_вх·exp(−τ)`` при ``τ = k/LHSV``::

        dS/dT = −S·τ·Ea/(R·T²)

    Обратная величина переводит «градусы потерянной активности» в «миллиграммы
    выросшей серы» — то, что видит оператор, если ничего не делать.
    """
    kelvin = wabt_c + 273.15
    conversion = float(np.log(sulfur_in / sulfur_out))
    sensitivity = sulfur_out * conversion * activation_kj / (GAS_CONSTANT_KJ * kelvin ** 2)
    return float(rate_c_per_month * sensitivity)


def remaining_by_analogue(frame: pd.DataFrame, cycles: list[Cycle],
                          level_c: float) -> list[dict]:
    """Сколько цикл прожил ПОСЛЕ того, как достиг текущего уровня активности.

    Непараметрическая оценка: никакой линейности не предполагает, спрашивает у
    истории самой установки. Уровень NWABT сглаживается медианой по 21 анализу,
    иначе первый же шумный выброс засчитается за достижение уровня.
    """
    rows = []
    for cycle in cycles:
        if not cycle.completed:
            continue
        part = frame[frame["cycle"] == cycle.index].copy()
        part["smooth"] = part["nwabt"].rolling(21, center=True, min_periods=7).median()
        reached = part[part["smooth"] >= level_c]
        if reached.empty:
            rows.append({"цикл": cycle.index, "достиг уровня": None,
                         "оставалось, сут": None})
            continue
        day = float(reached["run_days"].iloc[0])
        rows.append({
            "цикл": cycle.index,
            "достиг уровня на сутки": round(day, 0),
            "вывод на сутки": round(cycle.days, 0),
            "оставалось, сут": round(cycle.days - day, 0),
            "оставалось, мес": round((cycle.days - day) / DAYS_IN_MONTH, 1),
            # Левообрезанный цикл достиг уровня ДО начала наблюдений, поэтому
            # «оставалось» у него занижено: настоящий остаток ещё меньше.
            "нижняя граница": cycle.left_censored and day <= 1.0,
        })
    return rows


# --------------------------------------------------------------------------- #
# сравнение циклов на одинаковой наработке
# --------------------------------------------------------------------------- #

# Полуширина окна, в котором берётся уровень NWABT «на такие-то сутки». Анализы
# идут примерно раз в сутки, так что ±20 суток — это около сорока точек: хватает
# на устойчивую медиану и мало по сравнению с длиной цикла.
LEVEL_WINDOW_DAYS = 20


def level_at_runday(frame: pd.DataFrame, cycle: int, day: float,
                    half_window: int = LEVEL_WINDOW_DAYS, seed: int = 42,
                    n: int = 4000) -> dict | None:
    """Уровень активности цикла на заданной наработке, с интервалом.

    Сравнивать циклы по НАКЛОНУ на коротком окне почти бесполезно: интервал
    наклона по сорока точкам шире самой разницы. Уровень устойчивее — он
    накапливает всю предысторию цикла, а не последние две недели.
    """
    part = frame[(frame["cycle"] == cycle)
                 & (frame["run_days"] > day - half_window)
                 & (frame["run_days"] <= day + half_window)]
    values = part["nwabt"].to_numpy(dtype="float64")
    if len(values) < 8:
        return None
    rng = np.random.default_rng(seed)
    boots = [float(np.median(rng.choice(values, len(values)))) for _ in range(n)]
    return {"цикл": cycle, "сутки": round(float(day), 0), "анализов": int(len(values)),
            "уровень, °C": round(float(np.median(values)), 1),
            "95% ДИ": [round(float(np.percentile(boots, 2.5)), 1),
                       round(float(np.percentile(boots, 97.5)), 1)]}


def lag_against_reference(frame: pd.DataFrame, cycle: int, reference: int,
                          days: list[float], half_window: int = LEVEL_WINDOW_DAYS,
                          seed: int = 42, n: int = 4000) -> list[dict]:
    """Насколько цикл отстаёт от эталонного на одинаковой наработке.

    Положительное отставание — цикл требует БОЛЬШЕЙ температуры на том же сроке,
    то есть катализатор слабее. Интервал считается бутстрепом разности медиан:
    без него разница в пару градусов неотличима от шума лаборатории.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for day in days:
        pair = {}
        for name, number in (("эталон", reference), ("цикл", cycle)):
            part = frame[(frame["cycle"] == number)
                         & (frame["run_days"] > day - half_window)
                         & (frame["run_days"] <= day + half_window)]
            pair[name] = part["nwabt"].to_numpy(dtype="float64")
        if min(len(v) for v in pair.values()) < 8:
            continue
        diff = float(np.median(pair["цикл"]) - np.median(pair["эталон"]))
        boots = [float(np.median(rng.choice(pair["цикл"], len(pair["цикл"])))
                       - np.median(rng.choice(pair["эталон"], len(pair["эталон"]))))
                 for _ in range(n)]
        low, high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        rows.append({
            "сутки": round(float(day), 0),
            f"цикл {reference}, °C": round(float(np.median(pair["эталон"])), 1),
            f"цикл {cycle}, °C": round(float(np.median(pair["цикл"])), 1),
            "отставание, °C": round(diff, 2),
            "95% ДИ": [round(low, 2), round(high, 2)],
            "значимо": bool(low > 0 or high < 0),
        })
    return rows


def remaining_by_lag(frame: pd.DataFrame, cycles: list[Cycle], current: Cycle,
                     reference: Cycle, rate_c_per_month: float,
                     half_window: int = LEVEL_WINDOW_DAYS) -> dict | None:
    """Остаток ресурса через отставание от полностью наблюдённого предшественника.

    Способ отвечает на вопрос иначе, чем ``remaining_by_analogue``, и разница
    принципиальна. Аналогия сопоставляет циклы по УРОВНЮ активности: «когда
    предшественник был так же слаб и сколько после этого прожил». Здесь
    сопоставление по НАРАБОТКЕ: предшественник на том же сроке прожил ещё
    столько-то, а наш отстаёт на столько-то градусов, то есть на столько-то
    месяцев наработки — вычитаем.

    Первый способ верен, если ресурс определяется накопленной дезактивацией;
    второй — если траектория та же, но сдвинутая. Что из этого правда, по двум
    наблюдённым циклам не решить, поэтому считаем оба и показываем разброс.
    """
    if not reference.completed or reference.left_censored:
        return None
    day = float(current.days)
    ours = level_at_runday(frame, current.index, day, half_window)
    theirs = level_at_runday(frame, reference.index, day, half_window)
    if ours is None or theirs is None or rate_c_per_month <= 0:
        return None
    lag = ours["уровень, °C"] - theirs["уровень, °C"]
    reference_left = (reference.days - day) / DAYS_IN_MONTH
    return {
        "эталонный цикл": reference.index,
        "наработка, сут": round(day, 0),
        "эталон прожил ещё, мес": round(reference_left, 1),
        "отставание, °C": round(lag, 2),
        "отставание, мес наработки": round(lag / rate_c_per_month, 1),
        "остаток, мес": round(reference_left - lag / rate_c_per_month, 1),
    }


def local_rate(frame: pd.DataFrame, cycle: int, since_day: float,
               until_day: float) -> float | None:
    """Наклон NWABT на участке наработки, °C/мес. None — если точек мало."""
    part = frame[(frame["cycle"] == cycle) & (frame["run_days"] >= since_day)
                 & (frame["run_days"] <= until_day)]
    if len(part) < 20:
        return None
    slope = np.polyfit(part["run_days"], part["nwabt"], 1)[0]
    return float(slope * DAYS_IN_MONTH)


def feed_normalized_wabt(wabt: pd.Series, feed: pd.Series, feed_reference: float,
                         activation_kj: float = ACTIVATION_ENERGY_KJ) -> pd.Series:
    """WABT, приведённая к опорной нагрузке. Прокси дезактивации БЕЗ лаборатории.

    Зачем отдельно от ``normalized_wabt``. Полная нормировка отвечает на вопрос
    «какая температура дала бы эталонную серу» — для этого нужна сера, то есть
    лаборатория. Агенту надёжности такой вопрос не нужен и лаборатории у него
    нет: severity считается на десятиминутной сетке, а анализ приходит раз в
    сутки. Здесь остаётся только та часть нормировки, которая считается по
    телеметрии::

        1/T_норм = 1/T_факт + (R/Ea)·[ ln(расход) − ln(расход_эталон) ]

    Смысл: выше нагрузка — меньше время контакта, и ту же глубину очистки надо
    добирать температурой. Убрав этот вклад, видим собственно активность.

    **Почему серного члена здесь нет — это измерено, а не упрощение.** Разложив
    полную нормировку на две поправки и посмотрев на шум ряда вокруг тренда
    дезактивации внутри цикла (`docs/CATALYST_LIFE.md` §9):

    ===========================  =====  ==========
    вариант                      шум    сигнал/шум
    ===========================  =====  ==========
    сырой WABT                   8.34    0.096
    только поправка на серу      8.53    0.093
    только поправка на нагрузку  5.35    0.159
    обе                          5.39    0.157
    ===========================  =====  ==========

    Весь выигрыш даёт нагрузка. Серный член добавляет шум лаборатории и не
    добавляет сигнала: сера меряется раз в сутки с разбросом в несколько мг/кг, а
    дезактивация идёт со скоростью 0.85 °C в месяц.

    **В матрицу признаков эту величину добавлять НЕ НАДО, и проверять это прогоном
    тоже не надо.** Доказывается алгеброй. В матрице уже есть ``reg_kinetic``::

        reg_kinetic = exp(−Ea/(R·T)) / расход
        ln(reg_kinetic) = −Ea/(R·T) − ln(расход) + const

    Домножив на −(R/Ea) и подставив определение выше::

        1/T_норм = −(R/Ea)·ln(reg_kinetic) + const

    То есть ``1/T_норм`` — аффинная функция ``ln(reg_kinetic)``, а сама ``T_норм``
    — монотонное преобразование ``reg_kinetic``. Дерево делит выборку ПО ПОРЯДКУ,
    монотонное преобразование порядок сохраняет, значит новой информации ровно
    ноль — не «мало», а ноль. Проверено численно: ранговая корреляция
    **1.0000000000** на 181 349 отсчётах. Это свойство преобразования, а не наших
    данных, поэтому оно не изменится ни от версии матрицы, ни от сида.

    Оговорка, которая понадобится при усреднении: точное равенство поточечное. На
    режиме, усреднённом за 6 часов (как считается в разборе ресурса), неравенство
    Йенсена его слегка рвёт — ранговая корреляция 0.9989, а не единица.

    **А для severity та же величина осмысленна, и это не противоречие.** severity
    — не дерево, а взвешенный индекс, где масштаб и единицы значат всё: «столько-то
    градусов приведённой температуры» складывается с другими факторами, а
    ``exp(−Ea/RT)/расход`` в градусах не измеряется и складываться с ними не может.
    Одна и та же формула полезна в одном агенте и бессмысленна в другом, и причина
    не в данных, а в устройстве потребителя.
    """
    shift = (GAS_CONSTANT_KJ / activation_kj) * (np.log(feed.where(feed > 0))
                                                 - np.log(feed_reference))
    inverse = 1.0 / (wabt + 273.15) + shift
    return 1.0 / inverse.where(inverse > 0) - 273.15


# --------------------------------------------------------------------------- #
# ряды для агента надёжности
# --------------------------------------------------------------------------- #

def hours_since_marks(index: pd.DatetimeIndex, marks) -> pd.Series:
    """Часы с последней отметки (замены катализатора) на момент каждого отсчёта.

    До первой отметки — часы от начала данных: катализатор в первом цикле уже
    какого-то возраста, и какого, неизвестно — ровно так же считает и прежний
    вариант по остановам.
    """
    stamps = sorted(pd.Timestamp(m) for m in marks)
    last = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    for stamp in stamps:
        last[index >= stamp] = stamp
    since = (index.to_series() - last).dt.total_seconds() / 3600
    start = (index.to_series() - index[0]).dt.total_seconds() / 3600
    return since.fillna(start)


def activity_level_series(ht: pd.DataFrame, raw_feed: pd.Series,
                          train: tuple[str, str], window_days: int = 30,
                          min_days: int = 7) -> pd.Series | None:
    """Износ катализатора для severity: уровень WABT, приведённой к нагрузке.

    Медиана за ``window_days`` суток по ПРОШЛОМУ, нормированная на p05…p95
    обучающего периода — той же шкалой, что остальные факторы тяжести режима.

    Почему уровень, а не дрейф от начала цикла: дрейф требует даты начала, то есть
    той самой даты сброса, в которой и была ошибка. Уровень её не требует — после
    замены катализатора оператор держит ту же серу холоднее, и уровень падает сам,
    а после ремонта без замены — нет.

    Причинность соблюдена везде: маска останова причинная, часовые значения
    помечены правой границей часа, скользящая медиана кончается в текущем часе.
    """
    temps = [t for t in REACTOR_TEMPS if t in ht.columns]
    if not temps or FEED not in ht.columns:
        return None
    feed = ht[FEED]
    reference = float(feed.loc[train[0]:train[1]].median())
    if not reference > 0:
        return None
    down = outage_mask(raw_feed.reindex(ht.index), min_outage_hours=6.0, train_bounds=train)
    # Малый расход на пуске и в провалах нагрузки делает логарифм нагрузки
    # огромным — такие отсчёты не рабочий режим и в уровень не идут.
    working = ~down & (feed > 0.5 * reference)
    wabt = ht[temps].mean(axis=1).where(working)
    level = feed_normalized_wabt(wabt, feed.where(working), reference)
    hourly = level.resample("1h", label="right", closed="right").mean()
    smooth = hourly.rolling(f"{window_days}D", min_periods=min_days * 24).median()
    train_part = smooth.loc[train[0]:train[1]].dropna()
    if len(train_part) < 100:
        return None
    low, high = train_part.quantile([0.05, 0.95])
    if not high > low:
        return None
    factor = ((smooth - low) / (high - low)).clip(0.0, 1.5)
    return factor.reindex(ht.index, method="ffill")
