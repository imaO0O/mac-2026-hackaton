"""Эпизоды, где режим менялся не в ответ на серу.

Зачем. Отклик серы на режим — главное допущение системы: оптимизатор считает его по
кинетике первого порядка (−22 % на градус), литература даёт 5–10 %, и по истории
его не выделить в лоб. Причина — обратная связь оператора. Он поднимает
температуру, УВИДЕВ рост серы, и в корреляции «температура → сера» два эффекта
смешиваются: процесс снижает серу, а сера перед этим подняла температуру. Оценить
отклик можно только на изменениях режима, причиной которых сера не была.

Здесь такие изменения собираются в список с датами и контекстом. Типов три:

* **смена катализатора** — активность меняется скачком, независимо от серы;
* **выход на режим после останова** — установку поднимают по регламенту пуска;
* **ступенька режима** (температура, нагрузка, давление) вне пусков, у которой
  есть признак, что это не реакция на серу.

Как отличить ступеньку-реакцию от внешней. Прямой разметки нет, поэтому смотрим на
то, что видел оператор до ступеньки: наклон поточного анализатора за 12 часов и
последний опубликованный анализ. У каждой переменной реакция на рост серы своего
знака — температура и давление растут, нагрузка падает. Ступенька в сторону
реакции при растущей сере похожа на реакцию и из списка чистых выпадает.
Ступенька ПРОТИВ тренда серы реакцией быть не может. Ступенька при ровной сере —
вероятно, плановая.

Это не доказательство внешнего характера, а отсев очевидной обратной связи, и
метки это называют прямо. Измеренное основание критерия: у ступенек температуры
перевес «похоже на реакцию» над «против тренда серы» почти вчетверо (181 против
47), у нагрузки (78 против 83) и давления (73 против 75) перевеса нет — там
изменения в среднем внешние. Критерий различает ровно там, где должен.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nefte.models.regime import FEED, PRESSURE, REACTOR_TEMPS, outage_mask

# Ступенька: разность средних за 6 часов после и 6 часов до. Шесть часов — чуть
# больше постоянной времени канала серы (4.6 ч): короче окно ловит шум, длиннее —
# склеивает соседние манёвры.
STEP_WINDOW_H = 6
# Соседние ступеньки одной переменной разносим не меньше чем на сутки: иначе один
# манёвр, растянутый на несколько часов, засчитывается несколько раз.
MIN_SEPARATION_H = 24
# Пороги — около p90–p95 распределения ступенек на работающей установке: реже —
# теряется выборка, чаще — это уже колебания регулятора, а не изменение режима.
THRESHOLDS = {"wabt": 2.0, "feed": 13.0, "pressure": 0.1}
UNITS = {"wabt": "°C", "feed": "м3/ч", "pressure": "МПа"}
NAMES = {"wabt": "температура реакторного блока", "feed": "нагрузка по сырью",
         "pressure": "давление"}
# Знак реакции на РОСТ серы: оператор поднимает температуру и давление, снижает
# нагрузку. Реакция — когда знак ступеньки совпадает со знаком тренда серы,
# умноженным на этот множитель.
RESPONSE_SIGN = {"wabt": +1, "feed": -1, "pressure": +1}
# Окно выхода на режим после останова: трое суток. В них ступеньки — это регламент
# пуска, и они идут отдельным типом, а не как ступеньки режима.
STARTUP_WINDOW_H = 72
# Вокруг смены катализатора ступеньки режима не берём: там меняется сама
# активность, и отклик на режим смешан с ней.
CATALYST_GUARD_DAYS = 7
TREND_HOURS = 12
# Доля валидных часов ПАК в окне, ниже которой контекст серы считается неизвестным.
MIN_VALID_SHARE = 0.6
# Какое сопутствующее изменение другой переменной портит эпизод. Сначала порогом
# было «выше её собственного порога» — и чистым прошёл подъём WABT на 6.4 °C при
# одновременном росте нагрузки на 10.3 м3/ч (порог 13). Четыре процента нагрузки
# при глубине превращения около 7 меняют серу сравнимо с этими шестью градусами,
# и отклик на одну из двух переменных такой эпизод уже не выделяет. Половина порога
# оставляет 80 эпизодов из 180; треть — 44, и на валидации их почти не остаётся.
CONCURRENT_SHARE = 0.5


@dataclass(frozen=True)
class NoiseBand:
    """Шумовая полоса наклона серы: какой наклон считать «ровным», а какой явным.

    Берётся из распределения того же наклона в случайные часы: «ровно» — не
    больше нижней четверти, «явно» — больше медианы. Промежуток между ними
    честно неопределённый.
    """

    flat: float
    clear: float


def hourly_regime(ht: pd.DataFrame) -> pd.DataFrame:
    """Часовой режим работающей установки: температура, нагрузка, давление.

    Остановы вырезаются ретроспективной маской — это разбор истории, и первые
    часы простоя тоже были остановом.
    """
    down = outage_mask(ht[FEED], min_outage_hours=6.0, retrospective=True)
    working = ht.where(~down)
    temps = [t for t in REACTOR_TEMPS if t in ht.columns]
    frame = pd.DataFrame({
        "wabt": working[temps].mean(axis=1),
        "feed": working[FEED],
        "pressure": working[PRESSURE] if PRESSURE in ht.columns else np.nan,
    }, index=ht.index)
    return frame.resample("1h").mean()


def step_series(hourly: pd.DataFrame, window: int = STEP_WINDOW_H) -> pd.DataFrame:
    """Разность средних за ``window`` часов после и до каждого часа."""
    before = hourly.rolling(window, min_periods=window // 2).mean()
    after = before.shift(-window)
    return after - before


def detect_steps(steps: pd.Series, threshold: float,
                 separation: int = MIN_SEPARATION_H) -> pd.Series:
    """Ступеньки: локальные максимумы |разности| не ниже порога, разнесённые на сутки."""
    magnitude = steps.abs()
    local_max = magnitude == magnitude.rolling(2 * separation + 1, center=True,
                                               min_periods=1).max()
    return steps[(magnitude >= threshold) & local_max]


def sulfur_trend(pak_hourly: pd.Series, hours: int = TREND_HOURS) -> pd.Series:
    """Наклон серы за ``hours`` часов до момента: среднее последних 3 ч минус первых 3 ч."""
    tail = pak_hourly.rolling(3, min_periods=2).mean()
    head = pak_hourly.shift(hours - 3).rolling(3, min_periods=2).mean()
    return tail - head


def noise_band(pak_hourly: pd.Series, hours: int = TREND_HOURS) -> NoiseBand:
    trend = sulfur_trend(pak_hourly, hours).abs().dropna()
    return NoiseBand(flat=float(trend.quantile(0.25)), clear=float(trend.quantile(0.5)))


def classify(variable: str, delta: float, trend: float | None, pak_level: float | None,
             lims_last: float | None, spec: float, band: NoiseBand) -> str:
    """Метка ступеньки: насколько вероятно, что это НЕ реакция на серу.

    Порядок проверок важен. Сера выше предела перед ступенькой — повод
    вмешаться независимо от наклона, поэтому такая ступенька не считается
    внешней, даже если наклон ровный.
    """
    if trend is None or trend != trend:
        return "нет данных о сере"
    above = any(v is not None and v == v and v > spec for v in (pak_level, lims_last))
    if above:
        return "сера выше предела"
    if abs(trend) <= band.flat:
        return "сера ровная"
    if abs(trend) <= band.clear:
        return "неопределённо"
    reaction = np.sign(delta) == RESPONSE_SIGN[variable] * np.sign(trend)
    return "похоже на реакцию" if reaction else "против тренда серы"


# Метки, при которых ступенька годится для оценки отклика. «Против тренда» —
# сильный признак, «ровная» — слабый; оба в списке, но различимы.
CLEAN_LABELS = ("против тренда серы", "сера ровная")


def outage_intervals(feed: pd.Series, min_hours: float = 6.0) -> list[tuple]:
    down = outage_mask(feed, min_outage_hours=min_hours, retrospective=True)
    blocks = (down != down.shift()).cumsum()
    return [(part.index[0], part.index[-1]) for _, part in feed.groupby(blocks)
            if down.loc[part.index[0]]]


def near(ts: pd.Timestamp, marks: list[pd.Timestamp], before: pd.Timedelta,
         after: pd.Timedelta) -> bool:
    return any(mark - before <= ts <= mark + after for mark in marks)


def regime_step_episodes(hourly: pd.DataFrame, pak_hourly: pd.Series,
                         lims_published: pd.Series, spec: float,
                         outage_ends: list[pd.Timestamp],
                         catalyst_changes: list[pd.Timestamp],
                         thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    """Ступеньки режима вне пусков и смен катализатора, с контекстом серы.

    ``lims_published`` — анализы, уже сдвинутые на задержку публикации: оператор
    видит анализ не в момент отбора, а когда он опубликован.
    """
    thresholds = thresholds or THRESHOLDS
    steps = step_series(hourly)
    pak_valid = pak_hourly.notna().astype(float)
    trend = sulfur_trend(pak_hourly)
    pak_level = pak_hourly.rolling(TREND_HOURS, min_periods=TREND_HOURS // 2).mean()
    valid_before = pak_valid.rolling(TREND_HOURS, min_periods=1).mean()
    valid_after = pak_valid[::-1].rolling(TREND_HOURS, min_periods=1).mean()[::-1].shift(-1)
    band = noise_band(pak_hourly)
    startup = pd.Timedelta(hours=STARTUP_WINDOW_H)
    guard = pd.Timedelta(days=CATALYST_GUARD_DAYS)

    rows = []
    for variable, threshold in thresholds.items():
        if variable not in steps or steps[variable].isna().all():
            continue
        for ts, delta in detect_steps(steps[variable], threshold).items():
            if near(ts, outage_ends, pd.Timedelta(0), startup):
                continue
            if near(ts, catalyst_changes, pd.Timedelta(0), guard):
                continue
            enough = valid_before.get(ts, 0.0) >= MIN_VALID_SHARE
            tr = float(trend.get(ts)) if enough and trend.get(ts) == trend.get(ts) else None
            level = float(pak_level.get(ts)) if enough else None
            published = lims_published.loc[:ts]
            lims_last = float(published.iloc[-1]) if len(published) else None
            label = classify(variable, float(delta), tr, level, lims_last, spec, band)
            others = {other: round(float(steps[other].get(ts)), 3)
                      for other in thresholds if other != variable
                      and steps[other].get(ts) == steps[other].get(ts)}
            concurrent = [other for other, value in others.items()
                          if abs(value) >= CONCURRENT_SHARE * thresholds[other]]
            rows.append({
                "момент": ts,
                "тип": "ступенька режима",
                "переменная": variable,
                "изменение": round(float(delta), 3),
                "единица": UNITS[variable],
                "одновременно": others,
                "одновременно заметно": concurrent,
                "наклон серы за 12 ч": None if tr is None else round(tr, 2),
                "ПАК до, мг/кг": None if level is None else round(level, 2),
                "последний опубликованный анализ": lims_last,
                "ПАК валиден после, доля": round(float(valid_after.get(ts, 0.0)), 2),
                "метка": label,
                "чистый": bool(label in CLEAN_LABELS and not concurrent
                               and valid_after.get(ts, 0.0) >= MIN_VALID_SHARE),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("момент").reset_index(drop=True)
    frame.attrs["noise_band"] = band
    return frame


def startup_episodes(hourly: pd.DataFrame, outages: list[tuple],
                     hours: int = STARTUP_WINDOW_H) -> pd.DataFrame:
    """Выходы на режим: чем установка жила в первые трое суток после останова.

    Если в окне установка снова встала, уровня «через трое суток» нет — это
    называется прямо, а не оставляется пустым числом.
    """
    rows = []
    starts = [start for start, _ in outages]
    for start, end in outages:
        window = hourly.loc[end:end + pd.Timedelta(hours=hours)]
        settled = hourly.loc[end + pd.Timedelta(hours=hours):
                             end + pd.Timedelta(hours=hours + 48)]
        if window.dropna(how="all").empty:
            continue
        first = window.dropna(how="all").iloc[:6].mean()
        horizon = end + pd.Timedelta(hours=hours + 48)
        again = any(end < other <= horizon for other in starts)
        rows.append({
            "момент": end,
            "тип": "выход на режим",
            "простой, ч": round((end - start).total_seconds() / 3600, 0),
            "первые 6 ч": {k: None if v != v else round(float(v), 2) for k, v in first.items()},
            "через 3 сут": {k: None if v != v else round(float(v), 2)
                            for k, v in settled.mean().items()},
            "повторный останов в окне": bool(again),
        })
    return pd.DataFrame(rows)
