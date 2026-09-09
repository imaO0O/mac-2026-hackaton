"""Признаки режима реактора гидроочистки.

Зачем. Модель качества обучалась на «сырых» тегах и почти не реагировала на
уставки: сдвиг реакторной температуры на 2 °C менял прогноз на 0.006 мг/кг, и
оптимизатор выбирал из неразличимых вариантов. Причина в том, что обессеривание
зависит не от температуры как таковой, а от сочетания температуры, времени
контакта и парциального давления водорода. Бустинг такие сочетания сам не
собирает — их нужно дать явно.

Что считаем (всё из выданных тегов, физика стандартная для гидроочистки):

* **WABT** — средняя температура реакторного блока (`T5`, `T6`, `T11`);
* **экзотерма** `T11 − T5` — косвенная мера глубины реакций;
* **объёмная скорость** — расход сырья `F26`: чем выше, тем меньше время контакта;
* **кратность газ/сырьё** `F2 / F26` и подпитка водородом `P24 / F26`;
* **квенч** `F15 / F26` — съём тепла между слоями;
* **парциальное давление водорода** `P13 × F2 / F26` — прокси, чистоту ВСГ не меряют;
* **кинетический индекс** `exp(−Ea / R·T) / F26` — скорость реакции, отнесённая
  ко времени контакта. Именно он связывает температуру и нагрузку так, как это
  происходит в реакторе, а не по отдельности;
* **наработка от начала цикла** и **дрейф WABT** — прокси дезактивации катализатора.

Допущения объявлены явно: энергия активации 100 кДж/моль (типовая для ГДС, в
пакете её нет), объём катализатора неизвестен, поэтому объёмная скорость входит
пропорционально расходу сырья, чистота ВСГ не измеряется.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ДОПУЩЕНИЕ: типовая энергия активации гидрообессеривания, кДж/моль.
ACTIVATION_ENERGY_KJ = 100.0
GAS_CONSTANT_KJ = 0.008314          # кДж/(моль·К)

REACTOR_TEMPS = ["T5", "T6", "T11"]
FEED = "F26"            # расход сырья на установку (объёмный)
RECYCLE_GAS = "F2"      # газовая схема, расход от ЦК-201
MAKEUP_H2 = "P24"       # расход свежего ВСГ с КЦА
QUENCH = "F15"          # расход квенча в Р-202
PRESSURE = "P13"        # давление в реакторном блоке

# Теги, изменение которых требует пересчёта признаков режима.
MOVE_SENSITIVE = set(REACTOR_TEMPS) | {FEED, RECYCLE_GAS, MAKEUP_H2, QUENCH, PRESSURE}

INSTANT_FEATURES = [
    "reg_wabt", "reg_dt_react", "reg_h2_oil", "reg_makeup_h2_oil",
    "reg_quench_ratio", "reg_h2_partial", "reg_kinetic", "reg_drive",
]


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Отношение с защитой от нулевого и отрицательного знаменателя."""
    den = den.where(den > 1e-6)
    return num / den


def instant_features(ht: pd.DataFrame) -> pd.DataFrame:
    """Признаки, полностью определяемые мгновенными значениями тегов.

    Именно они пересчитываются, когда оптимизатор пробует новую уставку.
    """
    temps = [t for t in REACTOR_TEMPS if t in ht.columns]
    out = pd.DataFrame(index=ht.index)

    if temps:
        out["reg_wabt"] = ht[temps].mean(axis=1)
    if "T11" in ht.columns and "T5" in ht.columns:
        out["reg_dt_react"] = ht["T11"] - ht["T5"]

    feed = ht[FEED] if FEED in ht.columns else None
    if feed is not None:
        # На остановах и провалах нагрузки отношения к расходу сырья не имеют
        # смысла: делить на почти ноль — значит породить выброс, а не признак.
        floor = float(feed.median()) * 0.1 if feed.notna().any() else 0.0
        feed = feed.where(feed > max(floor, 1e-6))
        if RECYCLE_GAS in ht.columns:
            out["reg_h2_oil"] = _safe_ratio(ht[RECYCLE_GAS], feed)
        if MAKEUP_H2 in ht.columns:
            out["reg_makeup_h2_oil"] = _safe_ratio(ht[MAKEUP_H2], feed)
        if QUENCH in ht.columns:
            out["reg_quench_ratio"] = _safe_ratio(ht[QUENCH], feed)
        if PRESSURE in ht.columns and "reg_h2_oil" in out:
            # прокси парциального давления водорода: чистоту ВСГ не измеряют
            out["reg_h2_partial"] = ht[PRESSURE] * out["reg_h2_oil"]

    if "reg_wabt" in out and feed is not None:
        kelvin = out["reg_wabt"] + 273.15
        rate = np.exp(-ACTIVATION_ENERGY_KJ / (GAS_CONSTANT_KJ * kelvin.where(kelvin > 0)))
        # скорость реакции, отнесённая ко времени контакта; масштаб произволен
        out["reg_kinetic"] = _safe_ratio(rate * 1e12, feed)
        if "reg_h2_partial" in out:
            out["reg_drive"] = out["reg_kinetic"] * out["reg_h2_partial"]

    return out


def history_features(ht: pd.DataFrame, steps_per_hour: int = 6) -> pd.DataFrame:
    """Признаки, требующие истории: наработка и дрейф режима.

    Разметки остановов и замен катализатора в пакете нет, поэтому останов
    определяется по провалу расхода сырья ниже 20 % от медианы дольше шести часов.
    Дрейф WABT — стандартный прокси дезактивации: чтобы держать качество,
    температуру приходится поднимать.
    """
    out = pd.DataFrame(index=ht.index)
    if FEED not in ht.columns:
        return out

    out["reg_run_hours"] = hours_since_outage(ht[FEED], min_outage_hours=6,
                                              steps_per_hour=steps_per_hour)

    temps = [t for t in REACTOR_TEMPS if t in ht.columns]
    if temps:
        wabt = ht[temps].mean(axis=1)
        window = 30 * 24 * steps_per_hour
        out["reg_wabt_dev30"] = wabt - wabt.rolling(window, min_periods=window // 4).median()
        out["reg_wabt_slope7"] = wabt - wabt.shift(7 * 24 * steps_per_hour)
    return out


def outage_mask(feed: pd.Series, min_outage_hours: float = 6.0,
                steps_per_hour: int = 6, level: float = 0.2) -> pd.Series:
    """Маска останова: расход сырья ниже ``level`` от медианы дольше заданного срока.

    Считать нужно по СЫРОМУ сигналу. Детектор достоверности справедливо убирает
    замороженный на нуле расход как «не живое измерение», но именно этот
    замороженный ноль и есть факт останова: на очищенных данных из десяти
    остановов виден один.
    """
    down = feed < feed.median() * level
    block = (down != down.shift()).cumsum()
    long_enough = down.groupby(block).transform("size") >= min_outage_hours * steps_per_hour
    return down & long_enough


def hours_since_outage(feed: pd.Series, min_outage_hours: float = 6.0,
                       steps_per_hour: int = 6) -> pd.Series:
    """Часы с последнего останова заданной длительности."""
    shutdown = outage_mask(feed, min_outage_hours, steps_per_hour)
    index = feed.index.to_series()
    marks = index.where(shutdown).ffill()
    since = (index - marks).dt.total_seconds() / 3600
    return since.fillna((index - feed.index[0]).dt.total_seconds() / 3600)


def regime_features(ht: pd.DataFrame, with_history: bool = True,
                    steps_per_hour: int = 6) -> pd.DataFrame:
    """Все признаки режима по «сырым» тегам установки 24-2000."""
    parts = [instant_features(ht)]
    if with_history:
        parts.append(history_features(ht, steps_per_hour))
    return pd.concat(parts, axis=1).astype("float32")


# --------------------------------------------------------------------------- #
# пересчёт при изменении уставок
# --------------------------------------------------------------------------- #

def apply_moves_to_rows(rows: pd.DataFrame, moves: dict[str, float],
                        relative: bool = False, prefix: str = "ht_") -> pd.DataFrame:
    """Возвращает копию строк признаков с новыми уставками и пересчитанным режимом.

    Без этого пересчёта подмена одной колонки бессмысленна: модель опирается на
    производные признаки, и они остались бы от старого режима.

    Дополнительно сдвигается часовое скользящее среднее изменённого тега:
    рекомендация подразумевает, что новая уставка удерживается, и за час среднее
    подтянется к ней. Окна 6 и 24 часа не трогаем — это уже история.
    """
    out = rows.copy()
    raw = {}
    for tag in MOVE_SENSITIVE:
        column = f"{prefix}{tag}"
        if column in out.columns:
            raw[tag] = out[column].astype("float64")

    for tag, value in moves.items():
        column = f"{prefix}{tag}"
        if column not in out.columns:
            continue
        new = raw[tag] + value if relative else pd.Series(value, index=out.index)
        delta = new - raw[tag]
        raw[tag] = new
        out[column] = new.astype(out[column].dtype)
        hourly = f"{column}_mean6"
        if hourly in out.columns:
            out[hourly] = (out[hourly].astype("float64") + delta).astype(out[hourly].dtype)

    if not raw:
        return out

    recomputed = instant_features(pd.DataFrame(raw, index=out.index))
    for column in recomputed.columns:
        if column in out.columns:
            out[column] = recomputed[column].astype(out[column].dtype)
    return out
