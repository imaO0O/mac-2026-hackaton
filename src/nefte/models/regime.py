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

# Как температура Р-202 (T6) отвечает на управляющие уставки. ИЗМЕРЕНО, а не
# принято: совместная регрессия первых разностей на обучающем периоде, 126 753
# отсчёта по 10 минут, остановы исключены, R² = 0.63.
#
#     ΔT6 = 0.72·ΔT5 + 0.18·ΔT11
#
# Расход сырья и давление в неё не входят: их коэффициенты 0.014 и −0.079 при
# корреляции разностей 0.08 и −0.01, то есть неотличимы от нуля.
#
# Зачем это нужно. T6 — не уставка, её никто не задаёт напрямую: это температура
# НИЖЕ по потоку, следствие того, что оператор сделал с T5 и T11. Но в формулу
# виртуального анализатора Т95 входит именно T6, с коэффициентом 0.50. Пока связь
# не была смоделирована, проверка Т95 в оптимизаторе оказывалась мёртвой: система
# двигала T5, формула этого не видела, и «наш вклад в Т95» выходил ровно нулевым
# во всех точках имитации. Это и обнаружилось на прогоне замкнутого контура.
#
# Полная цепочка получается такой: ΔТ95 = 0.50·ΔT6 = 0.36·ΔT5 + 0.09·ΔT11.
# Шаг в 2 °C по T5 поднимает Т95 на 0.72 °C — при типичном запасе до предела
# в 3–5 °C это уже не мелочь.
T6_RESPONSE = {"T5": 0.72, "T11": 0.18}


def implied_t6(current: dict[str, float | None], moves: dict[str, float]) -> float | None:
    """Температура Р-202 при заданных уставках.

    Если T6 задан явно — берём его. Иначе выводим из измеренного отклика на T5 и
    T11. Возвращает None, когда исходной T6 нет: подставлять что-то вместо
    неизвестной температуры нельзя.
    """
    base = current.get("T6")
    if base is None or base != base:
        return None
    if "T6" in moves:
        return float(moves["T6"])
    delta = 0.0
    for tag, gain in T6_RESPONSE.items():
        new = moves.get(tag)
        now = current.get(tag)
        if new is None or now is None or now != now:
            continue
        delta += gain * (float(new) - float(now))
    return float(base) + delta

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
        # Уставка может быть и не из тех, что влияют на признаки режима (MOVE_SENSITIVE).
        # Раньше такой тег ронял расчёт по KeyError: обращение к raw[tag] шло раньше,
        # чем проверка, что тег вообще туда попал. Сейчас управляющие теги все
        # «чувствительные», но добавление любого другого сломало бы оптимизатор.
        current = raw.get(tag)
        if current is None:
            current = out[column].astype("float64")
        new = current + value if relative else pd.Series(value, index=out.index)
        delta = new - current
        if tag in raw:
            raw[tag] = new
        out[column] = new.astype(out[column].dtype)
        hourly = f"{column}_mean6"
        if hourly in out.columns:
            out[hourly] = (out[hourly].astype("float64") + delta).astype(out[hourly].dtype)

    if not raw:
        return out

    # T6 никто не задаёт уставкой: температура Р-202 — следствие того, что сделали
    # с T5 и T11. А в признаки режима она входит через WABT (среднее по трём
    # реакторным температурам), и без пересчёта WABT отвечала на шаг +2 °C по T5
    # прибавкой 0.67 °C вместо 1.15 — то есть суррогат «режим → качество»
    # недооценивал эффект подъёма температуры на 40 %. Это главный путь принятия
    # решения, так что ошибка шла прямо в выбор варианта.
    #
    # Пересчитываем только если T6 не задана явно и есть чем: связь измерена
    # (T6_RESPONSE), но она про ИЗМЕНЕНИЕ, а не про уровень.
    if "T6" in raw and "T6" not in moves:
        shift = 0.0
        for tag, gain in T6_RESPONSE.items():
            if tag in moves and tag in raw:
                base = rows[f"{prefix}{tag}"].astype("float64")
                shift = shift + gain * (raw[tag] - base)
        if not isinstance(shift, float):
            raw["T6"] = raw["T6"] + shift
            column = f"{prefix}T6"
            if column in out.columns:
                out[column] = raw["T6"].astype(out[column].dtype)

    recomputed = instant_features(pd.DataFrame(raw, index=out.index))
    for column in recomputed.columns:
        if column in out.columns:
            out[column] = recomputed[column].astype(out[column].dtype)
    return out
