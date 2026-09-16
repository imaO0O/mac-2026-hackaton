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
* **кратность газ/сырьё** `F2 / F26` и подпитка водородом `F25 / F26`;
* **квенч** `F14 / F26` — съём тепла между слоями;
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

# Теги — по таблице организаторов 15.09 (configs/tags_2026-09-15.csv). До неё
# подпиткой водородом считался P24, квенчем — F15: так стояло в листе «КИП» пакета,
# где описания 24-2000 перемешаны со строками. По данным P24 — давление 0.585 МПа
# (К-201), F15 — 3400 «м3/ч» без связи с нагрузкой (corr с F26 0.21); свежий ВСГ —
# F25, 13 600 нм3/ч, квенч — F14, 6 т/ч. docs/DATA_NOTES.md §5б.
REACTOR_TEMPS = ["T5", "T6", "T11"]   # выход Р-201, вход Р-202, выход Р-202
# Нагрузка. В таблице организаторов F26 — «расход ГОДТ в цех №8, объёмный», а
# сырьё — F9 (т/ч) и F15 (м3/ч). По данным F9 = 0.850·F26 с точностью до 0.05 %
# (corr 1.000), а F15 с ними не связан. Как мера нагрузки F26 и F9 неразличимы;
# F26 оставлен, чтобы не менять уставку, которой оперирует оптимизатор. Вопрос
# организаторам записан.
FEED = "F26"
RECYCLE_GAS = "F2"      # газовая схема, расход от ЦК-201
MAKEUP_H2 = "F25"       # расход свежего ВСГ с КЦА, нм3/ч
QUENCH = "F14"          # расход квенча в Р-202, т/ч
PRESSURE = "P13"        # давление на входе Р-202

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


def reference_feed(feed: pd.Series, train_bounds: tuple[str, str] | None = None) -> float:
    """Опорный расход сырья, от которого считаются пороги останова и нормировки.

    Берётся медиана, и вопрос только в том, по какому куску истории. По ВСЕЙ —
    значит определение признака зависит от будущего: порог, по которому мы в 2024
    году решаем «установка стоит», посчитан в том числе по 2026-му. Это тот же
    вид утечки, который уже исправляли в нормировке severity.

    Величину измерили, прежде чем чинить: медиана по всей истории 256.24 против
    252.56 по обучающему периоду, расхождение 1.5 %, и вердикт «останов» меняется
    ровно у ОДНОГО отсчёта из 189 217. То есть утечка настоящая, а последствий у
    неё нет. Чиним ради правила, а не ради чисел — при другом пороге или другой
    установке разница может оказаться не такой безобидной.

    Без ``train_bounds`` поведение прежнее: это нужно вызывающим, у которых нет
    конфига под рукой.
    """
    scope = feed if train_bounds is None else feed.loc[train_bounds[0]:train_bounds[1]]
    if not len(scope) or not scope.notna().any():
        scope = feed
    return float(scope.median())


def instant_features(ht: pd.DataFrame,
                     train_bounds: tuple[str, str] | None = None) -> pd.DataFrame:
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
        floor = (reference_feed(feed, train_bounds) * 0.1
                 if feed.notna().any() else 0.0)
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


def history_features(ht: pd.DataFrame, steps_per_hour: int = 6,
                     train_bounds: tuple[str, str] | None = None) -> pd.DataFrame:
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
                                              steps_per_hour=steps_per_hour,
                                              train_bounds=train_bounds)

    temps = [t for t in REACTOR_TEMPS if t in ht.columns]
    if temps:
        wabt = ht[temps].mean(axis=1)
        window = 30 * 24 * steps_per_hour
        out["reg_wabt_dev30"] = wabt - wabt.rolling(window, min_periods=window // 4).median()
        out["reg_wabt_slope7"] = wabt - wabt.shift(7 * 24 * steps_per_hour)
    return out


# Чувствительность логарифма константы скорости к температуре, 1/К: E/(R·T²) при
# E = 100 кДж/моль и T ≈ 613 К. То же допущение, что и в ACTIVATION_ENERGY_KJ.
KINETIC_SENSITIVITY = 0.032


def normalized_wabt(wabt: pd.Series, known_sulfur: pd.Series,
                    feed_sulfur: pd.Series | float = 9000.0,
                    reference_mgkg: float = 10.0) -> pd.Series:
    """Температура, приведённая к опорному качеству продукта.

    Прокси дезактивации катализатора: при постоянном режиме её рост означает, что
    катализатор отдаёт меньше, чем отдавал.

    **Порядок реакции здесь решает всё, и первая версия была неверной.** Она
    считала ``WABT + ln(S/S0)/0.032``, то есть неявно принимала, что отношение
    сер равно отношению констант скорости. Для псевдопервого порядка это не так:
    ``kτ = ln(S_вх/S)``, и менять надо ГЛУБИНУ ПРЕВРАЩЕНИЯ, а не саму серу::

        ΔT = ln[ ln(S_вх/S_опорн) / ln(S_вх/S) ] / (E/(R·T²))

    Разница не косметическая: на удвоение серы первая версия давала 21.7 °C,
    правильная — 3.4 °C, в 6.4 раза меньше. Поймал участник 2. Первая версия
    соответствует ВТОРОМУ порядку (при глубоком превращении ``S ≈ 1/(kτ)``), и в
    литературе по глубокой гидроочистке второй порядок действительно применяют —
    из-за трудноудаляемого хвоста. Но остальной проект считает по первому
    (`models/kinetics.py` берёт ровно ``τ = ln(S_вх/S_вых)``), и два разных порядка
    в одном коде — это ровно тот вид тихого расхождения, который мы уже ловили в
    двух путях очистки данных. Поэтому здесь первый порядок.

    Выбрать порядок ПО ДАННЫМ не удалось, и это записано честно: критерий
    «верный порядок даёт самый гладкий ряд» вырожденный — шум монотонно падает с
    ростом порядка просто потому, что поправка стремится к нулю.

    **Про ``known_sulfur``.** Это сера ПРОДУКТА, то есть целевая переменная модели
    качества. Текущее значение подавать нельзя ни в каком виде — получится утечка.
    Допустим только последний ОПУБЛИКОВАННЫЙ анализ, сдвинутый на задержку
    публикации и взятый строго до момента t (в матрице это ``lims_sulfur_prev``).

    **В модель качества признак НЕ входит, и это измерено, а не решено.** На обоих
    горизонтах он отбирался (на горизонте 2 даже вытеснил ``reg_wabt_dev30``
    полностью, то есть как сигнал он лучше), но качества не прибавил: ничья на
    горизонте 0, проигрыш на горизонте 2. Причина объяснимая — это комбинация двух
    признаков, которые у модели уже есть, а бустинг такие находит сам.

    Оставлено ради агента надёжности: там нормированная температура не конкурирует
    с лабораторным рядом и может оказаться лучшим прокси ресурса катализатора.
    Идея участника 2.
    """
    known = known_sulfur.where(known_sulfur > 0)
    feed = (feed_sulfur if isinstance(feed_sulfur, pd.Series)
            else pd.Series(float(feed_sulfur), index=known.index))
    feed = feed.where(feed > 0)
    # глубина превращения сейчас и та, что нужна для опорного качества
    depth_now = np.log(feed / known)
    depth_ref = np.log(feed / float(reference_mgkg))
    valid = (depth_now > 0) & (depth_ref > 0)
    shift = np.log(depth_ref.where(valid) / depth_now.where(valid)) / KINETIC_SENSITIVITY
    return wabt + shift


def outage_mask(feed: pd.Series, min_outage_hours: float = 6.0,
                steps_per_hour: int = 6, level: float = 0.2,
                retrospective: bool = False,
                train_bounds: tuple[str, str] | None = None) -> pd.Series:
    """Маска останова: расход сырья ниже ``level`` от медианы дольше заданного срока.

    Считать нужно по СЫРОМУ сигналу. Детектор достоверности справедливо убирает
    замороженный на нуле расход как «не живое измерение», но именно этот
    замороженный ноль и есть факт останова: на очищенных данных из десяти
    остановов виден один.

    ``retrospective`` различает два РАЗНЫХ вопроса, которые до сих пор считались
    одинаково:

    * **что известно в момент t** (по умолчанию) — останов объявляется, когда
      порог длительности набран по прошлому. Так и только так маска годится для
      признаков и для среза оператора;
    * **какие отсчёты относились к останову** (``retrospective=True``) — весь
      эпизод целиком, включая первые часы. Так правильно ОТСЕИВАТЬ остановы из
      разбора истории: те отсчёты и правда были остановом.

    Раньше был только второй вариант, и он же шёл в признаки. Влияние измерено и
    оказалось нулевым: расхождение задевает 315 отсчётов из 189 217 (0.17 %), и НИ
    ОДИН лабораторный анализ в эту зону не попадает — во время останова пробы не
    отбирают. То есть утечка была настоящей, но до обучающих строк не доходила.
    Исправлено не ради метрик, а чтобы правило не выстрелило при другом пороге или
    другой сетке.
    """
    down = feed < reference_feed(feed, train_bounds) * level
    block = (down != down.shift()).cumsum()
    needed = min_outage_hours * steps_per_hour
    grouped = down.groupby(block)
    length = (grouped.transform("size") if retrospective
              else grouped.cumcount() + 1)
    return down & (length >= needed)


def hours_since_outage(feed: pd.Series, min_outage_hours: float = 6.0,
                       steps_per_hour: int = 6,
                       train_bounds: tuple[str, str] | None = None) -> pd.Series:
    """Часы с последнего останова заданной длительности."""
    shutdown = outage_mask(feed, min_outage_hours, steps_per_hour,
                           train_bounds=train_bounds)
    index = feed.index.to_series()
    marks = index.where(shutdown).ffill()
    since = (index - marks).dt.total_seconds() / 3600
    return since.fillna((index - feed.index[0]).dt.total_seconds() / 3600)


def regime_features(ht: pd.DataFrame, with_history: bool = True,
                    steps_per_hour: int = 6,
                    train_bounds: tuple[str, str] | None = None) -> pd.DataFrame:
    """Все признаки режима по «сырым» тегам установки 24-2000."""
    parts = [instant_features(ht, train_bounds=train_bounds)]
    if with_history:
        parts.append(history_features(ht, steps_per_hour,
                                      train_bounds=train_bounds))
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
