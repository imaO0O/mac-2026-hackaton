"""Разбор прогноза по группам каналов: почему модель ждёт именно столько серы.

Вклад каждого признака в КОНКРЕТНЫЙ прогноз CatBoost считает точно (SHAP для
деревьев — не приближение: сумма вкладов плюс база в точности равна прогнозу),
и стоит это 0.04 мс на решение. Признаков 55, и по одному они оператору не нужны;
складываем их в группы, по которым в модели зашита физика (`PHYSICS_MONOTONE`),
плюс «сырьё с АВТ» — связка, ради которой в системе есть отдельный агент АВТ.

**Почему строка в карточке появляется не всегда.** Разбор померен на тесте
(`scripts/check_risk_attribution.py`, отчёт `reports/risk_attribution.json`):
ведущая группа — «показания по сере» в 85 % решений, а если убрать её как
бесполезную (оператор и так видит показание прибора первой строкой карточки) —
«сырьё с АВТ» в 81 %. Постоянная строка «виновато сырьё» — это фон, который
перестают читать, поэтому оба варианта записаны как измеренные отказы.

Остаётся то, что несёт информацию: сказать, когда причина НЕОБЫЧНА — когда ведёт
не сырьё, а канал, на который оператор может повлиять уставками. Это 19 % решений
на тесте и 31 % на валидации; правило («не реже 5 % и не чаще половины») записано
до подстановки чисел.

Порога по величине вклада нет, и это измеренное решение, а не упущение. Порог
пробовали привязать к действию: показывать строку, только если вклад не меньше
того, что даёт один градус уставки (0.3 мг/кг — нижняя граница отклика, измеренного
на практике установки). С таким порогом строка исчезает — 0.0 % решений на тесте и
0.1 % на валидации. Значит, «необычные» каналы ведут прогноз слабее одного градуса
действия, и правильный ответ — не молчать, а сказать это в карточке прямо.
"""
from __future__ import annotations

import pandas as pd

# Группа, которая просто повторяет измеренное значение: в разборе не участвует.
MEASURED = "показания по сере"
# Что ведёт обычно — измерено на тесте, см. reports/risk_attribution.json.
USUAL = "сырьё с АВТ"
OTHER = "прочая телеметрия"

GROUPS: list[tuple[str, tuple[str, ...]]] = [
    (MEASURED, ("ht_Q21", "pak_sulfur", "lims_sulfur_prev", "ht_Q20")),
    (USUAL, ("avt_", "vak_", "lims_feed_sulfur")),
    ("режим реактора", ("reg_wabt", "reg_kinetic", "reg_drive", "reg_dt_react",
                        "ht_T5", "ht_T11", "ht_T16", "ht_T18")),
    ("водород", ("reg_h2", "reg_makeup", "ht_P13", "ht_F25")),
    ("нагрузка и квенч", ("ht_F26", "ht_F22", "ht_F14", "reg_quench")),
    ("свежесть данных", ("feature_age_h",)),
]

# Группы, на которые оператор влияет уставками: ими карточка может закончиться
# действием, а не наблюдением.
CONTROLLABLE = ("режим реактора", "водород", "нагрузка и квенч")


def group_of(name: str) -> str:
    """Имя группы для признака; неизвестные каналы — в «прочую телеметрию»."""
    for title, prefixes in GROUPS:
        if any(name.startswith(prefix) for prefix in prefixes):
            return title
    return OTHER


def grouped_contributions(model, row: pd.DataFrame) -> tuple[pd.Series, float] | None:
    """Вклады по группам (мг/кг) и база модели для ОДНОЙ строки признаков.

    ``None`` — если модель не умеет отдавать вклады (персистенция, чужая модель
    или строка не собралась): разбор не обязателен, и карточка обойдётся без него.
    """
    booster = (getattr(model, "models", None) or {}).get("q50")
    if booster is None or row is None or row.empty:
        return None
    features = list(getattr(model, "features", []))
    if not features or any(f not in row.columns for f in features):
        return None
    try:
        from catboost import Pool

        shap = booster.get_feature_importance(Pool(row[features].head(1)),
                                              type="ShapValues")
    except Exception:       # noqa: BLE001 — карточка не должна падать из-за разбора
        return None
    contrib = pd.Series(shap[0, :-1], index=features)
    base = float(shap[0, -1]) + float(getattr(model, "y_offset", 0.0))
    return contrib.groupby(group_of).sum(), base


def cause_from_groups(groups, usual: str = USUAL) -> dict | None:
    """То же, что unusual_cause, но по УЖЕ посчитанным вкладам групп.

    Агент качества кладёт вклады в поле `drivers` контракта, и второй раз считать
    их в оркестраторе незачем: разбор обязан объяснять тот же прогноз, который
    ушёл в решение, а не пересчитанный по другой строке признаков.
    """
    rest = {name: float(value) for name, value in dict(groups).items()
            if name != MEASURED}
    if not rest:
        return None
    leader = max(rest, key=lambda name: abs(rest[name]))
    if leader == usual:
        return None
    return {
        "группа": str(leader),
        "вклад, мг/кг": round(float(rest[leader]), 2),
        "обычная причина": usual,
        "управляемая": bool(leader in CONTROLLABLE),
        "вклады по группам": {str(k): round(float(v), 2) for k, v in
                              sorted(dict(groups).items(),
                                     key=lambda kv: -abs(kv[1]))},
    }


def unusual_cause(model, row: pd.DataFrame, usual: str = USUAL) -> dict | None:
    """Ведущая причина, если она не обычная. Иначе ``None`` — строки в карточке нет.

    Считаем ведущей группу с наибольшим вкладом ПО МОДУЛЮ среди всех, кроме
    повторяющей измерение: разбор нужен и тогда, когда канал тянет серу вниз —
    «держим режим, потому что глубина реактора пока перекрывает тяжёлое сырьё»
    такое же объяснение, как и обратное.
    """
    result = grouped_contributions(model, row)
    if result is None:
        return None
    grouped, base = result
    rest = grouped.drop(index=[MEASURED], errors="ignore")
    if rest.empty:
        return None
    leader = rest.abs().idxmax()
    if leader == usual:
        return None
    return {
        "группа": str(leader),
        "вклад, мг/кг": round(float(rest[leader]), 2),
        "обычная причина": usual,
        "управляемая": bool(leader in CONTROLLABLE),
        "база модели, мг/кг": round(float(base), 2),
        "вклады по группам": {str(k): round(float(v), 2)
                              for k, v in grouped.sort_values(
                                  key=abs, ascending=False).items()},
    }


def cause_sentence(cause: dict, action_scale: tuple[float, float]) -> str:
    """Строка карточки. Пустая строка — если причина обычная и говорить нечего.

    ``action_scale`` — измеренный отклик серы на один градус (мг/кг, практика
    установки; рабочая копия константы живёт в оркестраторе). Он здесь не порог,
    а МАСШТАБ: вклад надо с чем-то сравнить, иначе «+0.21 мг/кг» оператору ни о
    чём не говорит. Порог по величине проверялся и отклонён: если показывать
    строку только при вкладе не меньше одного градуса (0.3 мг/кг), она исчезает
    почти полностью — 0.0 % решений на тесте и 0.1 % на валидации
    (``reports/risk_attribution.json``). Вывод из этого не «поднять порог», а
    прямая фраза в карточке: канал ведёт прогноз, но на градус действия не тянет.
    """
    if not cause:
        return ""
    value = cause["вклад, мг/кг"]
    where = "вверх" if value > 0 else "вниз"
    low = float(action_scale[0])
    if abs(value) >= low:
        tail = ("на это можно ответить уставками"
                if cause["управляемая"] else "уставками это не правится")
    else:
        tail = (f"это меньше, чем даёт один градус ({low:g} мг/кг), — объяснение, "
                f"а не повод двигать уставки")
    return (f"В этот раз прогноз ведёт не {cause['обычная причина']}, "
            f"а {cause['группа']}: {value:+.2f} мг/кг {where}, {tail}.")
