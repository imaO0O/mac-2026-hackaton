"""Синхронизация источников и признаки — без утечки из будущего.

Три правила, которые здесь зашиты (все три требует ТЗ):
1. Совмещение источников только по времени и только «назад» (``direction="backward"``):
   в момент t можно знать лишь то, что уже измерено.
2. Вместе со значением всегда возвращается его ВОЗРАСТ в часах — свежесть анализа
   часть решения, а не деталь реализации.
3. Разбиение train/val/test — только по времени, с эмбарго на границе.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nefte.config import load_config


def asof_features(index: pd.DatetimeIndex, series: pd.Series, name: str,
                  max_age_hours: float | None = None,
                  allow_exact_matches: bool = True) -> pd.DataFrame:
    """Последнее известное значение ``series`` на каждый момент ``index`` + его возраст.

    Parameters
    ----------
    allow_exact_matches : если False, значение, измеренное РОВНО в момент t, не
        используется. Это критично, когда ``series`` — тот же показатель, что и
        целевая переменная: иначе модель «предсказывает» анализ, зная его.

    Returns
    -------
    DataFrame с колонками ``{name}`` и ``{name}_age_h``. Если значение старше
    ``max_age_hours``, оно обнуляется в NaN, но возраст сохраняется — агент
    качества должен видеть, что данные устарели, а не думать, что их нет.
    """
    left = pd.DataFrame({"ts": pd.DatetimeIndex(index)}).sort_values("ts")
    right = (series.dropna().rename("value").reset_index()
             .rename(columns={series.index.name or "index": "ts"})
             .sort_values("ts"))
    right.columns = ["ts", "value"]

    merged = pd.merge_asof(left, right.assign(src_ts=right["ts"]), on="ts",
                           direction="backward",
                           allow_exact_matches=allow_exact_matches)
    # возраст = t - время последнего фактического измерения
    age_h = (merged["ts"] - merged["src_ts"]).dt.total_seconds() / 3600.0

    out = pd.DataFrame({name: merged["value"].to_numpy(),
                        f"{name}_age_h": age_h.to_numpy()},
                       index=pd.DatetimeIndex(merged["ts"]))
    if max_age_hours is not None:
        out.loc[out[f"{name}_age_h"] > max_age_hours, name] = np.nan
    return out.reindex(index)


def known_from(series: pd.Series, delay_hours: float) -> pd.Series:
    """Тот же ряд, но с меткой времени, когда значение стало ИЗВЕСТНО.

    Метка ЛИМС — момент отбора пробы; результат появляется в системе позже
    (организаторы: до 4 часов). Сдвиг индекса вперёд превращает «когда измерено»
    в «когда стало видно оператору», и все as-of соединения после этого честны
    автоматически.

    Применять только ко ВХОДАМ решения. К факту, с которым сверяются прогоны,
    сдвиг не применяется: превышение спецификации случилось в момент отбора.
    """
    if not delay_hours:
        return series
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=float(delay_hours))
    return shifted


# Здесь лежали ещё три помощника — ``add_lags``, ``add_rollings`` и
# ``make_supervised``. Все три не вызывались ниоткуда, и все три дублировали то,
# что матрица признаков делает у себя (`models/dataset.py`).
#
# Удалены не за мёртвость, а за то, что дублирование в этом файле уже один раз
# выстрелило: ``wabt`` нормировала веса по полному набору столбцов и при отказе
# датчика давала 233 °C вместо 350, пока живой расчёт агентов считал правильно.
#
# У ``make_supervised`` был отдельный повод. Она сдвигала ЦЕЛЬ вперёд
# (``y.shift(-h)``), то есть помечала строку моментом признаков. Матрица делает
# обратное: строка помечается моментом АНАЛИЗА, а признаки берутся за H часов до
# него. Разница не косметическая — при делении на train/val/test одна и та же
# пара «признаки, анализ» оказывается по разные стороны границы. Соглашение
# выбрано и записано (docs/PLAN.md, «горизонт сдвигает признаки назад, а не цель
# вперёд»), и держать рядом функцию с противоположным значило оставлять грабли.


def wabt(temps: pd.DataFrame, weights: list[float] | None = None) -> pd.Series:
    """Средневзвешенная температура слоя (прокси жёсткости режима гидроочистки).

    Прямой разметки состояния катализатора в пакете нет, поэтому используем
    общепринятый прокси. Это ДОПУЩЕНИЕ, оно объявлено в docs/DATA_NOTES.md.

    Веса нормируются по ДОСТУПНЫМ значениям, а не по всем столбцам. Раньше здесь
    стоял ``np.nansum(values * w)`` с весами, нормированными по полному набору:
    пропущенная температура входила в сумму нулём, а её вес из знаменателя не
    уходил. На трёх датчиках отказ одного давал 233 °C вместо 350 — ошибку в
    117 °C, и не в виде пропуска, а в виде правдоподобного числа.

    Вживую это не стреляло: и признаки режима, и индекс тяжести считают WABT
    своим ``ht[temps].mean(axis=1)``, который пропуски пересчитывает правильно.
    Но функция лежала в ``data/features.py`` и выглядела канонической.
    """
    values = temps.to_numpy(dtype=float)
    if weights is None:
        w = np.full(values.shape[1], 1.0 / values.shape[1])
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    known = ~np.isnan(values)
    # знаменатель — сумма весов ТОЛЬКО доступных столбцов в каждой строке
    denom = (known * w).sum(axis=1)
    numer = np.nansum(np.where(known, values, 0.0) * w, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, numer / denom, np.nan)
    return pd.Series(out, index=temps.index, name="wabt")


def time_split(index: pd.DatetimeIndex, cfg: dict | None = None) -> dict[str, pd.Series]:
    """Маски train/val/test по времени с эмбарго на границах.

    Случайное перемешивание запрещено ТЗ: любые лаги/скользящие окна протекли бы
    из будущего в прошлое.
    """
    cfg = cfg or load_config()
    sp = cfg["split"]
    emb = pd.Timedelta(hours=int(sp.get("embargo_hours", 0)))
    idx = pd.DatetimeIndex(index)

    def _mask(bounds, shrink_left: bool) -> pd.Series:
        lo, hi = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]) + pd.Timedelta(days=1)
        if shrink_left:
            lo = lo + emb
        return pd.Series((idx >= lo) & (idx < hi), index=idx)

    return {
        "train": _mask(sp["train"], False),
        "val": _mask(sp["val"], True),
        "test": _mask(sp["test"], True),
    }


