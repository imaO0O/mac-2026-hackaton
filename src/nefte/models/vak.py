"""Виртуальные анализаторы из справочника (лист ВАК) как признаки.

В пакете выдано 17 готовых формул, связывающих теги КИП с показателями качества:
плотность и разгонка фракции 240-350 и 350, вязкость 350-500, а также T90, T50,
I250, D15, температура помутнения, T95, ПТФ и НК гидроочищенного дизтоплива.

Зачем они в модели серы. Это экспертные комбинации тех же тегов, отобранные
технологами: они описывают состав и качество потока, а сера продукта зависит от
того же состояния установки. Дать их модели дешевле и честнее, чем надеяться,
что бустинг соберёт такие же комбинации сам из 983 обучающих строк.

Две ловушки самого справочника, обе обработаны явно:

* ``AVT6:240-350:CFPP`` записана с непарной скобкой — ``(F65/F32+F30))``.
  По аналогии с формулой D15 того же блока подразумевалось, скорее всего,
  ``(F65/(F32+F30))``, но домысливать чужую формулу мы не станем: она
  пропускается, и её имя возвращается в списке ``skipped``.
* ``24-2000:GODT:CloudPoint`` заканчивается висящей константой ``+0.00011`` —
  похоже, ячейка обрезана. Формула считается как есть, константа ни на что не
  влияет, но знать об этом надо.

Формулы, опирающиеся на ЛИМС (``GODT:D15`` и ``GODT:T95``), берут лабораторное
значение строго ДО текущего момента: в реальном времени результат сегодняшнего
анализа ещё не известен, а его подстановка была бы утечкой.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from nefte.data.loaders import load_vak_formulas, parse_vak_formula

# Ссылки на ЛИМС внутри формул → ряды из нашего длинного формата.
LIMS_TOKENS: dict[str, tuple[str, str]] = {
    "LIMS:24-2000.Pipeline.D15": ("LIMS_D15", "Гидроочистка|2|D15"),
    "LIMS:24-2000.Pipeline.95%.T": ("LIMS_T95", "Гидроочистка|2|95%.T"),
}

# Какой установке принадлежат короткие имена тегов в формуле.
BLOCK_UNIT = {"ЭЛОУ-АВТ-6": "avt", "24-2000": "ht"}

# Физически осмысленные диапазоны показателей — проверка «формула вообще считает
# то, что обещает». Часть формул справочника на выданных тегах даёт бессмыслицу
# (T90 в сотнях тысяч градусов, температура помутнения в тысячах): вероятная
# причина — то самое расхождение короткого имени тега и его смысла на 24-2000,
# о котором предупреждает ТЗ. Такие формулы в признаки не идут.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "D15": (700.0, 950.0),          # плотность при 15 °C, кг/м3
    "IBP": (100.0, 260.0),          # начало кипения, °C
    "T50": (180.0, 340.0),
    "T90": (250.0, 400.0),
    "T95": (280.0, 400.0),
    "EBP": (280.0, 420.0),          # конец кипения
    "I250": (0.0, 100.0),           # отгон, % об.
    "I350": (0.0, 100.0),
    "CFPP": (-40.0, 15.0),          # предельная температура фильтруемости, °C
    "CloudPoint": (-40.0, 15.0),
    "ViscosityK": (1.0, 12.0),      # кинематическая вязкость, сСт
}
# Доля значений внутри диапазона, ниже которой формула считается неприменимой.
MIN_PLAUSIBLE_SHARE = 0.8

_SAFE_GLOBALS = {"__builtins__": {}, "np": np}
_NAME_RE = re.compile(r"\b([A-Z]{1,4}[0-9]{1,3})\b")


def _block_unit(block: str) -> str:
    for prefix, unit in BLOCK_UNIT.items():
        if str(block).startswith(prefix):
            return unit
    return "ht"


def compile_formulas() -> tuple[list[dict], list[dict]]:
    """Готовит формулы к вычислению.

    Returns
    -------
    ``(usable, skipped)`` — пригодные формулы с скомпилированным кодом и список
    отброшенных с причиной. Отбрасываем молча только то, что не компилируется:
    молчаливая «починка» чужой формулы хуже честного пропуска.
    """
    usable, skipped = [], []
    for _, row in load_vak_formulas().iterrows():
        expr = parse_vak_formula(row["formula"])
        for token, (name, _series) in LIMS_TOKENS.items():
            expr = expr.replace(token, name)

        target = row["target"]
        try:
            code = compile(expr, f"<vak:{target}>", "eval")
        except SyntaxError as err:
            skipped.append({"target": target, "reason": f"не компилируется: {err.msg}",
                            "formula": row["formula"]})
            continue

        usable.append({
            "target": target,
            "column": "vak_" + target.replace(":", "_").replace("-", "_"),
            "unit": _block_unit(row["block"]),
            "code": code,
            "expr": expr,
            "tags": sorted(set(_NAME_RE.findall(expr))),
            "uses_lims": bool(row["uses_lims"]),
        })
    return usable, skipped


def evaluate(avt: pd.DataFrame, ht: pd.DataFrame,
             lims_context: dict[str, pd.Series] | None = None,
             check_plausibility: bool = True,
             train_bounds: tuple[str, str] | None = None
             ) -> tuple[pd.DataFrame, list[dict]]:
    """Считает все пригодные формулы на «сырых» тегах.

    Parameters
    ----------
    avt, ht : телеметрия без префиксов, уже очищенная от заглушек.
    lims_context : ряды для формул, ссылающихся на ЛИМС; ключи — ``LIMS_D15``,
        ``LIMS_T95``. Ряды должны быть уже выровнены по индексу телеметрии и
        сдвинуты назад по времени вызывающей стороной.
    check_plausibility : отбрасывать формулы, выходящие за физический диапазон.
        Выключается только для диагностики — посмотреть, что именно считает
        формула, прежде чем решать, годится ли она.

    Returns
    -------
    ``(features, skipped)`` — таблица ``vak_*`` и список необсчитанных формул.
    """
    usable, skipped = compile_formulas()
    lims_context = lims_context or {}
    index = ht.index if len(ht) else avt.index
    out = pd.DataFrame(index=index)

    for item in usable:
        source = avt if item["unit"] == "avt" else ht
        namespace = {tag: source[tag] for tag in item["tags"] if tag in source.columns}
        missing = [t for t in item["tags"] if t not in namespace]

        if item["uses_lims"]:
            for name in LIMS_TOKENS.values():
                key = name[0]
                if key in item["expr"]:
                    if key in lims_context:
                        namespace[key] = lims_context[key]
                    else:
                        missing.append(key)

        if missing:
            skipped.append({"target": item["target"],
                            "reason": f"нет входов: {', '.join(missing)}"})
            continue

        try:
            value = eval(item["code"], _SAFE_GLOBALS, namespace)  # noqa: S307
        except Exception as err:                                  # noqa: BLE001
            skipped.append({"target": item["target"], "reason": f"ошибка счёта: {err}"})
            continue

        if np.isscalar(value):
            skipped.append({"target": item["target"], "reason": "формула не зависит от тегов"})
            continue

        series = pd.Series(value, index=index).astype("float64")
        # деление на ноль встречается в формулах АВТ (например F31/F57, где F57
        # обнуляется на 4 % отсчётов): бесконечность — это отсутствие значения
        series = series.replace([np.inf, -np.inf], np.nan)

        # Правдоподобие формулы — тоже решение об отборе признака, и считать его
        # надо по обучающему периоду, а не по всей истории вместе с тестом.
        scope = series.loc[train_bounds[0]:train_bounds[1]] if train_bounds else series
        share, bounds = _plausible_share(item["target"], scope if len(scope) else series)
        if check_plausibility and share is not None:
            if share < MIN_PLAUSIBLE_SHARE:
                skipped.append({
                    "target": item["target"],
                    "reason": (f"значения вне физического диапазона {bounds}: "
                               f"внутри лишь {share:.0%}"),
                })
                continue
            # редкие выбросы за диапазон — это деление на почти ноль, а не значение
            series = series.where((series >= bounds[0]) & (series <= bounds[1]))

        out[item["column"]] = series.astype("float32")

    return out, skipped


def _plausible_share(target: str, series: pd.Series) -> tuple[float | None, tuple | None]:
    """Доля значений внутри физического диапазона показателя."""
    kind = target.rsplit(":", 1)[-1]
    bounds = PLAUSIBLE_RANGES.get(kind)
    if bounds is None:
        return None, None
    finite = series.dropna()
    if finite.empty:
        return 0.0, bounds
    inside = ((finite >= bounds[0]) & (finite <= bounds[1])).mean()
    return float(inside), bounds
