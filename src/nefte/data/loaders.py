"""Загрузка всех источников хакатона в единый вид.

Ключевые особенности исходных данных (подробно — docs/DATA_NOTES.md):
* телеметрия: ровная сетка 10 мин, служебные колонки ``Unnamed: …``;
* ЛИМС: «широкий» лист, каждая серия занимает ДВЕ колонки (метка времени, значение),
  строка 0 — точка отбора, 1 — показатель, 2 — единица (сдвинута для первого блока!),
  3 — «Количество значений»; данные начинаются со строки 4;
* ПАК: два независимых ряда с разным покрытием в колонках 0-1 и 3-4.

Все функции кэшируют результат в parquet: повторная загрузка ~секунды вместо минут.
"""
from __future__ import annotations

import re

import pandas as pd

from nefte.config import cache_dir, data_file, source_file

# --------------------------------------------------------------------------- #
# телеметрия
# --------------------------------------------------------------------------- #

UNITS = {"avt": "avt_csv", "ht": "ht_csv"}


def load_telemetry(unit: str = "avt", columns: list[str] | None = None,
                   use_cache: bool = True) -> pd.DataFrame:
    """Телеметрия установки.

    Parameters
    ----------
    unit : ``"avt"`` (ЭЛОУ-АВТ-6, 71 тег) или ``"ht"`` (24-2000, 26 тегов).
    columns : подмножество тегов.

    Returns
    -------
    DataFrame с DatetimeIndex ``date``, шаг 10 минут, без пропусков времени.
    """
    if unit not in UNITS:
        raise ValueError(f"unit должен быть одним из {list(UNITS)}, получено {unit!r}")

    cache = cache_dir() / f"telemetry_{unit}.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        csv = data_file(UNITS[unit])
        if not csv.exists():
            raise FileNotFoundError(
                f"{csv} не найден. Распакуйте данные: python scripts/prepare_data.py"
            )
        df = pd.read_csv(csv)
        df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index().astype("float32")
        df.to_parquet(cache)

    if columns is not None:
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(f"нет тегов {sorted(missing)} в телеметрии {unit}")
        df = df[columns]
    return df


# --------------------------------------------------------------------------- #
# ЛИМС
# --------------------------------------------------------------------------- #

_POINT_RE = re.compile(
    r"Установка\s*'(?P<unit>[^']+)'\.*\s*Точка отбора\s*'(?P<point>[^']+)'"
)

# Единицы в строке 2 листа для первого блока сдвинуты (у 50%.T написано кг/м3).
# Поэтому единицу берём по имени показателя, а не по ячейке.
PARAM_UNITS = {
    "IBP.T": "°C", "50%.T": "°C", "90%.T": "°C", "95%.T": "°C", "EBP.T": "°C",
    "CFPP": "°C", "CloudPoint": "°C", "CloudPoint_1": "°C", "PourPoint": "°C",
    "FlashPoint": "°C", "FilterabilityLimit.T": "°C",
    "D15": "кг/м3", "Mg.Sulfur": "мг/кг", "Mass.Sulfur": "% масс.",
    "I250": "% об.", "I350": "% об.", "CetaneNumber": "ед.цет.ч.",
}


def load_lims(use_cache: bool = True) -> pd.DataFrame:
    """Лабораторные анализы в длинном формате.

    Returns
    -------
    DataFrame: ``ts, unit, point, param, series, value, unit_of_measure``.
    ``series`` — устойчивый ключ вида ``"Гидроочистка|2|Mg.Sulfur"``.
    """
    cache = cache_dir() / "lims_long.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    raw = pd.read_excel(source_file("lims_xlsx"), header=None)
    groups, params = raw.iloc[0], raw.iloc[1]

    frames, current = [], ""
    for j in range(0, raw.shape[1], 2):
        head = groups[j]
        if isinstance(head, str) and head.strip():
            current = head
        m = _POINT_RE.search(current)
        if not m:
            continue
        unit_name = m.group("unit").strip(". ")
        point = m.group("point").strip()
        param = str(params[j]).strip()

        block = raw.iloc[4:, [j, j + 1]].copy()
        block.columns = ["ts", "value"]
        block["ts"] = pd.to_datetime(block["ts"], errors="coerce")
        block["value"] = pd.to_numeric(block["value"], errors="coerce")
        block = block.dropna()
        if block.empty:
            continue
        block["unit"] = unit_name
        block["point"] = point
        block["param"] = param
        block["series"] = f"{unit_name}|{point}|{param}"
        block["unit_of_measure"] = PARAM_UNITS.get(param, "")
        frames.append(block)

    df = pd.concat(frames, ignore_index=True).sort_values(["series", "ts"])
    df = df[["ts", "unit", "point", "param", "series", "value", "unit_of_measure"]]
    df.to_parquet(cache, index=False)
    return df


def lims_series(name: str, lims: pd.DataFrame | None = None) -> pd.Series:
    """Один лабораторный ряд по ключу ``"Установка|точка|показатель"``."""
    lims = load_lims() if lims is None else lims
    sub = lims[lims["series"] == name]
    if sub.empty:
        raise KeyError(f"ряд {name!r} не найден; см. lims['series'].unique()")
    return sub.set_index("ts")["value"].sort_index()


# --------------------------------------------------------------------------- #
# ПАК
# --------------------------------------------------------------------------- #

def load_pak(use_cache: bool = True) -> dict[str, pd.Series]:
    """Поточные анализаторы.

    Returns
    -------
    ``{"sulfur_ppm": Series, "d15_kgm3": Series}``. Покрытие рядов РАЗНОЕ:
    сера — весь период с шагом 10 мин, D15 — только с 2025-03-05.
    """
    cache = cache_dir() / "pak.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        raw = pd.read_excel(source_file("pak_xlsx"), header=None, skiprows=2,
                            usecols=[0, 1, 3, 4],
                            names=["ts_s", "sulfur_ppm", "ts_d", "d15_kgm3"])
        s = raw[["ts_s", "sulfur_ppm"]].dropna()
        d = raw[["ts_d", "d15_kgm3"]].dropna()
        s["ts_s"] = pd.to_datetime(s["ts_s"])
        d["ts_d"] = pd.to_datetime(d["ts_d"])
        df = pd.concat([
            s.rename(columns={"ts_s": "ts", "sulfur_ppm": "value"}).assign(tag="sulfur_ppm"),
            d.rename(columns={"ts_d": "ts", "d15_kgm3": "value"}).assign(tag="d15_kgm3"),
        ], ignore_index=True)
        df.to_parquet(cache, index=False)

    return {
        tag: sub.set_index("ts")["value"].sort_index()
        for tag, sub in df.groupby("tag")
    }


# --------------------------------------------------------------------------- #
# справочники
# --------------------------------------------------------------------------- #

def load_tag_dictionary(use_cache: bool = True) -> pd.DataFrame:
    """Расшифровка тегов КИП: ``unit, code, description``.

    ВНИМАНИЕ: для установки 24-2000 буква в коротком имени НЕ соответствует
    физическому смыслу (``T11`` — температура, хотя описан как расход и т.п.).
    Смысл берём только из описания, диапазон — проверяем по данным.
    """
    cache = cache_dir() / "tags.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    raw = pd.read_excel(source_file("tags_xlsx"), sheet_name="КИП", header=0)
    cols = list(raw.columns)
    rows = []
    for unit, (desc_col, code_col) in {"avt": (cols[0], cols[1]),
                                       "ht": (cols[2], cols[3])}.items():
        sub = raw[[desc_col, code_col]].dropna()
        sub.columns = ["description", "code"]
        sub["unit"] = unit
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)[["unit", "code", "description"]]
    df.to_parquet(cache, index=False)
    return df


def load_vak_formulas() -> pd.DataFrame:
    """Формулы виртуальных анализаторов (лист ВАК).

    Returns
    -------
    ``block, target, formula, uses_lims`` — ``uses_lims=True`` означает, что
    формула опирается на лабораторное значение, то есть наследует его возраст.
    """
    raw = pd.read_excel(source_file("tags_xlsx"), sheet_name="ВАК", header=None)
    rows = []
    for j in range(0, raw.shape[1], 2):
        block = raw.iloc[0, j]
        for i in range(1, raw.shape[0]):
            target, formula = raw.iloc[i, j], raw.iloc[i, j + 1]
            if isinstance(target, str) and target.strip():
                rows.append({
                    "block": block,
                    "target": target.strip(),
                    "formula": str(formula).strip(),
                    "uses_lims": "LIMS:" in str(formula),
                })
    return pd.DataFrame(rows)


def parse_vak_formula(formula: str) -> str:
    """Приводит запись формулы к python-совместимому виду.

    В листе ВАК смешаны стили: ``0,52755xT66`` и ``0.27467*T42``.
    """
    out = formula.replace(" ", "")
    out = re.sub(r"(?<=\d),(?=\d)", ".", out)               # десятичная запятая
    out = re.sub(r"(?<=[\d\)])x(?=[A-Za-z(])", "*", out)    # 'x' как умножение
    return out
