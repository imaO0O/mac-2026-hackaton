"""Валидация и очистка данных.

Формально пропусков в телеметрии нет (0 % NaN), но брак закодирован значениями-
заглушками и «полками». Без этой очистки любые модели и агент качества поедут.
Каждая функция возвращает не только данные, но и отчёт — он идёт в объяснение
оператору («какие данные признаны недостоверными и почему»).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nefte.config import load_config

# Значения-заглушки системы сбора: 307 встречается в 64 тегах АВТ и 15 тегах
# 24-2000, 313 — реже. Это НЕ физическое значение.
SENTINELS = (307.0, 313.0)


@dataclass
class CleaningReport:
    """Что именно было забраковано — для объяснимости и для агента качества."""
    n_rows: int = 0
    sentinel_masked: dict[str, int] = field(default_factory=dict)
    frozen_masked: dict[str, int] = field(default_factory=dict)
    negative_masked: dict[str, int] = field(default_factory=dict)
    dropped_tags: list[str] = field(default_factory=list)

    def share_bad(self) -> pd.Series:
        """Доля забракованных отсчётов по тегам, %."""
        total: dict[str, int] = {}
        for d in (self.sentinel_masked, self.frozen_masked, self.negative_masked):
            for k, v in d.items():
                total[k] = total.get(k, 0) + v
        return (pd.Series(total, dtype="float64") / max(self.n_rows, 1) * 100).sort_values(
            ascending=False
        )

    def summary(self) -> str:
        bad = self.share_bad()
        top = ", ".join(f"{k}: {v:.1f}%" for k, v in bad.head(5).items())
        return (
            f"строк: {self.n_rows}; удалено тегов: {len(self.dropped_tags)}; "
            f"наиболее зашумлённые: {top or 'нет'}"
        )


def mask_sentinels(df: pd.DataFrame, values: tuple[float, ...] = SENTINELS,
                   report: CleaningReport | None = None) -> pd.DataFrame:
    """Заменяет значения-заглушки на NaN."""
    out = df.copy()
    mask = pd.DataFrame(False, index=out.index, columns=out.columns)
    for v in values:
        mask |= out.eq(v)
    if report is not None:
        counts = mask.sum()
        report.sentinel_masked = counts[counts > 0].astype(int).to_dict()
    return out.mask(mask)


def frozen_mask(series: pd.Series, min_samples: int = 18) -> pd.Series:
    """True там, где значение УЖЕ не менялось ``min_samples`` отсчётов подряд.

    Для 10-минутной сетки 18 отсчётов = 3 часа. Реальный процесс так не стоит:
    это признак отказа датчика/анализатора (см. замороженный ПАК 2026-04-15…27).

    .. warning::
       Слово «уже» здесь несёт всю нагрузку. Раньше считался размер ВСЕГО прогона
       (``transform("size")``), и маска становилась истинной с ПЕРВОГО отсчёта
       полки — то есть в момент, когда видно одно-единственное одинаковое
       значение. Это заглядывание вперёд: узнать, что сигнал простоит ещё три
       часа, в тот момент нельзя.

       Дефект был двойным. Как ПРИЗНАК (``pak_frozen`` в матрице) он давал модели
       знание о будущем отказе прибора. Как СОСТОЯНИЕ в срезе оператора он делал
       систему прозорливой в бэктесте: она объявляла анализатор замороженным
       раньше, чем это стало бы известно на самом деле.

       Теперь считается нарастающий номер отсчёта внутри прогона, и маска
       включается ровно тогда, когда порог набран по прошлому.
    """
    grp = (series != series.shift()).cumsum()
    position = series.groupby(grp).cumcount() + 1
    return (position >= min_samples) & series.notna()


def frozen_intervals(series: pd.Series, min_samples: int = 18) -> pd.DataFrame:
    """Список интервалов «замороженного» сигнала: ``start, end, value, n, hours``."""
    grp = (series != series.shift()).cumsum()
    agg = series.groupby(grp).agg(n="size", value="first",
                                  start=lambda s: s.index[0], end=lambda s: s.index[-1])
    out = agg[agg["n"] >= min_samples].reset_index(drop=True)
    if out.empty:
        return out.assign(hours=pd.Series(dtype="float64"))
    step_h = pd.Series(series.index).diff().median().total_seconds() / 3600
    out["hours"] = (out["n"] * step_h).round(1)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def clean_telemetry(df: pd.DataFrame, unit: str = "avt",
                    drop_dead: bool = True,
                    mask_frozen: bool = True,
                    nonnegative: list[str] | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Очистка телеметрии с текстовым отчётом.

    .. warning::
       Рабочий путь очистки в системе один — ``data.validity.SignalValidity``:
       он же строит срез оператора, он же матрицу признаков. Эта функция осталась
       для разбора данных и отчётов, где нужен ``CleaningReport``. Не подставляйте
       её в конвейер: две реализации одного правила разъезжаются, и в прошлый раз
       разъехались на 0.9 % значений АВТ.

    Parameters
    ----------
    unit : ``"avt"`` | ``"ht"``.
    drop_dead : выбросить теги из ``telemetry.dead_tags`` (например ``D10``).
    mask_frozen : занулить «полки» длиннее ``telemetry.frozen_min_samples``.
    nonnegative : теги, где отрицательное значение физически невозможно.
        Задаётся явно, а не по букве F. Буква тут ни при чём — она совпадает с
        величиной и на АВТ, и у проверенных тегов 24-2000 (ошибались описания
        ``T11``/``F19``, а не буквы), — но из «это расход» не следует «это
        неотрицательно»: ``F5`` расход и по букве, и по описанию, а значения в
        основном отрицательные.
    """
    cfg = load_config()["telemetry"]
    rep = CleaningReport(n_rows=len(df))
    out = df.copy()

    if drop_dead:
        dead = [c for c in cfg.get("dead_tags", []) if c in out.columns]
        if unit == "avt" and dead:
            out = out.drop(columns=dead)
            rep.dropped_tags = dead

    out = mask_sentinels(out, tuple(cfg.get("sentinel_values", SENTINELS)), rep)

    if mask_frozen:
        min_s = int(cfg.get("frozen_min_samples", 18))
        counts = {}
        for col in out.columns:
            m = frozen_mask(out[col], min_s)
            if m.any():
                out.loc[m, col] = np.nan
                counts[col] = int(m.sum())
        rep.frozen_masked = counts

    if nonnegative:
        counts = {}
        for col in nonnegative:
            if col in out.columns:
                m = out[col] < 0
                if m.any():
                    out.loc[m, col] = np.nan
                    counts[col] = int(m.sum())
        rep.negative_masked = counts

    return out, rep


def clean_lims_sulfur(series: pd.Series, max_plausible: float | None = None) -> pd.Series:
    """Убирает нефизичные лабораторные значения серы (2120, 120, 107 мг/кг).

    Такие значения не могут относиться к товарному ДТ (спец. ≤ 10 мг/кг):
    это либо иная точка отбора, либо ошибка ввода. Отбрасываем, а не чиним.
    """
    if max_plausible is None:
        max_plausible = load_config()["quality"]["lims_sulfur_outlier_above"]
    return series[(series > 0) & (series <= max_plausible)]


def clean_lims_distillation(series: pd.Series) -> pd.Series:
    """Нули в температурах разгонки — это пропуск, а не 0 °C."""
    return series[series > 0]
