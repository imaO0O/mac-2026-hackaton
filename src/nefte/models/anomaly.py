"""Многомерный детектор аномалий режима.

Зачем он нужен помимо потеговых проверок. `data/validity.py` ловит брак в
отдельном сигнале: заглушку, полку, отрицательный расход. Но опасное состояние
чаще выглядит иначе — каждый тег по отдельности в норме, а их сочетание
невозможно: например, высокая температура реактора при низком расходе водорода
или подъём перепада давления при упавшей нагрузке. Одномерные проверки такое
пропускают по построению.

Метод — расстояние Махаланобиса по нескольким описателям режима. Выбран
сознательно, а не потому что «модно»:

* считается на CPU за секунды и не требует обучения с подбором гиперпараметров;
* детерминирован — одинаковый вход даёт одинаковый выход, это требование
  воспроизводимости из ТЗ;
* **объясним**: вклад каждой переменной в расстояние раскладывается покомпонентно,
  и оператору можно сказать не «аномалия», а «нетипично низкая кратность
  газ/сырьё при этой температуре».

Изоляционный лес и автоэнкодер дали бы гибкость к нелинейностям, но объяснить
их вклад оператору сложнее, а выигрыш на шести переменных сомнителен.

Нормировка и ковариация считаются ТОЛЬКО по обучающему периоду: иначе аномалия
2026 года будет измеряться относительно 2026 года, то есть относительно будущего.
Центр и масштаб берутся робастно (медиана и межквартильный размах), чтобы сами
выбросы не задавали норму.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Множитель перевода межквартильного размаха в оценку стандартного отклонения
# для нормального распределения.
IQR_TO_SIGMA = 1.349


@dataclass
class RegimeAnomalyDetector:
    """Расстояние Махаланобиса по описателям режима."""

    columns: list[str] = field(default_factory=list)
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    inv_cov: np.ndarray | None = None
    threshold: float = 0.0
    quantile: float = 0.99

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str] | None = None,
            train: tuple[str, str] | None = None,
            quantile: float = 0.99) -> "RegimeAnomalyDetector":
        """Обучение на истории. ``train`` ограничивает период обучающим."""
        columns = [c for c in (columns or frame.columns) if c in frame.columns]
        data = frame[columns]
        if train is not None:
            data = data.loc[train[0]:train[1]]
        data = data.dropna()
        if len(data) < 100 or not columns:
            return cls(columns=columns)

        values = data.to_numpy(dtype="float64")
        center = np.median(values, axis=0)
        q75, q25 = np.percentile(values, [75, 25], axis=0)
        scale = (q75 - q25) / IQR_TO_SIGMA
        scale[scale <= 0] = 1.0

        z = (values - center) / scale
        cov = np.cov(z, rowvar=False)
        inv_cov = np.linalg.pinv(np.atleast_2d(cov))

        obj = cls(columns=columns, center=center, scale=scale, inv_cov=inv_cov,
                  quantile=quantile)
        distances = obj.distance(data)
        obj.threshold = float(np.nanquantile(distances, quantile))
        return obj

    @property
    def fitted(self) -> bool:
        return self.inv_cov is not None and self.threshold > 0

    # ------------------------------------------------------------------ #
    def distance(self, frame: pd.DataFrame) -> np.ndarray:
        """Расстояние Махаланобиса для каждой строки."""
        if self.inv_cov is None:
            return np.full(len(frame), np.nan)
        z = (frame[self.columns].to_numpy(dtype="float64") - self.center) / self.scale
        # (z · Σ⁻¹) ⊙ z, суммируем по признакам — так быстрее, чем матрица целиком
        return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", z, self.inv_cov, z), 0, None))

    def score_row(self, values: dict[str, float | None]) -> float | None:
        """Нормированная аномальность одного среза: 1.0 — уровень порога."""
        if not self.fitted:
            return None
        row = [values.get(c) for c in self.columns]
        if any(v is None or v != v for v in row):
            return None
        frame = pd.DataFrame([row], columns=self.columns)
        return float(self.distance(frame)[0] / self.threshold)

    def contributions(self, values: dict[str, float | None]) -> dict[str, float]:
        """Вклад каждой переменной в расстояние — то, что показываем оператору."""
        if not self.fitted:
            return {}
        row = np.array([values.get(c, np.nan) for c in self.columns], dtype="float64")
        if np.isnan(row).any():
            return {}
        z = (row - self.center) / self.scale
        parts = z * (self.inv_cov @ z)
        total = float(np.sum(np.abs(parts))) or 1.0
        return {c: round(float(abs(p) / total), 3)
                for c, p in sorted(zip(self.columns, parts),
                                   key=lambda kv: -abs(kv[1]))}

    def flags(self, frame: pd.DataFrame) -> pd.Series:
        """Булева маска «режим нетипичен» по всей истории."""
        if not self.fitted:
            return pd.Series(False, index=frame.index)
        return pd.Series(self.distance(frame) > self.threshold, index=frame.index)
