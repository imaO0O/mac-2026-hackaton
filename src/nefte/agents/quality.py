"""Агент качества.

Отвечает на три вопроса: какое качество сейчас, каким оно станет через горизонт H,
и какова вероятность выйти за спецификацию. Здесь лежит БАЗОВАЯ версия (персистенция
+ нормальная неопределённость), чтобы сквозной цикл работал с первого дня.
Участник 1 заменяет ``predict`` на обученную модель, не трогая контракт.
"""
from __future__ import annotations

import math

from nefte.agents.schemas import (
    DataQuality,
    Measurement,
    ProcessState,
    QualityAssessment,
    Source,
)
from nefte.config import load_config

# Разброс между лабораторией и поточным анализатором на исторических парах:
# MAE 1.70 мг/кг, смещение -0.26. Используется как априорная σ базовой модели.
BASELINE_SIGMA_MGKG = 1.7


def fuse_sulfur(state: ProcessState, cfg: dict | None = None) -> Measurement:
    """Выбирает значение серы по приоритету ЛИМС → ПАК → ВАК с учётом свежести.

    Лабораторный результат — контрольный факт (ТЗ). Но если он устарел, а ПАК
    исправен, оперативное значение полезнее устаревшего лабораторного: возвращаем
    ПАК и явно помечаем источник, чтобы это попало в объяснение оператору.
    """
    cfg = cfg or load_config()
    stale = cfg["quality"]["staleness_hours"]

    lims = state.quality.get("lims_sulfur_mgkg")
    pak = state.quality.get("pak_sulfur_ppm")

    if lims and lims.value is not None and (lims.age_hours or 0) <= stale["lims"]:
        return lims.model_copy(update={"source": Source.LIMS})
    if pak and pak.value is not None and not pak.is_frozen:
        comment = "ЛИМС устарел, взят поточный анализатор" if lims else "нет ЛИМС"
        return pak.model_copy(update={"source": Source.PAK, "comment": comment})
    if lims and lims.value is not None:
        return lims.model_copy(update={"source": Source.LIMS, "is_stale": True,
                                       "comment": "ПАК недостоверен, взят устаревший ЛИМС"})
    return Measurement(value=None, unit="мг/кг", source=Source.NONE,
                       comment="нет достоверного источника качества")


def spec_risk_normal(pred: float, sigma: float, limit: float) -> float:
    """P(значение > limit) при нормальной ошибке прогноза."""
    if sigma <= 0:
        return float(pred > limit)
    z = (limit - pred) / sigma
    return float(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


class QualityAgent:
    """Базовая реализация. Заменяемая часть — ``predict``."""

    def __init__(self, model=None, cfg: dict | None = None, horizon_hours: float = 2.0):
        self.model = model            # обученная модель (участник 1); None → персистенция
        self.cfg = cfg or load_config()
        self.horizon_hours = horizon_hours
        self.limit = self.cfg["spec"]["product_sulfur_mgkg"]["max"]

    def predict(self, state: ProcessState, current: Measurement) -> tuple[float, float]:
        """Прогноз серы на горизонте и σ. Возвращает ``(mean, sigma)``."""
        if self.model is not None:
            return self.model.predict_with_sigma(state)
        if current.value is None:
            return float("nan"), float("inf")
        sigma = BASELINE_SIGMA_MGKG
        if current.source is Source.PAK:
            sigma *= 1.3          # поточный анализатор шумнее лаборатории
        if current.is_stale:
            sigma *= 1.5 + (current.age_hours or 0) / 24.0
        return float(current.value), float(sigma)

    def assess(self, state: ProcessState) -> QualityAssessment:
        current = fuse_sulfur(state, self.cfg)
        mean, sigma = self.predict(state, current)

        notes: list[str] = []
        if current.source is Source.NONE:
            notes.append("Нет достоверного источника по сере — прогноз недоступен.")
        if current.is_frozen:
            notes.append("Поточный анализатор признан замороженным.")
        if current.is_stale:
            notes.append(f"Лабораторное значение устарело: {current.age_hours:.1f} ч.")

        if mean != mean:      # NaN
            return QualityAssessment(ts=state.ts, confidence=0.0, notes=notes,
                                     horizon_hours=self.horizon_hours)

        risk = spec_risk_normal(mean, sigma, self.limit)
        confidence = max(0.05, min(0.95, 1.0 / (1.0 + sigma / BASELINE_SIGMA_MGKG - 1.0)))
        if not state.data_quality.usable:
            confidence *= 0.5

        return QualityAssessment(
            ts=state.ts,
            predictions={"product_sulfur_mgkg": mean},
            intervals={"product_sulfur_mgkg": (mean - 1.96 * sigma, mean + 1.96 * sigma)},
            spec_risk={"product_sulfur_mgkg": risk},
            horizon_hours=self.horizon_hours,
            confidence=confidence,
            notes=notes,
        )


def empty_data_quality() -> DataQuality:
    return DataQuality(missing_share=0.0, usable=True)
