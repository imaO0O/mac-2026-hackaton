"""Агент качества.

Отвечает на три вопроса: какое качество сейчас, каким оно станет через горизонт H,
и какова вероятность выйти за спецификацию. Здесь лежит БАЗОВАЯ версия (персистенция
+ нормальная неопределённость), чтобы сквозной цикл работал с первого дня.
Участник 1 заменяет ``predict`` на обученную модель, не трогая контракт.
"""
from __future__ import annotations

import math

import numpy as np

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
# Неопределённость нашего знания о ТЕКУЩЕМ Т95, °C. Это не точность формулы ВАК, а
# разброс СУТОЧНОГО изменения самого показателя: уровень мы берём из последнего
# лабораторного анализа, а анализы идут раз в сутки (медиана шага 24 ч), и за это
# время Т95 успевает уехать. Измерено по обучающему периоду: ст. откл. изменения
# между соседними анализами 6.64 °C, MAE 4.86.
#
# Число важное: запас до предела 360 °C обычно около 12 °C, то есть меньше двух
# сигм. Показывать Т95 как точно известную величину нельзя.
T95_SIGMA_C = 6.64

BASELINE_SIGMA_MGKG = 1.7

# Множители уверенности по источнику значения. ДОПУЩЕНИЕ: прямого сравнения в
# данных нет, числа выбраны консервативно и по смыслу. Лаборатория — контрольный
# факт (ТЗ), поточный прибор шумнее её (MAE 1.7 на исторических парах), а
# виртуальный анализатор работает вообще без измерения продукта.
SOURCE_CONFIDENCE = {Source.LIMS: 1.0, Source.PAK: 0.85, Source.VAK: 0.5,
                     Source.NONE: 0.0}
VAK_CONFIDENCE_FACTOR = SOURCE_CONFIDENCE[Source.VAK]


def confidence_parts(sigma: float, source: Source, age_hours: float | None,
                     stale_after_hours: float, usable: bool) -> dict[str, float]:
    """Из чего складывается уверенность. Возвращает множители, а не одно число.

    Раньше уверенность считалась только по σ прогноза — и оказывалась 0.95 почти
    всегда, потому что σ обученной модели стабильна. В карточке оператора это было
    украшением: цифра, которая не меняется, решению не помогает.

    Теперь учитывается ещё и то, ОТКУДА взято значение и насколько оно свежее.
    Для модели это принципиально: σ ничего не знает ни про устаревший анализ, ни
    про то, что прибор молчит, — а решение зависит от этого сильнее, чем от
    ширины интервала.
    """
    parts = {
        # разброс прогноза: пока он не хуже исторического разброса «лаборатория
        # против поточного прибора», уверенность не режем
        "σ прогноза": 1.0 / (1.0 + max(sigma - BASELINE_SIGMA_MGKG, 0.0)
                             / BASELINE_SIGMA_MGKG),
        "источник": SOURCE_CONFIDENCE.get(source, 0.0),
    }
    if age_hours is not None and stale_after_hours > 0:
        overdue = max(age_hours - stale_after_hours, 0.0)
        parts["свежесть"] = 1.0 / (1.0 + overdue / stale_after_hours)
    if not usable:
        parts["достоверность данных"] = 0.5
    return parts


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

    def __init__(self, model=None, cfg: dict | None = None, horizon_hours: float = 2.0,
                 t95_fn=None):
        self.model = model            # обученная модель (участник 1); None → персистенция
        self.cfg = cfg or load_config()
        self.horizon_hours = getattr(model, "horizon_hours", horizon_hours)
        self.limit = self.cfg["spec"]["product_sulfur_mgkg"]["max"]
        # Т95 — ВТОРОЙ обязательный показатель качества по ответу организаторов, и
        # считать его должен агент качества, а не оптимизатор. Раньше оценка жила в
        # `optimizer.default_t95_estimator`: оптимизатор сам вычислял показатель
        # качества, то есть делал чужую работу, и в `QualityAssessment` этого
        # показателя не было вовсе — ни в логах прогонов, ни на дашборде.
        #
        # Функция, а не число: оптимизатору нужен Т95 для КАЖДОГО варианта уставок.
        from nefte.agents.optimizer import default_t95_estimator

        self.t95_fn = t95_fn if t95_fn is not None else default_t95_estimator()
        self.t95_limit = float(self.cfg["spec"]["t95_c"]["max"])
        # порог тревоги подобран на валидации вместе с моделью; без модели —
        # консервативное значение по умолчанию
        self.alarm_threshold = float(getattr(model, "alarm_threshold", 0.2))

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
        # Ни лаборатории, ни поточного анализатора. Если обученной модели нет,
        # прогнозировать нечем. Если есть — это ровно тот случай, для которого ТЗ
        # предусматривает третий приоритет: виртуальный анализатор. Называть его
        # надо своим именем и с пониженной уверенностью, а не выдавать за измерение.
        # ...и только если модель ДЕЙСТВИТЕЛЬНО что-то выдала. Раньше примечание
        # «значение получено виртуальным анализатором» появлялось и тогда, когда
        # модель вернула NaN (например, момент раньше начала истории): объяснение
        # утверждало то, чего не было.
        from_vak = (current.source is Source.NONE and self.model is not None
                    and mean == mean)
        if current.source is Source.NONE:
            notes.append(
                "Ни ЛИМС, ни ПАК недостоверны: значение получено виртуальным "
                "анализатором (ВАК, третий приоритет по ТЗ), уверенность снижена."
                if from_vak else
                "Нет достоверного источника по сере — прогноз недоступен.")
        if current.is_frozen:
            notes.append("Поточный анализатор признан замороженным.")
        if current.is_stale:
            notes.append(f"Лабораторное значение устарело: {current.age_hours:.1f} ч.")

        source = Source.VAK if from_vak else current.source
        if mean != mean:      # NaN
            return QualityAssessment(ts=state.ts, confidence=0.0, notes=notes,
                                     source=Source.NONE,
                                     horizon_hours=self.horizon_hours)

        # если у модели есть отдельный классификатор превышения — доверяем ему:
        # редкие события он оценивает лучше, чем нормальное приближение по σ
        risk = None
        if self.model is not None and hasattr(self.model, "risk_for_state"):
            risk = self.model.risk_for_state(state)
        if risk is None:
            risk = spec_risk_normal(mean, sigma, self.limit)
        stale_after = float(self.cfg["quality"]["staleness_hours"]["lims"])
        parts = confidence_parts(sigma, source, current.age_hours, stale_after,
                                 state.data_quality.usable)
        confidence = max(0.05, min(0.95, float(np.prod(list(parts.values())))))
        weakest = min(parts, key=parts.get)
        if parts[weakest] < 0.9:
            notes.append(f"Уверенность {confidence:.2f}; сильнее всего её снижает "
                         f"«{weakest}» (множитель {parts[weakest]:.2f}).")

        predictions = {"product_sulfur_mgkg": mean}
        intervals = {"product_sulfur_mgkg": (mean - 1.96 * sigma, mean + 1.96 * sigma)}
        risks = {"product_sulfur_mgkg": risk}

        # Т95 текущего режима. Уровень — последний лабораторный анализ, поэтому и
        # неопределённость берётся ЕГО: между анализами сутки, и Т95 успевает
        # уехать. Разброс суточного изменения измерен по обучающему периоду и
        # равен 6.6 °C — это и есть σ нашего знания о текущем Т95, а вовсе не
        # точность формулы.
        t95 = self.t95_fn(state, {}) if self.t95_fn else None
        if t95 is not None and t95 == t95:
            predictions["product_t95_c"] = float(t95)
            intervals["product_t95_c"] = (t95 - 1.96 * T95_SIGMA_C,
                                          t95 + 1.96 * T95_SIGMA_C)
            risks["product_t95_c"] = spec_risk_normal(t95, T95_SIGMA_C, self.t95_limit)
            if risks["product_t95_c"] > 0.2:
                notes.append(
                    f"Т95 {t95:.1f} °C при пределе {self.t95_limit:.0f}: "
                    f"вероятность выхода {risks['product_t95_c']:.0%}. Уровень взят "
                    "из последнего анализа, между анализами он уезжает на 6.6 °C.")

        return QualityAssessment(
            ts=state.ts,
            source=source,
            predictions=predictions,
            intervals=intervals,
            spec_risk=risks,
            horizon_hours=self.horizon_hours,
            confidence=confidence,
            notes=notes,
        )


def empty_data_quality() -> DataQuality:
    return DataQuality(missing_share=0.0, usable=True)
