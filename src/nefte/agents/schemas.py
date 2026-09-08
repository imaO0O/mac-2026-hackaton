"""Контракты между агентами. ЭТОТ ФАЙЛ — ГРАНИЦА ОТВЕТСТВЕННОСТИ УЧАСТНИКОВ.

Меняется только по общему согласию: три человека пишут агентов параллельно и
собираются в оркестраторе именно через эти структуры. Пока контракт стабилен,
любой агент можно заменить заглушкой и демо всё равно соберётся.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Source(str, Enum):
    """Источник значения качества. Приоритет ЛИМС → ПАК → ВАК задан ТЗ."""
    LIMS = "lims"
    PAK = "pak"
    VAK = "vak"
    NONE = "none"


class Measurement(BaseModel):
    """Значение показателя вместе с его возрастом и происхождением."""
    value: float | None
    unit: str
    source: Source
    age_hours: float | None = None
    is_stale: bool = False
    is_frozen: bool = False
    comment: str = ""


class DataQuality(BaseModel):
    """Оценка пригодности входных данных для принятия решения."""
    missing_share: float = Field(ge=0.0, le=1.0)
    frozen_tags: list[str] = []
    sentinel_tags: list[str] = []
    stale_sources: list[str] = []
    usable: bool = True
    notes: list[str] = []


class ProcessState(BaseModel):
    """Срез состояния процесса на момент ``ts`` — вход всей системы."""
    ts: datetime
    telemetry_avt: dict[str, float | None] = {}
    telemetry_ht: dict[str, float | None] = {}
    quality: dict[str, Measurement] = {}
    data_quality: DataQuality


class QualityAssessment(BaseModel):
    """Ответ агента качества."""
    ts: datetime
    predictions: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    spec_risk: dict[str, float] = Field(default_factory=dict,
                                        description="P(нарушение) по каждому показателю, 0..1")
    horizon_hours: float = 2.0
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    drivers: dict[str, float] = Field(default_factory=dict,
                                      description="вклад признаков (SHAP/коэффициенты)")
    notes: list[str] = []


class ReliabilityAssessment(BaseModel):
    """Ответ агента надёжности."""
    ts: datetime
    severity_index: float = Field(ge=0.0, le=1.0, description="0 — мягкий режим, 1 — предельный")
    risk_class: Literal["low", "medium", "high"] = "low"
    factors: dict[str, float] = {}
    admissible: bool = True
    constraints: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="доп. сужение диапазонов для оптимизатора")
    notes: list[str] = []


class Candidate(BaseModel):
    """Один вариант управляющего воздействия."""
    id: str
    moves: dict[str, float] = Field(description="тег → новое абсолютное значение")
    deltas: dict[str, float] = Field(default_factory=dict, description="тег → изменение")
    predicted_quality: dict[str, float] = {}
    spec_risk: dict[str, float] = {}
    throughput: float | None = None
    energy_proxy: float | None = None
    severity_index: float | None = None
    feasible: bool = True
    violations: list[str] = []
    score: float | None = None
    pareto_rank: int | None = None


class Recommendation(BaseModel):
    """Итог цикла. Структура повторяет п.5 ТЗ «Пример рекомендаций оператору»."""
    ts: datetime
    state_summary: dict[str, float | str | None] = {}
    freshness: dict[str, float | None] = Field(
        default_factory=dict, description="возраст ЛИМС/ПАК в часах")
    problem: str
    action: Candidate | None = None
    expected_effect: dict[str, float | str] = {}
    checked_constraints: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    explanation: str = ""
    alternatives: list[Candidate] = []
    abstained: bool = False
    abstain_reason: str = ""

    def to_operator_text(self) -> str:
        """Человекочитаемый вид для демо и логов."""
        head = f"[{self.ts:%Y-%m-%d %H:%M}] "
        if self.abstained:
            return head + f"РЕКОМЕНДАЦИИ НЕТ: {self.abstain_reason}"
        moves = ", ".join(
            f"{tag}: {self.action.deltas.get(tag, 0):+.2f} → {val:.2f}"
            for tag, val in (self.action.moves if self.action else {}).items()
        )
        return (
            f"{head}{self.problem}\n"
            f"  Действие: {moves or 'без изменений'}\n"
            f"  Эффект: {self.expected_effect}\n"
            f"  Проверено: {'; '.join(self.checked_constraints)}\n"
            f"  Уверенность: {self.confidence:.2f}\n"
            f"  Почему: {self.explanation}"
        )
