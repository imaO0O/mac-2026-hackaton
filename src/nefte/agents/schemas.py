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
    # Источник, по которому принято решение. ЛИМС → ПАК → ВАК (приоритет из ТЗ).
    # Без этого поля оркестратор показывал оператору источник, которого в решении
    # не было: в срезе есть и сера сырья, и сера продукта.
    source: Source = Source.NONE
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
    # P(нарушение) ПО СУРРОГАТУ, а не по калиброванному классификатору агента
    # качества. Величины считаются разной машинерией и сравнимы между вариантами,
    # но НЕ с риском в шапке рекомендации: там классификатор для текущего режима.
    spec_risk: dict[str, float] = Field(
        default_factory=dict, description="P(нарушение) по суррогату, для сравнения вариантов")
    throughput: float | None = None
    energy_proxy: float | None = None
    severity_index: float | None = None
    feasible: bool = True
    # True — прогноз укладывается в предел С ЗАПАСОМ на неопределённость;
    # False — вариант лишь улучшает качество относительно бездействия, гарантии нет
    guaranteed: bool = True
    violations: list[str] = []
    score: float | None = None
    pareto_rank: int | None = None


class BlendComponent(BaseModel):
    """Компонент смешения с его свойствами и располагаемым расходом."""
    name: str
    sulfur_mgkg: float
    density_15c: float | None = None
    t95_c: float | None = None
    cfpp_c: float | None = None
    # Цетановое число — третий обязательный показатель по ответу организаторов.
    # В ЛИМС меряется раз в месяц, поэтому у большинства компонентов его нет, и
    # None здесь — нормальное состояние, а не ошибка.
    cetane_number: float | None = None
    available_tph: float = Field(default=0.0, description="располагаемый расход, т/ч")
    is_assumption: bool = Field(default=False,
                                description="свойства взяты из допущения, а не из ЛИМС")


class BlendRecipe(BaseModel):
    """Ответ агента смешения: рецептура и свойства полученной смеси."""
    fractions: dict[str, float] = Field(description="компонент → массовая доля, сумма = 1")
    properties: dict[str, float] = {}
    throughput_tph: float = 0.0
    additive_ppm: float = 0.0
    # Цетаноповышающая присадка: доза в % массы (организаторы — не выше 3) и её
    # стоимость в долях цены тонны дизеля (организаторы — присадка дороже в 100
    # раз). Держим отдельно от депрессорной: у них разное назначение и разная цена.
    cetane_improver_pct: float = 0.0
    improver_cost_share: float = 0.0
    # Чистая ценность рецептуры: выпуск за вычетом стоимости присадки, в тоннах
    # дизеля. Ровно на неё оптимизируем, а не на голый выпуск — иначе присадка
    # выглядит бесплатной и назначается «на всякий случай».
    net_value_tph: float = 0.0
    feasible: bool = True
    # Обязательные показатели (сера, Т95, ЦЧ), которые проверить не удалось.
    # Пустой список — рецептура подтверждена по всем трём; непустой — годной её
    # называть нельзя, и оператор видит, чего именно не хватило.
    uncertified: list[str] = []
    violations: list[str] = []
    notes: list[str] = []
    # сера, на которой считалась рецептура: в цикле это ПРОГНОЗ агента качества,
    # а не последний анализ — иначе рецептура относится к уже прошедшему режиму
    basis_sulfur_mgkg: float | None = None
    basis: Literal["lims", "forecast"] = "lims"

    def fractions_sum(self) -> float:
        return float(sum(self.fractions.values()))


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
    blend: BlendRecipe | None = Field(
        default=None, description="рецептура смешения при рекомендуемом режиме")
    abstained: bool = False
    abstain_reason: str = ""

    def outcome(self) -> Literal["отказ", "меняем уставки", "держим режим"]:
        """Что система решила — одним словом.

        Считалось в пяти местах независимо: в прогоне по тесту, в сравнении
        архитектур, в двух проверках устойчивости и в имитации. Пять копий одного
        правила расходятся всегда, вопрос только когда, а сравнивать прогоны с
        разошедшимся определением исхода бессмысленно.
        """
        if self.abstained:
            return "отказ"
        moves = (self.action.deltas if self.action else {})
        return ("меняем уставки" if any(abs(d) > 1e-6 for d in moves.values())
                else "держим режим")

    def to_operator_text(self) -> str:
        """Человекочитаемый вид для демо и логов."""
        head = f"[{self.ts:%Y-%m-%d %H:%M}] "
        if self.abstained:
            # рецептуру показываем и при отказе: вопрос «а смешением не вытянуть?»
            # оператор задаёт именно тогда, когда рекомендации по режиму нет
            text = head + f"РЕКОМЕНДАЦИИ НЕТ: {self.abstain_reason}"
        else:
            moves = ", ".join(
                f"{tag}: {self.action.deltas.get(tag, 0):+.2f} → {val:.2f}"
                for tag, val in (self.action.moves if self.action else {}).items()
            )
            text = (
                f"{head}{self.problem}\n"
                f"  Действие: {moves or 'без изменений'}\n"
                f"  Эффект: {self.expected_effect}\n"
                f"  Проверено: {'; '.join(self.checked_constraints)}\n"
                f"  Уверенность: {self.confidence:.2f}\n"
                f"  Почему: {self.explanation}"
            )
        if self.blend is not None:
            shares = ", ".join(f"{name} {share * 100:.1f} %"
                               for name, share in self.blend.fractions.items() if share > 0)
            status = "допустима" if self.blend.feasible else "НЕДОПУСТИМА"
            text += (f"\n  Смешение ({status}, сумма долей "
                     f"{self.blend.fractions_sum() * 100:.1f} %): {shares}")
            if self.blend.cetane_improver_pct > 0:
                # Присадка дороже топлива в сто раз: её доза и её цена — не деталь
                # рецептуры, а отдельное решение, и прятать его в примечаниях нельзя.
                text += (f"\n    присадка ЦЧ {self.blend.cetane_improver_pct:.3f} % "
                         f"массы — {self.blend.improver_cost_share * 100:.1f} % цены "
                         f"тонны; чистая ценность {self.blend.net_value_tph:.1f} т/ч "
                         f"при выпуске {self.blend.throughput_tph:.1f}")
            if self.blend.uncertified:
                text += "\n    не подтверждено: " + ", ".join(self.blend.uncertified)
            for violation in self.blend.violations:
                text += f"\n    нарушение: {violation}"
        return text
