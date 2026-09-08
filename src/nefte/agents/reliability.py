"""Агент надёжности.

Прямой разметки отказов, наработки катализатора и предельных уставок в пакете НЕТ.
Поэтому оцениваем тяжесть режима прокси-метриками и объявляем это допущением
(ТЗ, п.3: «при отсутствии прямой разметки допускаются прокси-метрики с явным
описанием допущений»).

Прокси, которые используем:
* WABT реакторного блока (T5, T6, T11) — чем выше, тем жёстче режим и быстрее
  дезактивируется катализатор;
* перепад давления на Р-202 (W10) — рост означает закоксовывание/забивку слоя;
* температура на выходе печи П-3 (T55) — тепловая напряжённость печи;
* скорость изменения режима — резкие броски сами по себе фактор риска.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nefte.agents.schemas import ProcessState, ReliabilityAssessment


@dataclass
class SeverityNorms:
    """Нормировка прокси-факторов по историческим квантилям.

    ``fit`` считает p05/p95 по обучающему периоду — это модельный диапазон,
    а не промышленный предел. Так требует ТЗ (правило границ).
    """
    bounds: dict[str, tuple[float, float]]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str],
            q: tuple[float, float] = (0.05, 0.95)) -> "SeverityNorms":
        bounds = {}
        for c in columns:
            if c in df.columns:
                lo, hi = df[c].quantile(q[0]), df[c].quantile(q[1])
                bounds[c] = (float(lo), float(hi))
        return cls(bounds=bounds)

    def normalize(self, name: str, value: float | None) -> float | None:
        if value is None or name not in self.bounds:
            return None
        lo, hi = self.bounds[name]
        if hi <= lo:
            return None
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.5))


class ReliabilityAgent:
    """Оценивает тяжесть режима и сужает диапазоны для оптимизатора."""

    WEIGHTS = {"wabt": 0.45, "dp_r202": 0.30, "furnace": 0.15, "ramp": 0.10}
    REACTOR_TEMPS = ["T5", "T6", "T11"]
    DP_TAG = "W10"
    FURNACE_TAG = "T55"

    def __init__(self, norms: SeverityNorms | None = None, ramp: dict[str, float] | None = None):
        self.norms = norms or SeverityNorms(bounds={})
        self.ramp = ramp or {}

    def _factors(self, state: ProcessState) -> dict[str, float]:
        ht, avt = state.telemetry_ht, state.telemetry_avt
        temps = [ht.get(t) for t in self.REACTOR_TEMPS if ht.get(t) is not None]
        factors: dict[str, float] = {}

        if temps:
            wabt = float(np.mean(temps))
            n = self.norms.normalize("wabt", wabt)
            if n is not None:
                factors["wabt"] = n

        for key, tag, store in (("dp_r202", self.DP_TAG, ht), ("furnace", self.FURNACE_TAG, avt)):
            n = self.norms.normalize(tag, store.get(tag))
            if n is not None:
                factors[key] = n

        if self.ramp:
            factors["ramp"] = float(np.clip(max(self.ramp.values()), 0.0, 1.0))
        return factors

    def assess(self, state: ProcessState) -> ReliabilityAssessment:
        factors = self._factors(state)
        if not factors:
            return ReliabilityAssessment(
                ts=state.ts, severity_index=0.5, risk_class="medium", admissible=True,
                notes=["Нет данных для оценки тяжести режима — принята средняя оценка."],
            )

        w = {k: self.WEIGHTS[k] for k in factors}
        total = sum(w.values())
        severity = float(sum(factors[k] * w[k] for k in factors) / total)
        severity = float(np.clip(severity, 0.0, 1.0))

        risk_class = "low" if severity < 0.5 else "medium" if severity < 0.8 else "high"
        notes = []
        constraints: dict[str, tuple[float, float]] = {}

        if risk_class == "high":
            notes.append("Тяжёлый режим: повышение температур запрещено до снижения severity.")
            for tag in self.REACTOR_TEMPS:
                v = state.telemetry_ht.get(tag)
                if v is not None:
                    constraints[tag] = (v - 3.0, v)      # только вниз
        elif risk_class == "medium":
            for tag in self.REACTOR_TEMPS:
                v = state.telemetry_ht.get(tag)
                if v is not None:
                    constraints[tag] = (v - 3.0, v + 1.0)

        return ReliabilityAssessment(
            ts=state.ts,
            severity_index=severity,
            risk_class=risk_class,
            factors=factors,
            admissible=risk_class != "high",
            constraints=constraints,
            notes=notes,
        )
