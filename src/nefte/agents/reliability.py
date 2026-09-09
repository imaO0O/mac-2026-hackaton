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
* скорость изменения режима за час — резкие броски сами по себе фактор риска.

Нормировка считается ТОЛЬКО по обучающему периоду. Если брать всю историю,
severity в 2026 году будет нормирован по данным 2026 года, то есть по будущему.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nefte.agents.schemas import ProcessState, ReliabilityAssessment
from nefte.config import load_config


@dataclass
class SeverityNorms:
    """Нормировка прокси-факторов по историческим квантилям.

    ``fit`` считает p05/p95 по обучающему периоду — это модельный диапазон,
    а не промышленный предел. Так требует ТЗ (правило границ).
    """
    bounds: dict[str, tuple[float, float]]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str],
            q: tuple[float, float] = (0.05, 0.95),
            train: tuple[str, str] | None = None) -> "SeverityNorms":
        """``train`` — границы обучающего периода; вне его данные не используются."""
        if train is not None:
            df = df.loc[train[0]:train[1]]
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

    WEIGHTS = {"wabt": 0.40, "dp_r202": 0.25, "furnace": 0.15, "ramp": 0.20}
    REACTOR_TEMPS = ["T5", "T6", "T11"]
    DP_TAG = "W10"
    FURNACE_TAG = "T55"

    def __init__(self, norms: SeverityNorms | None = None,
                 ramp_series: pd.Series | None = None):
        self.norms = norms or SeverityNorms(bounds={})
        # нормированная скорость изменения режима, посчитанная по истории
        self.ramp_series = ramp_series

    # ------------------------------------------------------------------ #
    @classmethod
    def from_history(cls, avt: pd.DataFrame, ht: pd.DataFrame,
                     cfg: dict | None = None) -> "ReliabilityAgent":
        """Собирает агента по истории: нормировка и скорость — только по train."""
        cfg = cfg or load_config()
        train = tuple(cfg["split"]["train"])

        temps = [t for t in cls.REACTOR_TEMPS if t in ht.columns]
        frame = pd.DataFrame(index=ht.index)
        if temps:
            frame["wabt"] = ht[temps].mean(axis=1)
        if cls.DP_TAG in ht.columns:
            frame[cls.DP_TAG] = ht[cls.DP_TAG]
        if cls.FURNACE_TAG in avt.columns:
            frame[cls.FURNACE_TAG] = avt[cls.FURNACE_TAG]

        norms = SeverityNorms.fit(frame, list(frame.columns), train=train)

        # скорость изменения режима: максимальный по реакторным температурам модуль
        # изменения за час, нормированный на p95 обучающего периода
        ramp = None
        if temps:
            step = pd.Series(ht.index).diff().median()
            per_hour = max(int(pd.Timedelta("1h") / step), 1) if step else 6
            delta = ht[temps].diff(per_hour).abs().max(axis=1)
            scale = float(delta.loc[train[0]:train[1]].quantile(0.95)) or 1.0
            ramp = (delta / scale).clip(0.0, 1.5)

        return cls(norms=norms, ramp_series=ramp)

    # ------------------------------------------------------------------ #
    def _ramp_at(self, ts) -> float | None:
        if self.ramp_series is None:
            return None
        sub = self.ramp_series.loc[:ts].dropna()
        return None if sub.empty else float(sub.iloc[-1])

    def _factors(self, state: ProcessState) -> dict[str, float]:
        ht, avt = state.telemetry_ht, state.telemetry_avt
        temps = [ht.get(t) for t in self.REACTOR_TEMPS if ht.get(t) is not None]
        factors: dict[str, float] = {}

        if temps:
            n = self.norms.normalize("wabt", float(np.mean(temps)))
            if n is not None:
                factors["wabt"] = n

        for key, tag, store in (("dp_r202", self.DP_TAG, ht), ("furnace", self.FURNACE_TAG, avt)):
            n = self.norms.normalize(tag, store.get(tag))
            if n is not None:
                factors[key] = n

        ramp = self._ramp_at(state.ts)
        if ramp is not None:
            factors["ramp"] = float(np.clip(ramp, 0.0, 1.0))
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
        severity = float(np.clip(sum(factors[k] * w[k] for k in factors) / total, 0.0, 1.0))

        risk_class = "low" if severity < 0.5 else "medium" if severity < 0.8 else "high"
        notes = []
        constraints: dict[str, tuple[float, float]] = {}

        top = max(factors, key=factors.get)
        notes.append(f"Основной вклад в тяжесть режима: {top} ({factors[top]:.2f}).")

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

        if factors.get("ramp", 0) > 0.8:
            notes.append("Режим меняется быстро — дополнительные воздействия нежелательны.")

        return ReliabilityAssessment(
            ts=state.ts,
            severity_index=severity,
            risk_class=risk_class,
            factors=factors,
            admissible=risk_class != "high",
            constraints=constraints,
            notes=notes,
        )
