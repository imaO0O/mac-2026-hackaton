"""Агент оптимизации.

Генерирует варианты изменения режима, отсекает недопустимые и ранжирует
оставшиеся. Жёсткие ограничения проверяются ДО ранжирования: никакая экономика
не может «перевесить» нарушение спецификации (ключевой принцип ТЗ).
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from nefte.agents.quality import spec_risk_normal
from nefte.agents.schemas import (
    Candidate,
    ProcessState,
    QualityAssessment,
    ReliabilityAssessment,
)
from nefte.config import load_config


class QualitySurrogate(Protocol):
    """Модель «режим → качество», которой пользуется оптимизатор.

    Реализует участник 1 поверх обученной модели агента качества.
    """

    def __call__(self, state: ProcessState, moves: dict[str, float]) -> dict[str, float]:
        ...


def linear_surrogate(sensitivities: dict[str, float], base_key: str = "product_sulfur_mgkg"
                     ) -> QualitySurrogate:
    """Заглушка-суррогат: линейный отклик качества на изменение уставок.

    Нужна только чтобы цикл работал до появления обученной модели.
    Коэффициенты — ДОПУЩЕНИЕ, в демо не использовать без пометки.
    """

    def _fn(state: ProcessState, moves: dict[str, float]) -> dict[str, float]:
        base = state.quality.get("pak_sulfur_ppm")
        value = base.value if base and base.value is not None else 8.5
        for tag, new in moves.items():
            cur = state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))
            if cur is not None and tag in sensitivities:
                value += sensitivities[tag] * (new - cur)
        return {base_key: float(value)}

    return _fn


class OptimizerAgent:
    """Сэмплирование кандидатов в допустимой области + многокритериальный выбор."""

    def __init__(self, bounds: dict[str, tuple[float, float]],
                 surrogate: QualitySurrogate,
                 cfg: dict | None = None,
                 throughput_fn: Callable[[ProcessState, dict[str, float]], float] | None = None,
                 energy_fn: Callable[[ProcessState, dict[str, float]], float] | None = None,
                 max_spec_risk: float = 0.2):
        self.bounds = bounds                 # модельные диапазоны (допущение!)
        self.surrogate = surrogate
        self.cfg = cfg or load_config()
        # максимально допустимая вероятность нарушения спецификации
        self.max_spec_risk = max_spec_risk
        self.throughput_fn = throughput_fn
        self.energy_fn = energy_fn
        self.rng = np.random.default_rng(self.cfg["optimization"]["random_seed"])
        # последний оценённый набор кандидатов — нужен оркестратору, чтобы объяснить отказ
        self.last_evaluated: list[Candidate] = []

    # ------------------------------------------------------------------ #
    def _effective_bounds(self, state: ProcessState,
                          reliability: ReliabilityAssessment) -> dict[str, tuple[float, float]]:
        """Пересечение модельного диапазона, ограничения агента надёжности и шага."""
        steps = self.cfg["limits"]["max_step_per_cycle"]
        out: dict[str, tuple[float, float]] = {}
        for tag, (lo, hi) in self.bounds.items():
            cur = state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))
            if cur is None:
                continue
            step = steps["temperature_c"] if tag.startswith("T") else (
                steps["pressure_mpa"] if tag.startswith("P") else abs(cur) * steps["flow_rel"])
            lo_e, hi_e = max(lo, cur - step), min(hi, cur + step)
            if tag in reliability.constraints:
                r_lo, r_hi = reliability.constraints[tag]
                lo_e, hi_e = max(lo_e, r_lo), min(hi_e, r_hi)
            if hi_e >= lo_e:
                out[tag] = (lo_e, hi_e)
        return out

    def generate(self, state: ProcessState, reliability: ReliabilityAssessment,
                 n: int | None = None) -> list[Candidate]:
        """Кандидаты: «ничего не делать» + случайные точки допустимой области."""
        n = n or self.cfg["optimization"]["n_candidates"]
        bounds = self._effective_bounds(state, reliability)
        current = {t: state.telemetry_ht.get(t, state.telemetry_avt.get(t)) for t in bounds}

        cands = [Candidate(id="hold", moves={t: v for t, v in current.items() if v is not None},
                           deltas={t: 0.0 for t in bounds})]
        for i in range(n):
            moves, deltas = {}, {}
            for tag, (lo, hi) in bounds.items():
                val = float(self.rng.uniform(lo, hi))
                moves[tag] = val
                deltas[tag] = val - (current[tag] or val)
            cands.append(Candidate(id=f"cand_{i:03d}", moves=moves, deltas=deltas))
        return cands

    # ------------------------------------------------------------------ #
    def evaluate(self, state: ProcessState, cands: list[Candidate],
                 quality: QualityAssessment,
                 reliability: ReliabilityAssessment) -> list[Candidate]:
        """Прогноз качества и проверка жёстких ограничений для каждого кандидата."""
        limit = self.cfg["spec"]["product_sulfur_mgkg"]["max"]
        # запас на неопределённость прогноза: σ восстанавливаем из 95% интервала
        lo, hi = quality.intervals.get("product_sulfur_mgkg", (limit, limit))
        sigma = max((hi - lo) / (2 * 1.96), 0.0)

        for c in cands:
            pred = self.surrogate(state, c.moves)
            c.predicted_quality = pred
            sulfur = pred.get("product_sulfur_mgkg")
            violations = []
            if sulfur is None or sulfur != sulfur:
                risk = 1.0
                violations.append("нет прогноза качества")
            else:
                # допустимость — по ВЕРОЯТНОСТИ нарушения, а не по точечному прогнозу:
                # 9.9 мг/кг при σ=2 это не «в спецификации», а «монетка».
                risk = spec_risk_normal(sulfur, sigma, limit)
                if risk > self.max_spec_risk:
                    violations.append(
                        f"P(сера > {limit}) = {risk:.0%} при допустимых "
                        f"{self.max_spec_risk:.0%} (прогноз {sulfur:.2f} ± {1.96 * sigma:.2f})")
            if not reliability.admissible:
                violations.append("режим признан недопустимым агентом надёжности")

            c.spec_risk = {"product_sulfur_mgkg": risk}
            c.throughput = self.throughput_fn(state, c.moves) if self.throughput_fn else None
            c.energy_proxy = self.energy_fn(state, c.moves) if self.energy_fn else None
            c.severity_index = reliability.severity_index
            c.violations = violations
            c.feasible = not violations
        return cands

    # ------------------------------------------------------------------ #
    def rank(self, cands: list[Candidate]) -> list[Candidate]:
        """Взвешенная свёртка + ранг Парето. Ранжируются ТОЛЬКО допустимые."""
        w = self.cfg["optimization"]["objective_weights"]
        limit = self.cfg["spec"]["product_sulfur_mgkg"]["max"]
        feas = [c for c in cands if c.feasible]
        if not feas:
            return []

        def norm(values: list[float]) -> list[float]:
            arr = np.asarray(values, dtype=float)
            rng = np.nanmax(arr) - np.nanmin(arr)
            return list((arr - np.nanmin(arr)) / rng) if rng > 0 else [0.5] * len(arr)

        margin = norm([limit - (c.predicted_quality.get("product_sulfur_mgkg") or limit)
                       for c in feas])
        thr = norm([c.throughput if c.throughput is not None else 0.0 for c in feas])
        eng = norm([c.energy_proxy if c.energy_proxy is not None else 0.0 for c in feas])
        sev = norm([c.severity_index if c.severity_index is not None else 0.0 for c in feas])

        for i, c in enumerate(feas):
            c.score = float(w["quality_margin"] * margin[i] + w["throughput"] * thr[i]
                            - w["energy_proxy"] * eng[i] - w["severity"] * sev[i])

        objectives = np.column_stack([margin, thr, -np.asarray(eng), -np.asarray(sev)])
        for i, c in enumerate(feas):
            dominated = np.all(objectives >= objectives[i], axis=1) & \
                        np.any(objectives > objectives[i], axis=1)
            c.pareto_rank = int(dominated.sum())

        return sorted(feas, key=lambda c: (-(c.score or 0.0), c.pareto_rank or 0))

    def propose(self, state: ProcessState, quality: QualityAssessment,
                reliability: ReliabilityAssessment) -> list[Candidate]:
        cands = self.generate(state, reliability)
        cands = self.evaluate(state, cands, quality, reliability)
        self.last_evaluated = cands
        return self.rank(cands)

    def rejection_summary(self) -> str:
        """Почему не осталось допустимых вариантов — текст для оператора."""
        from collections import Counter
        reasons = Counter(v.split(" (")[0] for c in self.last_evaluated for v in c.violations)
        if not reasons:
            return "нет данных о причинах отсева"
        top = "; ".join(f"{r} ({n} из {len(self.last_evaluated)} вариантов)"
                        for r, n in reasons.most_common(2))
        return top
