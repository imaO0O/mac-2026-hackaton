"""Оркестратор: один цикл принятия решения.

Порядок ровно как в п.1 ТЗ: состояние → проверка данных → качество → надёжность →
варианты → отсев → сравнение → рекомендация либо мотивированный отказ.

Каждый прогон сохраняется в reports/runs/*.json: входные данные, ответы агентов и
итог — чтобы логику решения можно было проверить постфактум (требование ТЗ).
"""
from __future__ import annotations

import json
from pathlib import Path

from nefte.agents.optimizer import OptimizerAgent
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent
from nefte.agents.schemas import ProcessState, Recommendation
from nefte.config import ROOT, load_config

RUNS_DIR = ROOT / "reports" / "runs"


class Orchestrator:
    """Связывает агентов и разрешает конфликт целей."""

    def __init__(self, quality: QualityAgent, reliability: ReliabilityAgent,
                 optimizer: OptimizerAgent, cfg: dict | None = None,
                 min_confidence: float = 0.35, act_risk_threshold: float | None = None,
                 log_runs: bool = True):
        self.quality = quality
        self.reliability = reliability
        self.optimizer = optimizer
        self.cfg = cfg or load_config()
        self.min_confidence = min_confidence
        # порог, начиная с которого вмешиваемся: берём подобранный вместе с моделью
        self.act_risk_threshold = (act_risk_threshold if act_risk_threshold is not None
                                   else getattr(quality, "alarm_threshold", 0.2))
        self.log_runs = log_runs

    # ------------------------------------------------------------------ #
    def run(self, state: ProcessState) -> Recommendation:
        q = self.quality.assess(state)
        r = self.reliability.assess(state)

        freshness = {
            key: m.age_hours for key, m in state.quality.items()
        }
        state_summary = {
            "sulfur_source": next((m.source.value for m in state.quality.values()
                                   if m.value is not None), "нет"),
            "severity_index": round(r.severity_index, 3),
            "risk_class": r.risk_class,
        }

        # --- отказ 1: данные непригодны ---------------------------------
        if not state.data_quality.usable or q.confidence < self.min_confidence:
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem="Недостаточно достоверных данных для оценки качества",
                abstained=True, confidence=q.confidence,
                abstain_reason="; ".join(q.notes + state.data_quality.notes)
                or "низкая уверенность прогноза",
            )
            return self._finish(rec, state, q, r)

        risk = q.spec_risk.get("product_sulfur_mgkg", 0.0)
        pred = q.predictions.get("product_sulfur_mgkg")

        candidates = self.optimizer.propose(state, q, r)

        # --- отказ 2: допустимых вариантов нет --------------------------
        if not candidates:
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem=f"Риск нарушения спецификации по сере: {risk:.0%}",
                abstained=True, confidence=q.confidence,
                abstain_reason="Ни один вариант не проходит жёсткие ограничения. "
                               f"{self.optimizer.rejection_summary()}. "
                               "Требуется решение технолога.",
            )
            return self._finish(rec, state, q, r)

        best = candidates[0]
        hold = next((c for c in candidates if c.id == "hold"), None)

        # --- нормальный режим: не создаём лишних воздействий ------------
        if risk < self.act_risk_threshold and hold is not None and hold.feasible:
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem=("Режим устойчив, риск выхода за спецификацию "
                         f"{risk:.0%}" if risk < self.act_risk_threshold / 2 else
                         f"Риск {risk:.0%} — ниже порога вмешательства "
                         f"{self.act_risk_threshold:.0%}, режим держим под наблюдением"),
                action=hold, confidence=q.confidence,
                expected_effect={"сера, мг/кг": round(pred, 2) if pred else "н/д"},
                checked_constraints=self._constraint_log(),
                explanation="Текущий режим удовлетворяет ограничениям, "
                            "изменение уставок не требуется.",
                alternatives=candidates[1:4],
            )
            return self._finish(rec, state, q, r)

        # --- есть риск: рекомендуем действие ----------------------------
        rec = Recommendation(
            ts=state.ts, state_summary=state_summary, freshness=freshness,
            problem=f"Риск нарушения спецификации по сере: {risk:.0%} "
                    f"(прогноз {pred:.2f} мг/кг при пределе "
                    f"{self.cfg['spec']['product_sulfur_mgkg']['max']})",
            action=best, confidence=q.confidence,
            expected_effect={
                "сера, мг/кг": round(best.predicted_quality.get("product_sulfur_mgkg", float("nan")), 2),
                "тяжесть режима": round(r.severity_index, 2),
            },
            checked_constraints=self._constraint_log(),
            explanation=self._explain(best, candidates, q, r),
            alternatives=candidates[1:4],
        )
        return self._finish(rec, state, q, r)

    # ------------------------------------------------------------------ #
    def _constraint_log(self) -> list[str]:
        spec = self.cfg["spec"]
        out = [f"сера ≤ {spec['product_sulfur_mgkg']['max']} мг/кг (жёсткое)"]
        out.append("уставки внутри модельного диапазона (допущение, p05–p95 истории)")
        out.append("шаг изменения за цикл ограничен")
        return out

    def _explain(self, best, candidates, q, r) -> str:
        n_feas = len(candidates)
        moves = ", ".join(f"{t} {d:+.2f}" for t, d in best.deltas.items() if abs(d) > 1e-6)
        return (
            f"Из {n_feas} допустимых вариантов выбран {best.id}: {moves or 'без изменений'}. "
            f"Он даёт наибольший запас по сере при тяжести режима "
            f"{r.severity_index:.2f} ({r.risk_class}). "
            f"Уверенность прогноза {q.confidence:.2f}."
        )

    def _finish(self, rec: Recommendation, state, q, r) -> Recommendation:
        if self.log_runs:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            path: Path = RUNS_DIR / f"{state.ts:%Y%m%dT%H%M}.json"
            payload = {
                "state": json.loads(state.model_dump_json()),
                "quality": json.loads(q.model_dump_json()),
                "reliability": json.loads(r.model_dump_json()),
                "recommendation": json.loads(rec.model_dump_json()),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                       default=str), encoding="utf-8")
        return rec
