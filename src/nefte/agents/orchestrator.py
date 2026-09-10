"""Оркестратор: один цикл принятия решения.

Порядок ровно как в п.1 ТЗ: состояние → проверка данных → качество → надёжность →
варианты → отсев → сравнение → смешение → рекомендация либо мотивированный отказ.

Блок смешения стоит ПОСЛЕ выбора режима, а не рядом с ним: рецептуру считать
имеет смысл для того продукта, который получится в рекомендуемом режиме. Обратный
порядок — «подобрать рецептуру, чтобы вытянуть некачественный продукт» — запрещён
правилом системы: качество и технологические ограничения выше экономики.

Каждый прогон сохраняется в reports/runs/*.json: входные данные, ответы агентов и
итог — чтобы логику решения можно было проверить постфактум (требование ТЗ).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from nefte.agents.blending import BlendingAgent
from nefte.agents.optimizer import OptimizerAgent
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent
from nefte.agents.schemas import (
    BlendComponent,
    ProcessState,
    QualityAssessment,
    Recommendation,
)
from nefte.config import ROOT, load_config

RUNS_DIR = ROOT / "reports" / "runs"


class Orchestrator:
    """Связывает агентов и разрешает конфликт целей."""

    def __init__(self, quality: QualityAgent, reliability: ReliabilityAgent,
                 optimizer: OptimizerAgent, cfg: dict | None = None,
                 min_confidence: float = 0.35, act_risk_threshold: float | None = None,
                 log_runs: bool = True, min_hours_between_actions: float = 4.0,
                 blending: BlendingAgent | None = None,
                 components_fn: Callable[[object], list[BlendComponent]] | None = None,
                 additive_ppm: float = 0.0):
        self.quality = quality
        self.reliability = reliability
        self.optimizer = optimizer
        # Блок смешения необязателен: без него цикл работает как раньше, а
        # рекомендация просто не содержит рецептуры.
        self.blending = blending
        self.components_fn = components_fn
        self.additive_ppm = additive_ppm
        self.cfg = cfg or load_config()
        self.min_confidence = min_confidence
        # порог, начиная с которого вмешиваемся: берём подобранный вместе с моделью
        self.act_risk_threshold = (act_risk_threshold if act_risk_threshold is not None
                                   else getattr(quality, "alarm_threshold", 0.2))
        self.log_runs = log_runs
        # Ограничение частоты воздействий: дёргать уставки каждый цикл нельзя,
        # отклик качества запаздывает и режим не успевает устояться.
        self.min_hours_between_actions = min_hours_between_actions
        self._last_action_ts = None

    # ------------------------------------------------------------------ #
    def run(self, state: ProcessState) -> Recommendation:
        q = self.quality.assess(state)
        r = self.reliability.assess(state)

        freshness = {
            key: m.age_hours for key, m in state.quality.items()
        }
        state_summary = {
            # именно тот источник, по которому принято решение: раньше здесь
            # оказывался первый попавшийся в срезе — например, сера СЫРЬЯ
            "sulfur_source": q.source.value,
            "severity_index": round(r.severity_index, 3),
            "risk_class": r.risk_class,
        }

        # --- отказ 1: установка не в работе -----------------------------
        # Проверяется ПЕРВОЙ: на остановленной установке устаревший ЛИМС и
        # зависший анализатор — следствия останова, а не самостоятельные причины.
        # Назвать оператору следствие вместо причины значит сбить его с толку.
        if not r.admissible and any("остановлена" in note for note in r.notes):
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem="Установка не в работе",
                abstained=True, confidence=q.confidence,
                abstain_reason="; ".join(r.notes),
            )
            return self._finish(rec, state, q, r, blend=False)

        # --- отказ 2: данные непригодны ---------------------------------
        if not state.data_quality.usable or q.confidence < self.min_confidence:
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem="Недостаточно достоверных данных для оценки качества",
                abstained=True, confidence=q.confidence,
                abstain_reason="; ".join(q.notes + state.data_quality.notes)
                or "низкая уверенность прогноза",
            )
            return self._finish(rec, state, q, r, blend=False)

        risk = q.spec_risk.get("product_sulfur_mgkg", 0.0)
        pred = q.predictions.get("product_sulfur_mgkg")

        candidates = self.optimizer.propose(state, q, r)

        # --- отказ 3: допустимых вариантов нет --------------------------
        if not candidates:
            reasons = ["Ни один вариант не проходит жёсткие ограничения",
                       self.optimizer.rejection_summary()]
            reasons += [n for n in r.notes if n]
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem=f"Риск нарушения спецификации по сере: {risk:.0%}",
                abstained=True, confidence=q.confidence,
                abstain_reason=". ".join(x for x in reasons if x) + ". "
                               "Требуется решение технолога.",
            )
            return self._finish(rec, state, q, r)

        best = candidates[0]
        # точка отсчёта — всегда оценённое бездействие, даже если оно недопустимо
        hold = self.optimizer.last_hold() or next(
            (c for c in candidates if c.id == "hold"), None)

        # Воздействовали недавно — ждём отклика, а не накладываем новое изменение.
        # Переступаем через лимит только если продукт УЖЕ вне спецификации: тогда
        # ждать нельзя. Просто высокий риск ожидания не отменяет.
        already_off_spec = pred is not None and pred > self.cfg["spec"]["product_sulfur_mgkg"]["max"]
        if self._too_soon(state.ts) and hold is not None and not already_off_spec:
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem=f"Риск нарушения спецификации: {risk:.0%}",
                action=hold, confidence=q.confidence,
                expected_effect={"сера, мг/кг": round(pred, 2) if pred else "н/д"},
                checked_constraints=self._constraint_log(),
                explanation=(
                    f"Предыдущее воздействие было менее {self.min_hours_between_actions:.0f} ч "
                    "назад. Качество реагирует на изменение режима с запаздыванием, "
                    "поэтому ждём отклика вместо нового вмешательства. "
                    "Спецификация при этом не нарушена."),
                alternatives=self.optimizer.diverse_alternatives(candidates, 3),
            )
            return self._finish(rec, state, q, r)

        # --- нормальный режим: не создаём лишних воздействий ------------
        if risk < self.act_risk_threshold and hold is not None and hold.feasible:
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem=("Режим устойчив, риск выхода за спецификацию "
                         f"{risk:.0%}" if risk < self.act_risk_threshold / 2 else
                         f"Риск {risk:.0%} — ниже порога вмешательства "
                         f"{self.act_risk_threshold:.0%}, режим держим под наблюдением"),
                action=hold, confidence=q.confidence,
                expected_effect=self._effect(hold, hold, r),
                checked_constraints=self._constraint_log(),
                explanation="Текущий режим удовлетворяет ограничениям, "
                            "изменение уставок не требуется.",
                alternatives=self.optimizer.diverse_alternatives(candidates, 3),
            )
            return self._finish(rec, state, q, r)

        # --- есть риск: рекомендуем действие ----------------------------
        rec = Recommendation(
            ts=state.ts, state_summary=state_summary, freshness=freshness,
            problem=f"Риск нарушения спецификации по сере: {risk:.0%} "
                    f"(прогноз {pred:.2f} мг/кг при пределе "
                    f"{self.cfg['spec']['product_sulfur_mgkg']['max']})",
            action=best,
            expected_effect=self._effect(best, hold, r),
            checked_constraints=self._constraint_log(),
            explanation=self._explain(best, candidates, q, r,
                                      q.confidence * (1.0 if best.guaranteed else 0.6)),
            confidence=q.confidence * (1.0 if best.guaranteed else 0.6),
            alternatives=self.optimizer.diverse_alternatives(candidates[1:], 3),
        )
        self._last_action_ts = state.ts if any(
            abs(d) > 1e-6 for d in best.deltas.values()) else self._last_action_ts
        return self._finish(rec, state, q, r)

    # ------------------------------------------------------------------ #
    def _too_soon(self, ts) -> bool:
        if self._last_action_ts is None:
            return False
        hours = (pd.Timestamp(ts) - pd.Timestamp(self._last_action_ts)).total_seconds() / 3600
        return 0 <= hours < self.min_hours_between_actions

    def _effect(self, best, hold, r) -> dict:
        """Эффект показываем относительно бездействия — иначе он ни о чём не говорит."""
        out: dict[str, float | str] = {
            "сера, мг/кг": round(best.predicted_quality.get("product_sulfur_mgkg", float("nan")), 2),
            "тяжесть режима": round(r.severity_index, 2),
        }
        if hold is not None:
            base = hold.predicted_quality.get("product_sulfur_mgkg")
            cur = best.predicted_quality.get("product_sulfur_mgkg")
            if base is not None and cur is not None:
                out["сера к бездействию"] = round(cur - base, 2)
            if hold.throughput and best.throughput is not None:
                out["выпуск, %"] = round((best.throughput / hold.throughput - 1) * 100, 2)
            if hold.energy_proxy and best.energy_proxy is not None:
                out["энергия, %"] = round((best.energy_proxy / hold.energy_proxy - 1) * 100, 2)
        return out

    def _constraint_log(self) -> list[str]:
        """Что именно проверено — и, столь же важно, что НЕ проверено.

        Из показателей спецификации при смене режима мы прогнозируем только серу:
        модели плотности, Т95 и ПТФ у нас нет, данных на неё в пакете тоже. Эти
        показатели проверяются блоком смешения по последнему лабораторному анализу.
        Молчать об этом нельзя: оператор иначе решит, что проверена вся
        спецификация.
        """
        spec = self.cfg["spec"]
        out = [f"сера ≤ {spec['product_sulfur_mgkg']['max']} мг/кг (жёсткое, прогноз)",
               "уставки внутри модельного диапазона (допущение, p05–p95 истории)",
               "шаг изменения за цикл ограничен"]
        if self.blending is not None:
            out.append("плотность, Т95 и ПТФ — по последнему анализу, "
                       "при смене режима НЕ прогнозируются")
        return out

    def _explain(self, best, candidates, q, r, confidence: float | None = None) -> str:
        n_feas = len(candidates)
        n_front = len(self.optimizer.pareto_front(candidates))
        moves = ", ".join(f"{t} {d:+.2f}" for t, d in best.deltas.items() if abs(d) > 1e-6)
        text = (
            f"Из {n_feas} допустимых вариантов ({n_front} на фронте Парето) выбран "
            f"{best.id}: {moves or 'без изменений'}. Он даёт наибольший запас по сере "
            f"при тяжести режима {r.severity_index:.2f} ({r.risk_class}). "
            f"Уверенность {confidence if confidence is not None else q.confidence:.2f}."
        )
        # конфликт целей: качество тянет вверх по температуре, надёжность — вниз
        if not best.guaranteed:
            text += (" ВНИМАНИЕ: запас по неопределённости не гарантирован — вариант "
                     "снижает серу относительно бездействия, но остаётся близко к пределу. "
                     "Нужен контрольный лабораторный анализ.")
        if r.constraints:
            limited = ", ".join(sorted(r.constraints))
            text += (f" Агент надёжности ограничил {limited}, поэтому вариант выбран "
                     f"внутри суженного диапазона, а не по максимуму качества.")
        return text

    # ------------------------------------------------------------------ #
    def _attach_blend(self, rec: Recommendation, state: ProcessState,
                      q: QualityAssessment) -> None:
        """Рецептура смешения для того продукта, который даст выбранный режим.

        Смешение НЕ используется как способ вытянуть некачественный продукт: если
        прогноз серы уже за пределом, допустимой рецептуры не существует, и агент
        это показывает — разбавлять товарное ДТ прямогонкой под Евро-5 нельзя
        (предельная доля порядка сотых долей процента, docs/BLENDING.md).
        """
        if self.blending is None or self.components_fn is None:
            return
        sulfur = self._blend_basis(rec, q)
        if sulfur is None or sulfur != sulfur:
            return
        components = self.components_fn(state.ts)
        if not components:
            return

        recipe = self.blending.with_forecast(components, sulfur,
                                             additive_ppm=self.additive_ppm)
        rec.blend = recipe
        rec.checked_constraints.append(
            f"доли компонентов смешения дают {recipe.fractions_sum() * 100:.1f} % "
            "(жёсткое требование ТЗ)")
        if not recipe.feasible:
            rec.explanation = (rec.explanation + " Смешением это не компенсируется: "
                               + "; ".join(recipe.violations) + ".").strip()
        elif len(recipe.fractions) > 1:
            rec.expected_effect["выпуск смеси, т/ч"] = round(recipe.throughput_tph, 1)

    @staticmethod
    def _blend_basis(rec: Recommendation, q: QualityAssessment) -> float | None:
        """Сера, на которой считается рецептура: прогноз выбранного варианта.

        Если действие не выбрано (отказ), берём прогноз агента качества для
        текущего режима — вопрос «спасёт ли рецептура» задаётся именно к нему.
        """
        if rec.action is not None:
            value = rec.action.predicted_quality.get("product_sulfur_mgkg")
            if value is not None:
                return float(value)
        value = q.predictions.get("product_sulfur_mgkg")
        return None if value is None else float(value)

    def _finish(self, rec: Recommendation, state, q, r,
                blend: bool = True) -> Recommendation:
        if blend:
            self._attach_blend(rec, state, q)
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
