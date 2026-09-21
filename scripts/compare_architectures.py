"""Сравнение архитектурных подходов: что даёт каждый агент.

    python scripts/compare_architectures.py
    python scripts/compare_architectures.py --every 24h

ТЗ требует сравнить несколько архитектурных подходов и оценить их эффективность.
Вопрос, который за этим стоит, простой и неприятный: **а зачем вам четыре агента?**
Ответить на него можно только числом — собрать урезанные версии системы и прогнать
их по одному и тому же тестовому периоду.

Сравниваются четыре конфигурации:

1. **полная** — все агенты: качество, надёжность, оптимизация, смешение, оркестратор;
2. **без агента надёжности** — тяжесть режима не ограничивает диапазоны уставок и
   не участвует в свёртке; проверяем, что даёт ограничение «не греть тяжёлый режим»;
3. **без оптимизатора** — вместо перебора вариантов простое правило: риск выше
   порога → поднять реакторные температуры на максимально разрешённый шаг;
4. **одноагентная** — только прогноз качества и порог: «риск выше порога — тревога»,
   без ограничений, без выбора варианта, без отказов по достоверности данных.

Метрики одни и те же для всех: пропущенные превышения и ложные тревоги по
лаборатории, плюс цена вмешательств — сколько градусов система суммарно просит
подвинуть. Последнее важно: конфигурация, которая «ловит всё», обычно просто
дёргает уставки постоянно.

Результат: reports/architectures.json и таблица в консоли.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.agents.orchestrator import Orchestrator  # noqa: E402
from nefte.provenance import reliability_provenance, report_provenance  # noqa: E402
from nefte.agents.schemas import Candidate, ReliabilityAssessment  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import CONTROL_TAGS, build_system  # noqa: E402


class NoReliability:
    """Заглушка агента надёжности: режим всегда допустим, ограничений нет."""

    def assess(self, state) -> ReliabilityAssessment:
        return ReliabilityAssessment(ts=state.ts, severity_index=0.5, risk_class="low",
                                     admissible=True, constraints={}, factors={},
                                     notes=["агент надёжности отключён (эксперимент)"])

    # Остановку установки распознаёт агент надёжности; без него система о ней не
    # знает — так и отвечаем. Строку карточки «возврат в норму» оркестратор
    # показывает только на работающей установке и спрашивает об этом агента; без
    # метода сравнение архитектур падало (пересборка 21.09).
    def is_unit_down(self, state) -> bool:
        return False


class RuleOptimizer:
    """Вместо перебора — правило: «есть риск → грей реактор на разрешённый шаг».

    Так выглядит система без агента оптимизации: направление угадано физикой, но
    ни выбора между вариантами, ни проверки запаса, ни фронта Парето нет.
    """

    def __init__(self, real, cfg: dict):
        self.real = real                 # нужен для суррогата и диапазонов
        self.cfg = cfg
        self.surrogate = real.surrogate
        self.bounds = real.bounds
        self._hold: Candidate | None = None

    # Трасса цикла решения (orchestrator._trace) спрашивает у оптимизатора счёт
    # подбора. У правила подбора нет — так и отвечаем, а не падаем: без этого
    # сравнение архитектур перестало запускаться, когда появилась трасса.
    def last_stats(self) -> dict[str, int]:
        return {}

    def propose(self, state, quality, reliability) -> list[Candidate]:
        step = float(self.cfg["limits"]["max_step_per_cycle"]["temperature_c"])
        limit = float(self.cfg["spec"]["product_sulfur_mgkg"]["max"])
        risk = quality.spec_risk.get("product_sulfur_mgkg", 0.0)

        current = {t: state.telemetry_ht.get(t, state.telemetry_avt.get(t))
                   for t in self.bounds}
        current = {k: v for k, v in current.items() if v is not None}

        moves = dict(current)
        if risk >= 0.5:                  # правило грубое намеренно: это и есть смысл опыта
            for tag in ("T5", "T6", "T11"):
                if tag in moves:
                    lo, hi = self.bounds.get(tag, (moves[tag] - step, moves[tag] + step))
                    moves[tag] = min(moves[tag] + step, hi)

        deltas = {t: moves[t] - current[t] for t in moves}
        pred = self.surrogate(state, moves)
        sulfur = pred.get("product_sulfur_mgkg", float("nan"))
        candidate = Candidate(
            id="rule", moves=moves, deltas=deltas, predicted_quality=pred,
            spec_risk={"product_sulfur_mgkg": float(sulfur > limit) if sulfur == sulfur
                       else 1.0},
            throughput=None, energy_proxy=None,
            severity_index=reliability.severity_index, feasible=True,
            guaranteed=bool(sulfur == sulfur and sulfur < limit),
        )
        hold_pred = self.surrogate(state, current)
        self._hold = Candidate(id="hold", moves=current,
                               deltas={t: 0.0 for t in current},
                               predicted_quality=hold_pred, feasible=True,
                               severity_index=reliability.severity_index)
        return [candidate, self._hold] if deltas and any(
            abs(d) > 1e-6 for d in deltas.values()) else [self._hold]

    def last_hold(self) -> Candidate | None:
        return self._hold

    def rejection_summary(self) -> str:
        return "правило не нашло допустимого воздействия"

    @staticmethod
    def pareto_front(cands):
        return cands

    @staticmethod
    def diverse_alternatives(cands, k: int = 3, min_distance: float | None = None):
        return cands[:k]


def single_agent_decision(quality, threshold: float) -> str:
    """Одноагентная система: только прогноз и порог, без всего остального."""
    risk = quality.spec_risk.get("product_sulfur_mgkg", 0.0)
    return "меняем уставки" if risk >= threshold else "держим режим"


def effort_of(rec) -> float:
    """Сколько всего система просит подвинуть по реакторным температурам, °C."""
    if rec.abstained or rec.action is None:
        return 0.0
    return sum(abs(d) for t, d in rec.action.deltas.items()
               if t in ("T5", "T6", "T11"))


def evaluate(rows: list[dict], limit: float, hours: float = 24.0) -> dict:
    frame = pd.DataFrame(rows)
    column = f"превышение за {hours:.0f} ч"
    known = frame[frame[column].notna()]
    over = known[known[column].astype(bool)]
    calm = known[~known[column].astype(bool)]
    missed = over[over["исход"] == "держим режим"]
    false_alarm = calm[calm["исход"] == "меняем уставки"]
    return {
        "моментов": int(len(frame)),
        "вмешательств": int((frame["исход"] == "меняем уставки").sum()),
        "отказов": int((frame["исход"] == "отказ").sum()),
        "превышений": int(len(over)),
        "пропущено": int(len(missed)),
        "доля пропусков": round(len(missed) / len(over), 3) if len(over) else None,
        "ложных тревог": int(len(false_alarm)),
        "доля ложных": round(len(false_alarm) / len(calm), 3) if len(calm) else None,
        "суммарно °C": round(float(frame["усилие"].sum()), 1),
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", default="12h", help="шаг обхода тестового периода")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]
    lo, hi = cfg["split"]["test"]

    sb = StateBuilder(cfg)
    full = build_system(sb, cfg)
    full.log_runs = False
    lab = sb.lims_sulfur

    stamps = pd.date_range(lo, hi, freq=args.every)
    if args.limit:
        stamps = stamps[:args.limit]

    # --- конфигурации ---------------------------------------------------- #
    without_reliability = Orchestrator(
        full.quality, NoReliability(), full.optimizer, cfg=cfg, log_runs=False,
        blending=full.blending, components_fn=full.components_fn,
        act_risk_threshold=full.act_risk_threshold)
    rule_based = Orchestrator(
        full.quality, full.reliability, RuleOptimizer(full.optimizer, cfg), cfg=cfg,
        log_runs=False, act_risk_threshold=full.act_risk_threshold)

    systems = {"полная": full, "без надёжности": without_reliability,
               "без оптимизатора": rule_based}
    results = {name: [] for name in systems}
    results["одноагентная"] = []

    print(f"\nСравнение архитектур на тесте {lo} … {hi}, шаг {args.every}, "
          f"{len(stamps)} моментов\n")

    for ts in stamps:
        state = sb.build(ts)
        facts = {}
        for hours in (24.0,):
            window = lab.loc[ts:ts + pd.Timedelta(hours=hours)]
            fact = float(window.max()) if len(window) else None
            facts[f"превышение за {hours:.0f} ч"] = (None if fact is None
                                                     else bool(fact > limit))
        quality = full.quality.assess(state)

        for name, system in systems.items():
            # каждая конфигурация оценивается независимо: лимит частоты воздействий
            # у неё свой, и переносить его между архитектурами нельзя
            system._last_action_ts = None
            rec = system.run(state)
            results[name].append({"исход": rec.outcome(), "усилие": effort_of(rec),
                                  **facts})

        decision = single_agent_decision(quality, full.act_risk_threshold)
        step = float(cfg["limits"]["max_step_per_cycle"]["temperature_c"])
        results["одноагентная"].append({
            "исход": decision,
            "усилие": step * 3 if decision == "меняем уставки" else 0.0,
            **facts})

    table = {name: evaluate(rows, limit) for name, rows in results.items()}
    frame = pd.DataFrame(table).T
    print(frame.to_string())

    out = ROOT / "reports" / "architectures.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**report_provenance(), **reliability_provenance(),
                               "период": [str(lo), str(hi)], "шаг": args.every,
                               "конфигурации": table}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    print("\nЧитать таблицу надо парами «пропуски — ложные тревоги — суммарно °C»: "
          "конфигурация, которая ловит больше, обычно просто чаще дёргает уставки.")
    print(f"Управляющие теги: {', '.join(CONTROL_TAGS)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
