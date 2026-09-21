"""Вето тяжести режима: вычёркивать всё или только то, что тяжесть повышает. Только CPU.

    python scripts/check_severity_veto.py

Зачем. Найдено 21.09 (docs/PLAN.md, «Конец цикла катализатора»): при высоком классе
тяжести агент надёжности объявлял режим недопустимым, и оптимизатор вычёркивал ВСЕ
варианты — даже снижающие тяжесть. В конце цикла катализатора тяжесть выше порога
держит износ, уставками его не изменить, и система неделями отвечала «нет допустимых
вариантов»: 752 момента теста из 763 таких отказов, медиана риска 0.117 при пороге
0.17. Кандидат `reliability.severity_veto: raise_only` вычёркивает только варианты,
которые тяжесть повышают или поднимают температуру.

**Правило приёмки записано до счёта** (docs/PLAN.md). Кандидат включается, если
одновременно:

1. на валидационном прогоне доля превышений без вмешательства (24 ч) растёт не больше
   чем на 2 п.п.;
2. в имитации замкнутого контура шагов с превышением не больше, а вклад в Т95 выше
   не больше чем на 0.5 °C;
3. отказов «нет допустимых вариантов» на валидации становится меньше хотя бы вдвое;
4. ни одно рекомендованное действие при высокой тяжести не поднимает температуру.

Прогоны по валидации: нынешний — `val_period_step1h_lock4.json` (запрет 4 ч — это и
есть настройка по умолчанию), с кандидатом — `val_period_step1h_veto_raise_only.json`
(`run_test_period.py --split val --every 1h --tag step1h --severity-veto raise_only`).
Имитация — на валидационном окне `stable` и, для отчёта, на окне конца цикла
катализатора в тесте, где кандидат вообще может что-то изменить.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.sim import ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "severity_veto.json"
RUNS = {"all": "val_period_step1h_lock4.json",
        "raise_only": "val_period_step1h_veto_raise_only.json"}
TEST_RUNS = {"all": "test_period_step1h.json",
             "raise_only": "test_period_step1h_veto_raise_only.json"}
WINDOWS = {"stable": None,                                   # из demo_windows, валидация
           "конец цикла катализатора": ("2026-03-01", "2026-03-31")}
MAX_MISS_GROWTH = 0.02
MAX_T95_RISE = 0.5
SAMPLE = 60


def with_mode(cfg: dict, mode: str) -> dict:
    return {**cfg, "reliability": {**cfg["reliability"], "severity_veto": mode}}


def load_rows(name: str) -> pd.DataFrame | None:
    path = ROOT / "reports" / name
    if not path.exists():
        return None
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    return rows


def run_stats(rows: pd.DataFrame) -> dict:
    column = "превышение за 24 ч"
    known = rows[column].notna()
    over = rows[column].where(known, False).astype(bool)
    real = over & known
    silent = ~rows["исход"].eq("меняем уставки")
    return {
        "моментов": int(len(rows)),
        "исходы": rows["исход"].value_counts().to_dict(),
        "нет допустимых вариантов": int(rows["причина отказа"].eq("нет допустимых вариантов").sum()),
        "доля пропусков (24 ч)": round(float((real & silent).sum() / max(int(real.sum()), 1)), 3),
    }


def simulate(sb, cfg: dict, mode: str, window: tuple[str, str]) -> dict:
    system = build_system(sb, with_mode(cfg, mode))
    system.log_runs = False
    sim = ClosedLoopSimulator(sb, system, system.optimizer.surrogate)
    lo, hi = (pd.Timestamp(x) for x in window)
    steps = sim.run(pd.date_range(lo, hi, freq="4h"), hist_sulfur=sb.lims_sulfur)
    rep = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                    t95_limit=cfg["spec"]["t95_c"]["max"])
    paired = rep.get("сера_без_вмешательства_сим") or {}
    return {"шагов": rep["шагов"], "вмешательств": rep["вмешательств"],
            "исходы": rep["исходы"],
            "выше предела с нами, шагов": paired.get("выше предела с нами, шагов"),
            "выше предела без нас, шагов": paired.get("выше предела без нас, шагов"),
            "вклад в Т95, °C": (rep.get("Т95_наш_вклад") or {}).get("средний сдвиг")}


def temperature_rises(sb, cfg: dict, rows: pd.DataFrame) -> dict:
    """Условие 4: ни одно действие при высокой тяжести не поднимает температуру."""
    chosen = rows[rows["исход"].eq("меняем уставки") & rows["risk_class"].eq("high")]
    chosen = chosen.head(SAMPLE)
    system = build_system(sb, with_mode(cfg, "raise_only"))
    system.log_runs = False
    rises = []
    for ts in chosen["ts"]:
        system._last_action_ts = None
        rec = system.run(sb.build(ts))
        if rec.action is None:
            continue
        up = {tag: round(d, 3) for tag, d in rec.action.deltas.items()
              if tag.startswith("T") and d > 1e-6}
        if up:
            rises.append({"ts": str(ts), "подъём": up})
    return {"проверено действий при высокой тяжести": int(len(chosen)),
            "с подъёмом температуры": len(rises), "примеры": rises[:5]}


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    base, cand = load_rows(RUNS["all"]), load_rows(RUNS["raise_only"])
    if base is None or cand is None:
        print("нет прогонов по валидации — см. докстринг")
        return 1
    sb = StateBuilder(cfg)

    val = {"all": run_stats(base), "raise_only": run_stats(cand)}
    print("валидация, прогон:")
    for mode, stats in val.items():
        print(f"  {mode:10s} {stats}")

    windows = {"stable": tuple(cfg["demo_windows"]["stable"]),
               "конец цикла катализатора": WINDOWS["конец цикла катализатора"]}
    sims = {}
    for name, window in windows.items():
        sims[name] = {mode: simulate(sb, cfg, mode, window) for mode in ("all", "raise_only")}
        print(f"\nимитация «{name}» {window[0]} … {window[1]}:")
        for mode, stats in sims[name].items():
            print(f"  {mode:10s} {stats}")

    rises = temperature_rises(sb, cfg, cand)
    print(f"\nусловие 4: {rises}")

    stable = sims["stable"]
    over_all = stable["all"]["выше предела с нами, шагов"] or 0
    over_new = stable["raise_only"]["выше предела с нами, шагов"] or 0
    t95_all = stable["all"]["вклад в Т95, °C"] or 0.0
    t95_new = stable["raise_only"]["вклад в Т95, °C"] or 0.0
    rule = {
        "1. пропуски на валидации растут не больше чем на 2 п.п.":
            val["raise_only"]["доля пропусков (24 ч)"]
            - val["all"]["доля пропусков (24 ч)"] <= MAX_MISS_GROWTH + 1e-9,
        "2. в имитации превышений не больше, вклад в Т95 выше не больше 0.5 °C":
            over_new <= over_all and t95_new - t95_all <= MAX_T95_RISE + 1e-9,
        "3. отказов «нет допустимых вариантов» меньше хотя бы вдвое":
            val["raise_only"]["нет допустимых вариантов"]
            <= val["all"]["нет допустимых вариантов"] / 2,
        "4. ни одно действие при высокой тяжести не поднимает температуру":
            rises["с подъёмом температуры"] == 0,
    }
    accepted = all(rule.values())
    print("\nПравило приёмки:")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: severity_veto = {'raise_only' if accepted else 'all (кандидат отклонён)'}")

    test = {}
    base_t, cand_t = load_rows(TEST_RUNS["all"]), load_rows(TEST_RUNS["raise_only"])
    if base_t is not None and cand_t is not None:
        test = {"all": run_stats(base_t), "raise_only": run_stats(cand_t)}
        print("\nтест (для отчёта):")
        for mode, stats in test.items():
            print(f"  {mode:10s} {stats}")

    REPORT.write_text(json.dumps({
        **report_provenance(cfg),
        "условие: рост пропусков не больше": MAX_MISS_GROWTH,
        "условие: рост вклада в Т95 не больше, °C": MAX_T95_RISE,
        "валидация": val, "имитация": sims, "условие 4": rises, "тест": test,
        "правило": {k: bool(v) for k, v in rule.items()}, "принят": bool(accepted),
        "выбор": "raise_only" if accepted else "all",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
