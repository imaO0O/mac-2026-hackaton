"""Возврат к базовому режиму: что он даёт в замкнутом контуре и чем за это платят.

    python scripts/check_return_to_base.py

Оркестратор рекомендует действие только при риске выше порога, поэтому то, что
однажды сделано ради серы, без возврата не отменяется: расход сырья остаётся
срезанным, температуры — поднятыми, а сера — в разы ниже предела.

Прогон имитации на трёх окнах, с возвратом и без, на одном коде. Параметры возврата
выбираются на окне ВАЛИДАЦИОННОГО периода (`stable`), два окна тестового периода —
проверка. Правило принятия, записанное до счёта: превышений по сере не
прибавляется ни на одном окне и возвраты не отменяются качелями.

Мерится: вмешательства и возвраты; качели — возврат, отменённый действием ради
качества в течение суток; средняя сера и доля выше предела; вклад в Т95; средний
за прогон сдвиг сырья F26 и температуры T11 — именно среднее, а не итоговое
значение, потому что выпуск теряется всё время, пока сдвиг держится.

Результат: таблица в консоли и reports/return_to_base.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nefte.agents.orchestrator as orchestrator_module  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.sim import ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

WINDOWS = {"stable": "валидация (выбор)", "quality_risk": "тест (проверка)",
           "bad_data_frozen_pak": "тест (проверка)"}
REPORT = ROOT / "reports" / "return_to_base.json"


def run(sb, cfg, window: str, enabled: bool) -> dict:
    lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][window])
    local = {**cfg, "limits": {**cfg["limits"], "return_to_base": enabled}}
    system = build_system(sb, local)
    system.log_runs = False
    sim = ClosedLoopSimulator(sb, system, system.optimizer.surrogate)
    steps = sim.run(pd.date_range(lo, hi, freq="4h"), hist_sulfur=sb.lims_sulfur)
    rep = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                    t95_limit=cfg["spec"]["t95_c"]["max"])
    offsets = pd.DataFrame([s.offsets for s in steps]).fillna(0.0)
    contribution = rep.get("Т95_наш_вклад") or {}
    return {
        "шагов": rep["шагов"], "вмешательств": rep["вмешательств"],
        "возвратов": rep.get("из них возвратов к базе", 0),
        "качели": rep.get("качели после возврата", 0),
        "сера_среднее": rep["сера_сим"]["среднее"],
        "сера_выше_предела": rep["сера_сим"]["доля выше предела"],
        "сера_история_выше_предела": rep["сера_история"]["доля выше предела"],
        # та же имитация без наших воздействий — с ней и надо сравнивать
        "сера_без_нас_среднее": (rep.get("сера_без_вмешательства_сим") or {}).get("среднее без нас"),
        "выше_предела_шагов_с_нами": (rep.get("сера_без_вмешательства_сим") or {}).get(
            "выше предела с нами, шагов"),
        "выше_предела_шагов_без_нас": (rep.get("сера_без_вмешательства_сим") or {}).get(
            "выше предела без нас, шагов"),
        "Т95_вклад": contribution.get("средний сдвиг"),
        "F26_средний_сдвиг": round(float(offsets["F26"].mean()), 2) if "F26" in offsets else 0.0,
        "T11_средний_сдвиг": round(float(offsets["T11"].mean()), 2) if "T11" in offsets else 0.0,
    }


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)
    print(f"множитель запаса возврата: {orchestrator_module.RETURN_MARGIN_FACTOR}, пауза "
          f"{cfg['limits'].get('return_settle_hours')} ч\n")
    result: dict = {}
    rows = []
    for window, role in WINDOWS.items():
        result[window] = {"роль": role}
        for enabled in (False, True):
            key = "с возвратом" if enabled else "без возврата"
            result[window][key] = run(sb, cfg, window, enabled)
            rows.append({"окно": window, "роль": role, "вариант": key, **result[window][key]})
    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(frame.to_string(index=False))

    added = {w: (r["с возвратом"]["сера_выше_предела"] > r["без возврата"]["сера_выше_предела"])
             for w, r in result.items()}
    swings = sum(r["с возвратом"]["качели"] for r in result.values())
    accepted = not any(added.values()) and swings == 0
    print("\nПравило: превышений не прибавляется ни на одном окне и качелей нет.")
    print("  прибавились превышения: " + (", ".join(w for w, a in added.items() if a) or "нигде"))
    print(f"  качелей всего: {swings}")
    print(f"  ВЫВОД: возврат {'ПРИНЯТ' if accepted else 'НЕ принят'} для включения по умолчанию")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "множитель_запаса": orchestrator_module.RETURN_MARGIN_FACTOR,
                                  "пауза_ч": cfg["limits"].get("return_settle_hours"),
                                  "окна": result, "превышения_прибавились": added,
                                  "качелей_всего": swings, "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
