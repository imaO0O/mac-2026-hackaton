"""Что, если сера отвечает на температуру слабее, чем думает система. Только CPU.

    python scripts/check_kinetic_order.py
    python scripts/check_kinetic_order.py --orders 1 1.5 2 --window quality_risk --every 4h

Зачем. Отклик серы на уставки в системе не измерен, а принят: кинетика
гидрообессеривания первого порядка с энергией активации 100 кДж/моль
(`models/kinetics.py`). Устойчивость к энергии активации проверялась
(`docs/REGIME_FEATURES.md`, ±20 % в диапазоне 70–130 кДж/моль), а к порядку
реакции — нет, и порядок весит больше. При сере сырья 9000 и продукта 8 мг/кг:

* первый порядок — около −19 % серы на градус;
* порядок 1.5 — около −6 %;
* второй порядок — около −3 %.

Литература для глубокой гидроочистки называет 5–10 % на градус, то есть ближе к
порядку 1.5, чем к первому. На сессии вопросов 11.09 другая команда назвала
оценку по истории — 0.03 мг/кг на градус, в двадцать раз меньше литературной;
ответ эксперта в записи потерян. Историческая оценка занижена по построению:
организаторы на той же сессии сказали, что на установках стоят системы
управления с обратной связью, быстро отвечающие на изменение качества, а такой
контур гасит видимую связь «температура → сера».

Две части проверки.

1. **Отклик на +1 °C** реакторных температур в 12 моментах тестового периода при
   каждом порядке: в мг/кг и в процентах от уровня.
2. **Замкнутый контур с рассогласованием**: система, как и в работе, считает по
   первому порядку, а «процесс» в имитации отвечает по порядку n. Прогон с n = 1 —
   это обычный `scripts/run_simulation.py`; остальные показывают, что станет с
   серой, числом вмешательств и Т95, если физика слабее, чем думает оптимизатор.

Результат: таблицы в консоли и reports/kinetic_order.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.models.kinetics import make_kinetic_surrogate  # noqa: E402
from nefte.models.quality_model import make_model_surrogate  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.sim import ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "kinetic_order.json"
REACTOR = ("T5", "T6", "T11")


def response_table(sb, model, cfg, orders, n: int = 12) -> list[dict]:
    lo, hi = cfg["split"]["test"]
    stamps = pd.date_range(lo, hi, periods=n)
    base_fn = make_model_surrogate(model)
    rows = []
    for order in orders:
        fn = make_kinetic_surrogate(model, base_surrogate=base_fn, order=order)
        mg, rel = [], []
        for ts in stamps:
            state = sb.build(ts)
            temps = {t: state.telemetry_ht.get(t) for t in REACTOR}
            if any(v is None for v in temps.values()):
                continue
            base = fn(state, {})["product_sulfur_mgkg"]
            hotter = fn(state, {t: v + 1.0 for t, v in temps.items()})["product_sulfur_mgkg"]
            if base == base and hotter == hotter and base > 0:
                mg.append(hotter - base)
                rel.append((hotter - base) / base)
        rows.append({"порядок": order, "моментов": len(mg),
                     "Δ серы на +1 °C, мг/кг": round(float(pd.Series(mg).mean()), 3),
                     "Δ серы на +1 °C, %": round(100 * float(pd.Series(rel).mean()), 1)})
    return rows


def closed_loop(sb, cfg, order, stamps) -> dict:
    system = build_system(sb, cfg)
    system.log_runs = False
    model = system.quality.model
    process = (system.optimizer.surrogate if order == 1.0 else
               make_kinetic_surrogate(model, base_surrogate=make_model_surrogate(model),
                                      order=order))
    sim = ClosedLoopSimulator(sb, system, process)
    steps = sim.run(stamps, hist_sulfur=sb.lims_sulfur)
    report = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                       t95_limit=cfg["spec"]["t95_c"]["max"])
    # сколько раз уставка упиралась в предел дрейфа имитации: при слабой физике
    # система будет просить ещё и ещё, и это видно здесь раньше, чем в сере
    clamped = sum(1 for s in steps if s.outcome == "меняем уставки" and not s.applied)
    at_limit = {tag: round(v, 2) for tag, v in sim.offsets.items()
                if abs(v) >= sim.max_drift.get(tag, float("inf")) - 1e-9}
    return {
        "порядок процесса": order,
        "шагов": report["шагов"],
        "вмешательств": report["вмешательств"],
        "не применено (упёрлись в дрейф)": clamped,
        "сера_сим": report["сера_сим"],
        "сера_без_вмешательства_сим": report.get("сера_без_вмешательства_сим"),
        "сера_история": report["сера_история"],
        "уставки": report["уставки"],
        "в пределе дрейфа к концу": at_limit,
        "Т95_наш_вклад": report.get("Т95_наш_вклад"),
        "Т95": report.get("Т95"),
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--window", default="quality_risk")
    ap.add_argument("--every", default="4h")
    ap.add_argument("--no-loop", action="store_true", help="только таблица отклика")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    probe = build_system(sb, cfg)
    model = probe.quality.model
    if model is None:
        print("Нет обученной модели качества: scripts/train_quality.py --horizon 0")
        return 1

    table = response_table(sb, model, cfg, args.orders)
    print("\nОтклик серы на подъём реакторных температур на 1 °C "
          "(12 моментов тестового периода)\n")
    print(pd.DataFrame(table).to_string(index=False))
    print("\nСправка: литература для глубокой гидроочистки — 5–10 % на градус; "
          "оценка другой команды по истории — 0.03 мг/кг (занижена контуром "
          "регулирования).")

    loops = []
    if not args.no_loop:
        lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][args.window])
        stamps = pd.date_range(lo, hi, freq=args.every)
        print(f"\nЗамкнутый контур {lo:%Y-%m-%d} … {hi:%Y-%m-%d}, шаг {args.every}: "
              "система считает по первому порядку, процесс — по порядку n\n")
        for order in args.orders:
            row = closed_loop(sb, cfg, order, stamps)
            loops.append(row)
            paired = row["сера_без_вмешательства_сим"] or {}
            print(f"n = {order:g}: сера {paired.get('среднее с нами')} против "
                  f"{paired.get('среднее без нас')} без нас; выше предела "
                  f"{paired.get('выше предела с нами, шагов')} шагов против "
                  f"{paired.get('выше предела без нас, шагов')}; вмешательств "
                  f"{row['вмешательств']}, упёрлись в дрейф {row['не применено (упёрлись в дрейф)']}, "
                  f"в пределе к концу {row['в пределе дрейфа к концу'] or 'нет'}; "
                  f"вклад в Т95 {((row['Т95_наш_вклад'] or {}).get('средний сдвиг'))} °C",
                  flush=True)

    REPORT.write_text(json.dumps({
        **report_provenance(cfg),
        "окно": args.window, "шаг": args.every,
        "отклик_на_градус": table,
        "замкнутый_контур": loops,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
