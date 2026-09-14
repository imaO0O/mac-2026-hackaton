"""Замкнутый контур: что будет, если оператор выполняет рекомендации.

    python scripts/run_simulation.py                       # окно quality_risk, шаг 4 ч
    python scripts/run_simulation.py --window stable --every 6h
    python scripts/run_simulation.py --ts "2026-06-01" --days 20

Все остальные прогоны в проекте разомкнуты: система смотрит на историю и говорит,
что сделала бы. Здесь её рекомендации ПРИМЕНЯЮТСЯ — уставки сдвигаются, процесс
отвечает, и на следующем шаге система видит последствия собственного совета.

Что этот прогон может показать и чего не может — надо назвать сразу.

**Не может** проверить, верна ли кинетика: отклик считает та же модель, которой
пользуется оптимизатор, рассуждение замкнуто само на себя.

**Может** ответить на вопросы, на которые разомкнутый прогон не отвечает вообще:

* устойчив ли контур — не гоняет ли система уставки туда-обратно;
* не ползёт ли режим в одну сторону бесконечно;
* сколько всего воздействий на оборудование выходит за период;
* сколько времени продукт вне спецификации, если советы выполнять, и если не
  трогать режим вовсе.

Допущения имитации перечислены в `src/nefte/sim.py`; главные — постоянная времени
отклика 4 часа и то, что сырьё остаётся историческим.
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
from nefte.models.dataset import FEATURE_VERSION  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.sim import RESPONSE_TAU_HOURS, ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "simulation.json"


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="quality_risk",
                    help="имя окна из configs/config.yaml: demo_windows")
    ap.add_argument("--ts", help="начало прогона; заменяет --window")
    ap.add_argument("--days", type=float, default=14.0, help="длительность при --ts")
    ap.add_argument("--every", default="4h", help="шаг цикла управления")
    ap.add_argument("--tau", type=float, default=RESPONSE_TAU_HOURS,
                    help="постоянная времени отклика качества, часов (измерена: "
                         "scripts/find_delays.py)")
    args = ap.parse_args()

    cfg = load_config()
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]
    sb = StateBuilder(cfg)
    system = build_system(sb, cfg)
    system.log_runs = False

    if args.ts:
        lo = pd.Timestamp(args.ts)
        hi = lo + pd.Timedelta(days=args.days)
    else:
        lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][args.window])
    stamps = pd.date_range(lo, hi, freq=args.every)

    print(f"\nЗамкнутый контур: {lo:%Y-%m-%d} … {hi:%Y-%m-%d}, шаг {args.every}, "
          f"{len(stamps)} циклов, постоянная времени {args.tau:g} ч")
    print("Рекомендации ПРИМЕНЯЮТСЯ: уставки сдвигаются и удерживаются.\n")

    sim = ClosedLoopSimulator(sb, system, system.optimizer.surrogate,
                              tau_hours=args.tau)
    steps = sim.run(stamps, hist_sulfur=sb.lims_sulfur)
    report = summarize(steps, limit, t95_limit=cfg["spec"]["t95_c"]["max"])

    print("Исходы:", report["исходы"])
    print(f"Вмешательств: {report['вмешательств']} из {report['шагов']} циклов")
    print(f"\nСера в замкнутом контуре: среднее {report['сера_сим']['среднее']}, "
          f"выше предела {report['сера_сим']['доля выше предела']:.0%} времени")
    paired = report.get("сера_без_вмешательства_сим") or {}
    if paired:
        print(f"Та же имитация без наших воздействий, на тех же {paired['шагов']} шагах: "
              f"среднее {paired['среднее без нас']} против {paired['среднее с нами']}, "
              f"выше предела {paired['выше предела без нас, шагов']} шагов против "
              f"{paired['выше предела с нами, шагов']}")
    if report["сера_история"]["среднее"] is not None:
        print(f"Лаборатория за тот же период (другой ряд, не для сравнения долей): "
              f"среднее {report['сера_история']['среднее']}, выше предела "
              f"{report['сера_история']['доля выше предела']:.0%} времени")

    t95 = report.get("Т95")
    if t95:
        print(f"\nТ95: начало {t95['начало']} °C, конец {t95['конец']} °C, "
              f"уход за прогон {t95['уход за прогон']:+.2f} °C")
        if "предел" in t95:
            print(f"     предел {t95['предел']} °C, минимальный запас "
                  f"{t95['минимальный запас']:+.2f} °C, выше предела "
                  f"{t95['доля выше предела']:.0%} времени")
    hist95 = report.get("Т95_история")
    if t95 and hist95 and "предел" in hist95:
        print(f"     она же без нашего вмешательства: максимум {hist95['максимум']} °C, "
              f"выше предела {hist95['доля выше предела']:.0%} времени")
    contrib = report.get("Т95_наш_вклад")
    if contrib:
        print(f"     НАШ вклад: в среднем {contrib['средний сдвиг']:+.2f} °C, "
              f"худший случай {contrib['худший сдвиг']:+.2f} °C, подняли Т95 в "
              f"{contrib['доля моментов, где мы подняли Т95']:.0%} моментов")
        if "перевели через предел, шагов" in contrib:
            print(f"     перевели Т95 через предел в {contrib['перевели через предел, шагов']} "
                  f"шагах, наибольший сдвиг при этом "
                  f"{contrib['наибольший сдвиг при переводе, °C']:+.3f} °C")
        if report.get("Т95_шагов_без_оценки"):
            print(f"     ВНИМАНИЕ: в {report['Т95_шагов_без_оценки']} шагах Т95 не оценить — "
                  "доли выше посчитаны без них")
        print("     Это и есть проверка «починили серу, сломали Т95»: сравнивать "
              "надо с историей, а не с пределом — Т95 гуляет и без нас.")

    if report["уставки"]:
        print("\nЧто система сделала с уставками:")
        table = pd.DataFrame(report["уставки"]).T
        print(table.to_string())
        print("\n«Смен направления» — сколько раз система передумала и повела "
              "уставку в другую сторону. Много смен — раскачка контура.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "период": [str(lo), str(hi)], "шаг": args.every, "tau_hours": args.tau,
        # Версия матрицы и разбиение — чтобы отчёт попадал под контракт
        # свежести. Без них числа устаревают молча: проверка при отсутствии
        # поля делает skip, а пропуск неотличим от успеха. Ровно так девять
        # отчётов сетей оказались вне контракта, который их декларировал.
        "feature_version": FEATURE_VERSION,
        "split": {k: list(v) for k, v in cfg["split"].items()
                  if isinstance(v, (list, tuple))},
        "итог": report,
        "шаги": [{"ts": str(s.ts), "исход": s.outcome, "применено": s.applied,
                  "сера": round(s.sulfur_sim, 3),
                  "Т95": None if s.t95_sim is None else round(s.t95_sim, 2),
                  "сера_история": None if s.sulfur_hist is None else round(s.sulfur_hist, 3),
                  "смещения": {k: round(v, 3) for k, v in s.offsets.items()},
                  "шаг": {k: round(v, 3) for k, v in s.moved.items()}}
                 for s in steps],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
