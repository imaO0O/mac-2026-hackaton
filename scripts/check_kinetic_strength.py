"""Какую кинетику отклика брать рабочей: измерение против принятого. Только CPU.

    python scripts/check_kinetic_strength.py

Зачем. Отклик серы на уставки в системе был ПРИНЯТ (первый порядок, 100 кДж/моль), а
теперь ИЗМЕРЕН по 48 чистым эпизодам (`docs/SULFUR_RESPONSE.md`, участник 2):
наблюдаемое изменение серы — 0.035 от того, что обещает первый порядок (на
валидации 0.056, 90 % интервал −0.038…0.145). Первый порядок измерением исключён;
второй (0.14) лежит в интервале валидации, но не обучения.

Три варианта рабочей кинетики (`optimization.kinetic_order`, `kinetic_strength`):

* **A** — первый порядок в полную силу, как сейчас: −19 % серы на градус;
* **B** — второй порядок: −2.9 % на градус (0.14 от первого);
* **C** — первый порядок с измеренной силой 0.035: −0.7 % на градус;
* **D** — порядок 1.5: −5.6 % на градус (0.27 от первого).

**Опора сменилась, и это надо назвать прямо.** Первая редакция правила судила по
истории: отклик, измеренный по чистым эпизодам, составляет 0.035–0.056 от первого
порядка. По этому правилу выбор оставался за A (варианты слабее портили Т95), и это
записано как измеренный отказ. Затем в записи сессии 11.09 нашёлся ответ эксперта,
потерянный в присланной расшифровке (`docs/transcripts/qa_2026-09-11.txt`, 13:56):
**подъём температуры на 1 °C снижает серу на 0.3–1.0 ppm** при уровне около 8 ppm,
то есть на 4–12 % — литературные 5–10 %. Там же объяснено, почему история даёт
меньше: контуры обратной связи «сильно затрудняют эту оценку».

Опора практикой заказчика внешняя: она не выведена ни из наших данных, ни из теста.
Поэтому правило переписано и ЗАПИСАНО ДО НОВОГО СЧЁТА (`docs/PLAN.md`): берётся
вариант, чьё обещанное снижение серы на +1 °C попадает в названный экспертом
диапазон 0.3–1.0 ppm; среди попавших — ближайший к середине 0.65 ppm. Обязательные
условия прежние, считаются на валидационном окне имитации:

1. шагов с превышением по сере не больше, чем у варианта A;
2. вариант не переводит Т95 через предел: шагов, где Т95 из-под предела уходит за
   него из-за нас, — ноль, и доля времени выше предела не больше, чем без нас.

**Вторая редакция условия по Т95, и об этом надо сказать прямо.** Сначала условие
сравнивало вклад в Т95 с вариантом A: «не выше, чем у A, больше чем на 0.1 °C». По
нему отсеивался любой вариант, который действует чаще, — а слабее принятая кинетика,
тем больше шагов нужно, и тем выше вклад. То есть условие меряло активность, а не
безопасность Т95, и отсекало ровно то, ради чего его писали. Проверка по смыслу: в
обоих окнах ни один вариант не перевёл Т95 через предел ни разу, а доля времени выше
предела одинакова с нами и без нас (0.023 и 0.120 — Т95 выходит за 360 °C и без
всякого вмешательства). Условие переписано на это, прежняя редакция и её вердикт
остаются в отчёте полем «прежнее условие по Т95».

Ни один не проходит условия — остаётся A, и это записывается как измеренный отказ.
Робастная гарантия в сравнении не участвует: она проверяет запас при более слабой
кинетике и остаётся включённой при любом выборе.

Результат: reports/kinetic_strength.json.
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
from nefte.models.regime import REACTOR_TEMPS  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import reliability_provenance, report_provenance  # noqa: E402
from nefte.sim import ClosedLoopSimulator, summarize  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "kinetic_strength.json"
MEASURED = ROOT / "reports" / "sulfur_response.json"
# окно валидационного периода: выбор делается только на нём
WINDOW = "stable"
VARIANTS = {
    "A: первый порядок, как принято": {"kinetic_order": 1.0, "kinetic_strength": 1.0},
    "B: второй порядок": {"kinetic_order": 2.0, "kinetic_strength": 1.0},
    "C: первый порядок, измеренная сила": {"kinetic_order": 1.0, "kinetic_strength": 0.035},
    "D: порядок 1.5": {"kinetic_order": 1.5, "kinetic_strength": 1.0},
}
T95_TOLERANCE = 0.1
# Практика заказчика: 1 °C даёт от 0.3 до 1.0 ppm при уровне около 8 ppm
# (docs/transcripts/qa_2026-09-11.txt, 13:56). Середина — ориентир выбора.
PRACTICE_PPM = (0.3, 1.0)


def measured_beta() -> tuple[float | None, list | None]:
    """Измеренная доля первого порядка на ВАЛИДАЦИИ и её 90 % интервал."""
    if not MEASURED.exists():
        return None, None
    data = json.loads(MEASURED.read_text(encoding="utf-8"))
    block = ((data.get("итог") or {}).get("все чистые") or {}).get("val") or {}
    return block.get("β"), block.get("90% интервал")


def predicted_response(system, sb, stamps, base_fn) -> tuple[float, float]:
    """Обещанный откликом вариант: доля первого порядка и мг/кг на градус.

    Считается тем же способом, что и измерение: подъём реакторных температур на
    1 °C и изменение серы, усреднённое по моментам окна. Абсолютная величина нужна
    для сравнения с практикой заказчика (0.3–1.0 ppm на градус), относительная —
    с оценкой по истории.
    """
    ours, base, absolute = [], [], []
    for ts in stamps:
        state = sb.build(ts)
        temps = {t: state.telemetry_ht.get(t) for t in REACTOR_TEMPS}
        if any(v is None for v in temps.values()):
            continue
        hotter = {t: v + 1.0 for t, v in temps.items()}
        for fn, box in ((system.optimizer.surrogate, ours), (base_fn, base)):
            now = fn(state, {})["product_sulfur_mgkg"]
            up = fn(state, hotter)["product_sulfur_mgkg"]
            if now == now and up == up and now > 0:
                box.append((up - now) / now)
                if box is ours:
                    absolute.append(now - up)
    if not ours or not base:
        return float("nan"), float("nan")
    return (float(pd.Series(ours).mean() / pd.Series(base).mean()),
            float(pd.Series(absolute).mean()))


def run_variant(sb, cfg, knobs: dict, stamps) -> dict:
    local = {**cfg, "optimization": {**cfg["optimization"], **knobs}}
    system = build_system(sb, local)
    system.log_runs = False
    sim = ClosedLoopSimulator(sb, system, system.optimizer.surrogate)
    steps = sim.run(stamps, hist_sulfur=sb.lims_sulfur)
    rep = summarize(steps, cfg["spec"]["product_sulfur_mgkg"]["max"],
                    t95_limit=cfg["spec"]["t95_c"]["max"])
    paired = rep.get("сера_без_вмешательства_сим") or {}
    contribution = rep.get("Т95_наш_вклад") or {}
    t95, hist95 = rep.get("Т95") or {}, rep.get("Т95_история") or {}
    return {
        "настройки": knobs,
        "вмешательств": rep["вмешательств"],
        "шагов": rep["шагов"],
        "выше предела с нами, шагов": paired.get("выше предела с нами, шагов"),
        "выше предела без нас, шагов": paired.get("выше предела без нас, шагов"),
        "сера с нами": paired.get("среднее с нами"),
        "вклад в Т95, °C": contribution.get("средний сдвиг"),
        # безопасность Т95 по смыслу: перевели ли мы её через предел и выросла ли
        # доля времени за пределом по сравнению с тем же окном без вмешательств
        "Т95 перевели через предел, шагов": contribution.get("перевели через предел, шагов", 0),
        "Т95 выше предела с нами, доля": t95.get("доля выше предела"),
        "Т95 выше предела без нас, доля": hist95.get("доля выше предела"),
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default=WINDOW, help="окно валидационного периода")
    ap.add_argument("--every", default="4h")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][args.window])
    stamps = pd.date_range(lo, hi, freq=args.every)
    beta, interval = measured_beta()
    print(f"измеренная доля первого порядка на валидации: {beta} {interval}")
    print(f"окно {args.window}: {lo:%Y-%m-%d} … {hi:%Y-%m-%d}, шаг {args.every}\n")

    base_system = build_system(sb, {**cfg, "optimization": {
        **cfg["optimization"], "kinetic_order": 1.0, "kinetic_strength": 1.0}})
    base_fn = base_system.optimizer.surrogate

    rows = {}
    for name, knobs in VARIANTS.items():
        system = build_system(sb, {**cfg, "optimization": {**cfg["optimization"], **knobs}})
        system.log_runs = False
        row = run_variant(sb, cfg, knobs, stamps)
        share, ppm = predicted_response(system, sb, stamps[::4], base_fn)
        row["обещанная доля первого порядка"] = round(share, 4)
        row["обещанное снижение, мг/кг на градус"] = round(ppm, 3)
        row["в диапазоне практики"] = bool(PRACTICE_PPM[0] <= ppm <= PRACTICE_PPM[1])
        row["расстояние до середины практики"] = round(abs(ppm - sum(PRACTICE_PPM) / 2), 3)
        if beta is not None:
            row["расхождение с историей"] = round(
                abs(row["обещанная доля первого порядка"] - float(beta)), 4)
        rows[name] = row
        print(f"{name}: {row}", flush=True)

    base = rows["A: первый порядок, как принято"]
    verdict = {}
    for name, row in rows.items():
        over_ok = (row["выше предела с нами, шагов"] or 0) <= (
            base["выше предела с нами, шагов"] or 0)
        # прежняя редакция условия — сравнение вклада с наименее активным вариантом
        old_t95_ok = (row["вклад в Т95, °C"] or 0.0) <= (
            base["вклад в Т95, °C"] or 0.0) + T95_TOLERANCE
        crossed = int(row["Т95 перевели через предел, шагов"] or 0)
        share_ok = (row["Т95 выше предела с нами, доля"] or 0.0) <= (
            row["Т95 выше предела без нас, доля"] or 0.0) + 1e-9
        t95_ok = crossed == 0 and share_ok
        verdict[name] = {"превышений по сере не больше, чем у A": bool(over_ok),
                         "Т95 через предел не переведена": bool(t95_ok),
                         "в диапазоне практики 0.3–1.0 мг/кг на градус":
                             bool(row["в диапазоне практики"]),
                         "прежнее условие по Т95 (вклад не хуже A на 0.1 °C)": bool(old_t95_ok),
                         "допустим": bool(over_ok and t95_ok and row["в диапазоне практики"])}
    allowed = [n for n, v in verdict.items() if v["допустим"]]
    choice = (min(allowed, key=lambda n: rows[n]["расстояние до середины практики"])
              if allowed else "A: первый порядок, как принято")

    print("\nПравило приёмки:")
    for name, v in verdict.items():
        print(f"  {name}: {v}")
    print(f"  ВЫБОР: {choice}")

    REPORT.write_text(json.dumps({
        **report_provenance(cfg), **reliability_provenance(cfg),
        "окно": args.window, "шаг": args.every,
        "измеренная_доля_первого_порядка": {"валидация": beta, "90% интервал": interval},
        "практика_заказчика_мг_кг_на_градус": list(PRACTICE_PPM),
        "варианты": rows, "правило": verdict, "выбор": choice,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
