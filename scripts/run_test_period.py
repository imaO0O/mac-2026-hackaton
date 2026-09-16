"""Сквозной прогон системы по тестовому периоду 2026 года.

    python scripts/run_test_period.py                 # шаг 12 часов
    python scripts/run_test_period.py --every 6h
    python scripts/run_test_period.py --split val --every 1h --lockout-hours 8

Второй вариант нужен для подбора того, что подбирать на тесте нельзя: прогон по
валидационному периоду пишет отчёт с префиксом `val_`, а переопределённый запрет
частых воздействий — с суффиксом `lock<часы>`, так что рабочие отчёты не
затираются.

Тестовый период (`configs/config.yaml → split.test`) при разработке не трогали:
модель обучена на train, пороги подобраны на val. Здесь система проходит его
целиком и отвечает на вопрос, который на защите зададут первым: **что она вообще
делает в течение полугода эксплуатации и не пропускает ли превышения.**

Считается четыре вещи:

1. Из чего складываются исходы: держим режим, меняем уставки, отказываемся — и
   по какой причине отказываемся.
2. Пропуски: моменты, после которых лаборатория показала превышение по сере, а
   система советовала держать режим. Это цена ошибки, и её надо назвать самим.
3. Цена порога вмешательства: тот же прогон пересобирается при разных порогах,
   и видно, чем оплачивается каждый пойманный случай. Порог перебирается
   постфактум по сохранённому риску, заново считать цикл не нужно.
4. Смешение: сколько раз рецептура допустима и какова предельная доля прямогонки.

Ложные тревоги считаются симметрично: система вмешалась, а превышения не было.
Обе величины нужны вместе — иначе «ноль пропусков» достигается вмешательством на
каждом такте.

Результат: reports/test_period.json и таблица в консоли. Прогон только читает
данные и ничего не подбирает.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.agents.schemas import Recommendation  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.provenance import reliability_provenance  # noqa: E402
from nefte.models.dataset import FEATURE_VERSION  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "test_period.json"

# Окна, на которых сверяемся с лабораторией. Одного окна мало: рабочая модель —
# виртуальный анализатор (горизонт 0), он оценивает серу СЕЙЧАС, а не через сутки.
# На коротком окне видно, ловит ли система то, что уже происходит; на суточном —
# насколько она вообще способна предупреждать. Прогноз на 2 часа в этих данных не
# работает (AUC 0.48, docs/QUALITY_AGENT.md), поэтому разница между окнами
# показывает не качество кода, а предел, заданный самими данными.
CHECK_WINDOWS = (6.0, 12.0, 24.0)
MAIN_WINDOW = 24.0

# Рабочий порог вмешательства подставляется из оркестратора (подобран на val).
ACT_THRESHOLD = 0.2


def abstain_kind(rec: Recommendation) -> str:
    """Причина отказа в терминах, понятных технологу, а не по тексту целиком."""
    reason = rec.abstain_reason.lower()
    if "остановлен" in reason:
        return "установка остановлена"
    if "жёсткие ограничения" in reason or "жесткие ограничения" in reason:
        return "нет допустимых вариантов"
    if any(word in reason for word in ("заморожен", "устарел", "нет достоверного",
                                       "заглушк")):
        return "недостоверные данные"
    return "прочее"


# пороги, которые перебираются в отчёте; рабочий добавляется к ним сам
SWEEP_THRESHOLDS = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def threshold_sweep(frame: pd.DataFrame, act_threshold: float, hours: float,
                    thresholds=SWEEP_THRESHOLDS) -> list[dict]:
    """Чем платим за снижение порога вмешательства.

    Порог сравнивается с риском, поэтому перебрать его можно по уже
    сохранённому риску, не пересчитывая цикл. Но правило сложнее, чем
    «риск ≥ порога», и наивный перебор даёт числа, не сходящиеся с фактическим
    прогоном. Учитываем оба исключения оркестратора:

    * **вынужденное воздействие** — бездействие недопустимо, текущий режим уже
      у предела, и система вмешивается при любом пороге;
    * **отложенное воздействие** — сработал лимит частоты: недавно уже
      вмешивались и ждём отклика, порог тут ни при чём.

    Оба случая опознаются по расхождению фактического исхода с рабочим порогом,
    поэтому проверка сходимости в прогоне обязана дать совпадение.

    Функция вынесена на уровень модуля, чтобы тот же перебор работал по
    СОХРАНЁННЫМ отчётам (``scripts/compare_decision_curves.py``), а не копией.
    Копия уже однажды жила в черновике и давала таблицу защиты.

    Два счёта пропусков, потому что отказ в первом засчитывается как поимка, и
    такая метрика поощряет нерешение (``docs/HARD_CHECKS.md``):

    * ``доля пропусков`` — превышение, а система держала режим;
    * ``доля пропусков с отказами`` — превышение, а система не вмешалась, в том
      числе потому что отказалась.
    """
    column = f"превышение за {hours:.0f} ч"
    known = frame[frame[column].notna() & frame["риск"].notna()]
    refused = known["исход"].eq("отказ")
    forced = known["исход"].eq("меняем уставки") & known["риск"].lt(act_threshold)
    postponed = known["исход"].eq("держим режим") & known["риск"].ge(act_threshold)
    over = known[column].astype(bool)
    n_over, n_calm = max(int(over.sum()), 1), max(int((~over).sum()), 1)

    out = []
    for threshold in sorted({*(float(t) for t in thresholds),
                             round(float(act_threshold), 4)}):
        acts = ~refused & ~postponed & (known["риск"].ge(threshold) | forced)
        missed = int((over & ~acts & ~refused).sum())
        false_alarm = int((~over & acts).sum())
        out.append({
            "порог": round(float(threshold), 3),
            "поймано": int((over & (acts | refused)).sum()),
            "пропущено": missed,
            "доля пропусков": round(missed / n_over, 3),
            "доля пропусков с отказами": round(int((over & ~acts).sum()) / n_over, 3),
            "ложных тревог": false_alarm,
            "доля ложных тревог": round(false_alarm / n_calm, 3),
            "рабочий": bool(abs(threshold - act_threshold) < 1e-4),
        })
    return out


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", default="12h", help="шаг обхода тестового периода")
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N моментов")
    ap.add_argument("--model", choices=("boost", "seq"), default="boost",
                    help="какой виртуальный анализатор проверяем")
    ap.add_argument("--seq-horizon", type=float, default=None,
                    help="горизонт нейросетевой модели (0 или 2)")
    ap.add_argument("--quality-path", default=None,
                    help="конкретный каталог модели бустинга (сидовый вариант); "
                         "без него — рабочая models/sulfur_h0")
    ap.add_argument("--seq-path", default=None,
                    help="конкретный каталог модели сети (например, сидовый вариант); "
                         "без него берётся основная конфигурация по валидации")
    ap.add_argument("--tag", default="", help="суффикс имени отчёта, чтобы прогоны "
                                              "разных конфигураций не затирали друг друга")
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="какой период проходить; подбирать что-либо можно только на val")
    ap.add_argument("--lockout-hours", type=float, default=None,
                    help="переопределить limits.min_hours_between_actions")
    ap.add_argument("--robust-order", type=float, default=None,
                    help="включить робастную гарантию с кинетикой этого порядка")
    ap.add_argument("--no-robust", action="store_true",
                    help="выключить робастную гарантию (по умолчанию она включена): "
                         "база для scripts/check_robust_recommendation.py")
    ap.add_argument("--repeat-risk-increase", type=float, default=None,
                    help="повтор действия до проявления прошлого — только при росте риска")
    ap.add_argument("--catalyst-factor", choices=("age", "activity"), default=None,
                    help="переопределить reliability.catalyst_factor")
    ap.add_argument("--catalyst-reset", choices=("outage_48h", "catalyst_log"), default=None,
                    help="переопределить reliability.catalyst_reset")
    args = ap.parse_args()

    cfg = load_config()
    if args.lockout_hours is not None:
        cfg = {**cfg, "limits": {**cfg["limits"],
                                 "min_hours_between_actions": float(args.lockout_hours)}}
    if args.robust_order is not None:
        cfg = {**cfg, "optimization": {**cfg["optimization"],
                                       "robust_kinetic_order": float(args.robust_order)}}
    if args.no_robust:
        if args.robust_order is not None:
            ap.error("--no-robust и --robust-order вместе не имеют смысла")
        cfg = {**cfg, "optimization": {**cfg["optimization"], "robust_kinetic_order": None}}
    if args.repeat_risk_increase is not None:
        cfg = {**cfg, "limits": {**cfg["limits"],
                                 "repeat_min_risk_increase": float(args.repeat_risk_increase)}}
    for key in ("catalyst_factor", "catalyst_reset"):
        if getattr(args, key) is not None:
            cfg = {**cfg, "reliability": {**(cfg.get("reliability") or {}),
                                          key: getattr(args, key)}}
    limit_mgkg = cfg["spec"]["product_sulfur_mgkg"]["max"]
    lo, hi = cfg["split"][args.split]

    sb = StateBuilder(cfg)
    system = build_system(sb, cfg, model_kind=args.model, seq_horizon=args.seq_horizon,
                          seq_path=Path(args.seq_path) if args.seq_path else None,
                          quality_path=(Path(args.quality_path) if args.quality_path
                                        else None))
    system.log_runs = False        # полугодовой прогон не засоряет журнал демо
    global ACT_THRESHOLD
    ACT_THRESHOLD = system.act_risk_threshold

    stamps = pd.date_range(lo, hi, freq=args.every)
    if args.limit:
        stamps = stamps[:args.limit]

    lab = sb.lims_sulfur
    rows: list[dict] = []
    for ts in stamps:
        state = sb.build(ts)
        # риск нужен отдельно: в рекомендации он остаётся только текстом, а для
        # перебора порогов постфактум нужно число
        risk = system.quality.assess(state).spec_risk.get("product_sulfur_mgkg")
        rec = system.run(state)
        row = {
            "ts": str(ts),
            "исход": rec.outcome(),
            "риск": None if risk is None else round(float(risk), 4),
            "причина отказа": abstain_kind(rec) if rec.abstained else None,
            "risk_class": rec.state_summary.get("risk_class"),
            "источник": rec.state_summary.get("sulfur_source"),
            "прогноз": (None if rec.action is None else
                        rec.action.predicted_quality.get("product_sulfur_mgkg")),
            "рецептура допустима": None if rec.blend is None else rec.blend.feasible,
        }
        for hours in CHECK_WINDOWS:
            window = lab.loc[ts:ts + pd.Timedelta(hours=hours)]
            fact = float(window.max()) if len(window) else None
            row[f"факт за {hours:.0f} ч"] = fact
            row[f"превышение за {hours:.0f} ч"] = None if fact is None else bool(
                fact > limit_mgkg)
        rows.append(row)

    frame = pd.DataFrame(rows)

    def compare(hours: float) -> dict:
        """Сверка исходов с лабораторией на заданном окне."""
        column = f"превышение за {hours:.0f} ч"
        known = frame[frame[column].notna()]
        over = known[known[column].astype(bool)]
        calm = known[~known[column].astype(bool)]
        missed = over[over["исход"] == "держим режим"]
        reacted = over[over["исход"].isin(["меняем уставки", "отказ"])]
        false_alarm = calm[calm["исход"] == "меняем уставки"]
        return {
            "моментов с фактом": int(len(known)),
            "из них с превышением": int(len(over)),
            "система отреагировала": int(len(reacted)),
            "пропущено (держали режим)": int(len(missed)),
            "доля пропусков": (round(len(missed) / len(over), 3) if len(over) else None),
            "ложных тревог": int(len(false_alarm)),
            "доля ложных тревог": (round(len(false_alarm) / len(calm), 3)
                                   if len(calm) else None),
        }

    checks = {f"{hours:.0f}ч": compare(hours) for hours in CHECK_WINDOWS}

    sweep = threshold_sweep(frame, ACT_THRESHOLD, MAIN_WINDOW)
    # перебор обязан воспроизводить фактический прогон на рабочем пороге —
    # иначе таблица описывает не ту систему, которая только что отработала
    working = next(row for row in sweep if row["рабочий"])
    fact = checks[f"{MAIN_WINDOW:.0f}ч"]
    sweep_matches_run = (working["пропущено"] == fact["пропущено (держали режим)"]
                         and working["ложных тревог"] == fact["ложных тревог"])

    summary = {
        "период": [str(lo), str(hi)],
        "выборка": args.split,
        "запрет частых воздействий, ч": float(cfg["limits"]["min_hours_between_actions"]),
        # как агент надёжности мерил износ катализатора: severity и классы риска
        # зависят от этого, а имя отчёта без флагов об этом молчит
        "износ катализатора": {k: (cfg.get("reliability") or {}).get(k)
                               for k in ("catalyst_factor", "catalyst_reset")},
        "шаг": args.every,
        "модель": args.model,
        "горизонт сети": args.seq_horizon,
        # Какой артефакт модели реально отработал. Без этого поля сидовый вариант,
        # затёрший основную модель, прошёл прогон незамеченным.
        "артефакт модели": (getattr(system.quality.model, "source_path", None) and
                            Path(system.quality.model.source_path).name),
        "моментов": int(len(frame)),
        "исходы": {k: int(v) for k, v in Counter(frame["исход"]).items()},
        "причины отказа": {k: int(v) for k, v in
                           Counter(frame["причина отказа"].dropna()).items()},
        "risk_class": {str(k): int(v) for k, v in Counter(frame["risk_class"]).items()},
        "источник качества": {str(k): int(v) for k, v in Counter(frame["источник"]).items()},
        "сверка с лабораторией": checks,
        "порог вмешательства": {
            "рабочий": round(float(ACT_THRESHOLD), 4),
            "окно сверки, ч": MAIN_WINDOW,
            "сходится с прогоном": bool(sweep_matches_run),
            "перебор": sweep,
        },
        "смешение": {
            "рецептура посчитана": int(frame["рецептура допустима"].notna().sum()),
            "допустима": int((frame["рецептура допустима"] == True).sum()),   # noqa: E712
            "недопустима": int((frame["рецептура допустима"] == False).sum()),  # noqa: E712
        },
    }

    print(f"\nПериод {args.split} {lo} … {hi}, шаг {args.every}, {len(frame)} моментов, "
          f"запрет частых воздействий {cfg['limits']['min_hours_between_actions']:g} ч\n")
    print("Исходы:")
    for name, count in sorted(summary["исходы"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {count:5d}  ({count / len(frame):.0%})")
    if summary["причины отказа"]:
        print("\nПричины отказа:")
        for name, count in sorted(summary["причины отказа"].items(), key=lambda kv: -kv[1]):
            print(f"  {name:26s} {count:5d}")
    print("\nСверка с лабораторией по окнам (строка — окно проверки):")
    print(pd.DataFrame(checks).T.to_string())
    main = checks[f"{MAIN_WINDOW:.0f}ч"]
    print(f"\nНа основном окне {MAIN_WINDOW:.0f} ч: пропущено "
          f"{main['пропущено (держали режим)']} превышений из "
          f"{main['из них с превышением']} (доля {main['доля пропусков']}), "
          f"ложных тревог {main['ложных тревог']} (доля {main['доля ложных тревог']}).")
    print(f"\nЦена порога вмешательства (окно {MAIN_WINDOW:.0f} ч, "
          f"рабочий порог {ACT_THRESHOLD:.2f}):")
    print(pd.DataFrame(sweep).to_string(index=False))
    print("Сходимость с фактическим прогоном на рабочем пороге: "
          + ("да" if sweep_matches_run else "НЕТ — таблице верить нельзя"))

    blend = summary["смешение"]
    print(f"\nСмешение: рецептура посчитана {blend['рецептура посчитана']} раз, "
          f"допустима {blend['допустима']}, недопустима {blend['недопустима']}.")

    # Прогон НЕ рабочей модели не должен затирать отчёт рабочей: его числа идут в
    # документацию, и подмена осталась бы незамеченной. Механизм --tag для этого
    # уже был, но требовал, чтобы о нём помнили; теперь суффикс проставляется сам.
    # Ровно та же ошибка уже случилась с абляцией признаков справочника.
    tag = args.tag
    if not tag and args.model == "boost" and args.quality_path:
        seed = re.search(r"_s(\d+)$", Path(args.quality_path).name)
        tag = f"boost_s{seed.group(1)}" if seed else f"boost_{Path(args.quality_path).name}"
    if not tag and args.model != "boost":
        horizon = args.seq_horizon if args.seq_horizon is not None else 0
        tag = f"{args.model}_h{horizon:g}"
        if args.seq_path:
            seed = re.search(r"_s(\d+)_h", Path(args.seq_path).name)
            if seed:
                tag = f"{args.model}_s{seed.group(1)}_h{horizon:g}"
    # переопределённый запрет и валидационный период пишутся в свои отчёты: их числа
    # служат подбору и не должны подменять рабочий прогон по тесту
    if args.lockout_hours is not None:
        tag = "_".join(x for x in (tag, f"lock{args.lockout_hours:g}") if x)
    if args.robust_order is not None:
        tag = "_".join(x for x in (tag, f"robust{args.robust_order:g}") if x)
    if args.no_robust:
        tag = "_".join(x for x in (tag, "norobust") if x)
    if args.repeat_risk_increase is not None:
        tag = "_".join(x for x in (tag, f"repeat{args.repeat_risk_increase:g}") if x)
    if args.catalyst_factor is not None:
        tag = "_".join(x for x in (tag, f"catalyst_{args.catalyst_factor}") if x)
    if args.catalyst_reset is not None:
        tag = "_".join(x for x in (tag, f"reset_{args.catalyst_reset}") if x)
    stem = "test_period" if args.split == "test" else "val_period"
    report_path = (REPORT.with_name(f"{stem}.json") if not tag
                   else REPORT.with_name(f"{stem}_{tag}.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Версия матрицы и разбиение — чтобы отчёт попадал под контракт
        # свежести. Без них числа устаревают молча: проверка при отсутствии
        # поля делает skip, а пропуск неотличим от успеха. Ровно так девять
        # отчётов сетей оказались вне контракта, который их декларировал.
        "feature_version": FEATURE_VERSION,
        **reliability_provenance(cfg),
        "split": {k: list(v) for k, v in cfg["split"].items()
                  if isinstance(v, (list, tuple))},
        "summary": summary, "rows": rows,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                      default=str), encoding="utf-8")
    print(f"\nОтчёт: {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
