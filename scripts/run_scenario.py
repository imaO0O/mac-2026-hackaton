"""Сценарий с возмущением: эксперт меняет условия — система отвечает. Только CPU.

    python scripts/run_scenario.py --list
    python scripts/run_scenario.py --ts "2026-03-05" --days 3 --shock feed_sulfur=+3
    python scripts/run_scenario.py --window bad_data_frozen_pak --shock analyzer_freeze
    python scripts/run_scenario.py --ts "2026-06-01" --days 5 --shock wabt=-4

Зачем. Эксперты сказали прямо, как будут проверять решение (`docs/transcripts/
qa_2026-09-15.txt`, 08:29–09:43): команда показывает свой сценарий, **а потом
эксперты меняют сценарий и смотрят, как отвечает система**. Значит, менять сценарий
должно быть одной командой, а не правкой кода.

Прогон делается ДВАЖДЫ — на истории как есть и с возмущением, — и печатается
разница: изменились ли решения, риск, отказы. Это и есть ответ на вопрос «а если
сырьё станет сернистее» или «а если оба анализатора залипнут».

**Где проходит граница честности.** Возмущение вносится и в срез состояния, и в
матрицу признаков модели: иначе агенты увидели бы одно, а модель — другое, и
сравнение ничего бы не значило. Но процесс на возмущение НЕ отвечает: сера продукта
в истории остаётся прежней. То есть сценарий показывает, как меняется РЕШЕНИЕ при
изменившихся входах, а не как повёл бы себя реальный аппарат. Замкнутый контур с
откликом процесса — отдельный прогон, `scripts/run_simulation.py`.

Производные признаки (средние за 6/36/144 ч, отклонения, наклоны) при ступенчатом
возмущении сдвигаются вместе с уровнем. Для установившегося сдвига это верно точно,
в первые часы после ступени — приближённо; наклоны и отклонения намеренно не
трогаются, чтобы не выдумывать переходный процесс.
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
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

SHOCKS = {
    "feed_sulfur": "сера сырья гидроочистки, мг/кг (например +3)",
    "feed_heavy": "утяжеление сырья: температура отбора на АВТ, °C (например +5)",
    "wabt": "средняя температура реактора, °C (например -4)",
    "analyzer_freeze": "оба поточных анализатора залипают на последнем значении",
    "lab_outage": "лаборатория молчит указанное число часов (например 72)",
}


def parse_shock(text: str) -> tuple[str, float | None]:
    name, _, value = text.partition("=")
    name = name.strip()
    if name not in SHOCKS:
        raise SystemExit(f"неизвестное возмущение {name!r}; список: --list")
    if not value:
        return name, None
    return name, float(value.replace("+", ""))


def columns_like(frame: pd.DataFrame, *prefixes: str) -> list[str]:
    return [c for c in frame.columns if c.startswith(prefixes)]


def apply_shock(sb: StateBuilder, matrix: pd.DataFrame, name: str,
                value: float | None, since: pd.Timestamp) -> dict:
    """Возмущение вносится и в срез, и в признаки модели. Возвращает, что сделано."""
    touched: dict = {"возмущение": name, "величина": value, "с момента": str(since)}
    after = matrix.index >= since

    if name == "feed_sulfur":
        step = float(value if value is not None else 3.0)
        for column in columns_like(matrix, "lims_feed_sulfur"):
            matrix.loc[after, column] = matrix.loc[after, column] + step
        if sb.lims_feed_sulfur is not None:
            mask = sb.lims_feed_sulfur.index >= since
            sb.lims_feed_sulfur.loc[mask] = sb.lims_feed_sulfur.loc[mask] + step
        touched["каналы"] = columns_like(matrix, "lims_feed_sulfur") + ["ЛИМС сырья"]

    elif name == "feed_heavy":
        step = float(value if value is not None else 5.0)
        cols = columns_like(matrix, "avt_T66")
        for column in cols:
            matrix.loc[after, column] = matrix.loc[after, column] + step
        if "T66" in sb.avt.columns:
            mask = sb.avt.index >= since
            sb.avt.loc[mask, "T66"] = sb.avt.loc[mask, "T66"] + step
        touched["каналы"] = cols + ["AVT T66"]

    elif name == "wabt":
        step = float(value if value is not None else -4.0)
        # только УРОВНИ: наклоны и отклонения описывают переходный процесс,
        # которого мы не моделируем
        cols = [c for c in columns_like(matrix, "reg_wabt", "ht_T5", "ht_T11")
                if not c.endswith(("_slope7", "_dev30"))]
        for column in cols:
            matrix.loc[after, column] = matrix.loc[after, column] + step
        mask = sb.ht.index >= since
        for tag in ("T5", "T11", "T6"):
            if tag in sb.ht.columns:
                sb.ht.loc[mask, tag] = sb.ht.loc[mask, tag] + step
        touched["каналы"] = cols + ["HT T5/T6/T11"]

    elif name == "analyzer_freeze":
        for series_name in ("pak_sulfur", "q21_sulfur"):
            series = getattr(sb, series_name, None)
            if series is None or not len(series):
                continue
            before = series.loc[:since].dropna()
            if before.empty:
                continue
            series.loc[series.index >= since] = float(before.iloc[-1])
        for flag in ("pak_frozen", "q21_frozen"):
            series = getattr(sb, flag, None)
            if series is not None and len(series):
                series.loc[series.index >= since] = True
        for column in columns_like(matrix, "ht_Q21", "pak_sulfur"):
            history = matrix.loc[matrix.index < since, column].dropna()
            if not history.empty:
                matrix.loc[after, column] = float(history.iloc[-1])
        touched["каналы"] = ["ПАК", "Q21"] + columns_like(matrix, "ht_Q21", "pak_sulfur")

    elif name == "lab_outage":
        hours = float(value if value is not None else 72.0)
        until = since + pd.Timedelta(hours=hours)
        for series_name in ("lims_sulfur_known", "lims_sulfur", "lims_t95_known"):
            series = getattr(sb, series_name, None)
            if series is None or not len(series):
                continue
            # именно УДАЛЯЕМ точки: оставить NaN на прежнем месте значило бы
            # изобразить «свежий анализ без значения», а молчащая лаборатория —
            # это отсутствие анализа, и возраст последнего обязан расти
            keep = ~((series.index >= since) & (series.index < until))
            setattr(sb, series_name, series[keep])
        for column in columns_like(matrix, "lims_sulfur_prev"):
            history = matrix.loc[matrix.index < since, column].dropna()
            if not history.empty:
                window = after & (matrix.index < until)
                matrix.loc[window, column] = float(history.iloc[-1])
        touched["каналы"] = ["ЛИМС продукта", "ЛИМС Т95"]
        touched["до"] = str(until)

    return touched


def run(system, stamps, sb) -> list[dict]:
    rows = []
    for ts in stamps:
        rec = system.run(sb.build(ts))
        rows.append({
            "ts": str(ts),
            "исход": rec.outcome(),
            "уверенность": round(float(rec.confidence), 2),
            "источник": str(rec.state_summary.get("источник качества", "")),
            "риск": round(float(rec.expected_effect.get("риск", float("nan"))), 3)
            if "риск" in rec.expected_effect else None,
            "сера прогноз, мг/кг": rec.expected_effect.get("сера, мг/кг"),
            "причина отказа": rec.abstain_reason or None,
            "действие": {tag: round(delta, 3) for tag, delta
                         in (rec.action.deltas if rec.action else {}).items()
                         if abs(delta) > 1e-6} or None,
        })
    return rows


def compare(base: list[dict], shocked: list[dict]) -> dict:
    changed = [(b, s) for b, s in zip(base, shocked) if b["исход"] != s["исход"]]
    def share(rows, outcome):
        return round(sum(1 for r in rows if r["исход"] == outcome) / max(len(rows), 1), 3)

    def mean_confidence(rows):
        return round(sum(r["уверенность"] for r in rows) / max(len(rows), 1), 2)
        return round(sum(1 for r in rows if r["исход"] == outcome) / max(len(rows), 1), 3)
    return {
        "моментов": len(base),
        "решений изменилось": len(changed),
        "было: отказов / действий / удержаний": [share(base, "отказ"),
                                                 share(base, "меняем уставки"),
                                                 share(base, "держим режим")],
        "стало: отказов / действий / удержаний": [share(shocked, "отказ"),
                                                  share(shocked, "меняем уставки"),
                                                  share(shocked, "держим режим")],
        "средняя уверенность: было": mean_confidence(base),
        "средняя уверенность: стало": mean_confidence(shocked),
        "примеры": [{"ts": b["ts"], "было": b["исход"], "стало": s["исход"],
                     "причина отказа": s["причина отказа"]}
                    for b, s in changed[:5]],
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="какие возмущения есть")
    ap.add_argument("--window", default="quality_risk",
                    help="имя окна из configs/config.yaml: demo_windows")
    ap.add_argument("--ts", help="начало окна; заменяет --window")
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--every", default="4h")
    ap.add_argument("--shock", action="append", default=None,
                    help="возмущение вида имя=величина, см. --list; можно повторять: "
                         "--shock analyzer_freeze --shock lab_outage=72")
    ap.add_argument("--after", type=float, default=0.0,
                    help="часов от начала окна до возмущения")
    args = ap.parse_args()

    if args.list:
        print("Возмущения:")
        for name, text in SHOCKS.items():
            print(f"  {name:16s} {text}")
        return 0

    cfg = load_config()
    if args.ts:
        lo = pd.Timestamp(args.ts)
        hi = lo + pd.Timedelta(days=args.days)
    else:
        lo, hi = (pd.Timestamp(x) for x in cfg["demo_windows"][args.window])
    stamps = pd.date_range(lo, hi, freq=args.every)
    since = lo + pd.Timedelta(hours=args.after)
    shocks = [parse_shock(text) for text in (args.shock or ["feed_sulfur=+3"])]
    print(f"окно {lo} … {hi}, шаг {args.every}, моментов {len(stamps)}")

    base_builder = StateBuilder(cfg)
    base_system = build_system(base_builder, cfg)
    base_system.log_runs = False
    base = run(base_system, stamps, base_builder)

    shock_builder = StateBuilder(cfg)
    shock_system = build_system(shock_builder, cfg)
    shock_system.log_runs = False
    model = getattr(shock_system.quality, "model", None)
    matrix = getattr(model, "feature_matrix", None)
    if matrix is None:
        raise SystemExit("нет матрицы признаков: сценарий без обученной модели "
                         "покажет только срез, а не прогноз")
    matrix = matrix.copy()
    touched = [apply_shock(shock_builder, matrix, name, value, since)
               for name, value in shocks]
    model.feature_matrix = matrix
    shocked = run(shock_system, stamps, shock_builder)

    summary = compare(base, shocked)
    print("\nвозмущения: " + "; ".join(
        f"{block['возмущение']} {block.get('величина') or ''}".strip()
        for block in touched) + f" — с {since}")
    for key, value_ in summary.items():
        if key != "примеры":
            print(f"  {key}: {value_}")
    for row in summary["примеры"]:
        print(f"    {row['ts']}: {row['было']} → {row['стало']}"
              + (f" ({row['причина отказа']})" if row["причина отказа"] else ""))

    report = ROOT / "reports" / (
        "scenario_" + "_".join(name for name, _ in shocks) + ".json")
    report.write_text(json.dumps({**report_provenance(cfg),
                                  "окно": [str(lo), str(hi)], "шаг": args.every,
                                  "возмущение": touched, "сводка": summary,
                                  "без возмущения": base, "с возмущением": shocked},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
