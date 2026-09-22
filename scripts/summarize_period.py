"""Итог прогона по периоду — так, чтобы эксперт прочёл его за минуту. Только CPU.

    python scripts/summarize_period.py                       # тест, шаг 1 ч
    python scripts/summarize_period.py reports/val_period_step1h_lock4.json
    python scripts/summarize_period.py --cards               # плюс три карточки целиком

Зачем. Эксперты сказали, как будут проверять (18.09, 08:03–08:39): «мы на тесте
задаём период, на котором… решаем те или иные задачи и получаем результат». Прогон
по периоду (`scripts/run_test_period.py`) пишет JSON на тысячи строк — по нему нужно
ещё разобраться, что произошло. Здесь тот же прогон изложен по-человечески:

* что система решала и как часто;
* сверка с лабораторией: сколько превышений было, на сколько система отреагировала,
  сколько пропустила и сколько раз встревожилась зря;
* где система молчала — эпизоды отказов с датами и причинами, и совпадают ли они с
  неисправными периодами данных (`docs/DATA_PERIODS.md`);
* три показательных момента — действие, удержание, отказ — с командой, которая
  покажет карточку целиком (или сами карточки с флагом ``--cards``).

Ничего не пересчитывает, кроме карточек по флагу: всё берётся из отчёта прогона.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.config import ROOT  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

DEFAULT_RUN = ROOT / "reports" / "test_period_step1h.json"
EPISODE_GAP_H = 6.0          # отказы с промежутком не больше — один эпизод


def episodes(rows: pd.DataFrame) -> list[dict]:
    """Отказы подряд — эпизоды с датами и самой частой причиной."""
    refused = rows[rows["исход"].eq("отказ")].sort_values("ts")
    out: list[dict] = []
    for _, row in refused.iterrows():
        if out and (row["ts"] - out[-1]["по"]).total_seconds() / 3600 <= EPISODE_GAP_H:
            out[-1]["по"] = row["ts"]
            out[-1]["причины"].append(row["причина отказа"])
        else:
            out.append({"с": row["ts"], "по": row["ts"], "причины": [row["причина отказа"]]})
    for block in out:
        reasons = pd.Series(block.pop("причины")).fillna("—")
        block["причина"] = reasons.value_counts().index[0]
        block["часов"] = (block["по"] - block["с"]).total_seconds() / 3600
    return out


def data_periods() -> list[dict]:
    path = ROOT / "reports" / "data_periods.json"
    if not path.exists():
        return []
    return [{**p, "с": pd.Timestamp(p["с"]), "по": pd.Timestamp(p["по"])}
            for p in json.loads(path.read_text(encoding="utf-8"))["периоды"]]


def overlaps(block: dict, periods: list[dict]) -> str:
    kinds = sorted({p["что"] for p in periods
                    if p["с"] <= block["по"] and p["по"] >= block["с"]})
    return "; ".join(kinds) if kinds else "—"


def pick(rows: pd.DataFrame) -> dict[str, pd.Timestamp | None]:
    """Показательные моменты: действие с наибольшим риском, удержание у порога, отказ."""
    acts = rows[rows["исход"].eq("меняем уставки")].sort_values("риск", ascending=False)
    holds = rows[rows["исход"].eq("держим режим")].sort_values("риск", ascending=False)
    refusals = rows[rows["исход"].eq("отказ") & rows["причина отказа"].eq("недостоверные данные")]
    if refusals.empty:
        refusals = rows[rows["исход"].eq("отказ")]
    return {"действие": acts["ts"].iloc[0] if len(acts) else None,
            "удержание": holds["ts"].iloc[0] if len(holds) else None,
            "отказ": refusals["ts"].iloc[0] if len(refusals) else None}


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", default=str(DEFAULT_RUN), help="отчёт прогона")
    ap.add_argument("--cards", action="store_true", help="напечатать три карточки целиком")
    ap.add_argument("--out", default=None, help="сохранить итог в Markdown")
    args = ap.parse_args()

    path = Path(args.run)
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    rows = pd.DataFrame(data["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    total = len(rows)
    outcomes = rows["исход"].value_counts()
    check = (summary.get("сверка с лабораторией") or {}).get("24ч", {})

    lines = [f"# Итог прогона: {summary['период'][0]} … {summary['период'][1]}", "",
             f"Шаг {summary.get('шаг', '?')}, моментов решения {total}. "
             f"Отчёт: `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}`.", "",
             "## Что система решала", "", "| Исход | Моментов | Доля |", "|---|---|---|"]
    for outcome in ("меняем уставки", "держим режим", "отказ"):
        count = int(outcomes.get(outcome, 0))
        lines.append(f"| {outcome} | {count} | {count / max(total, 1):.1%} |")
    reasons = rows.loc[rows["исход"].eq("отказ"), "причина отказа"].value_counts()
    if len(reasons):
        lines += ["", "Причины отказов: " + "; ".join(
            f"{reason} — {count}" for reason, count in reasons.items()) + "."]

    if check:
        lines += ["", "## Сверка с лабораторией: превышение в ближайшие 24 ч", "",
                  f"* моментов с известным фактом: {check['моментов с фактом']};",
                  f"* из них перед превышением: {check['из них с превышением']};",
                  f"* система отреагировала: {check['система отреагировала']}, "
                  f"пропустила: {check['пропущено (держали режим)']} "
                  f"(доля пропусков {check['доля пропусков']:.0%});",
                  f"* зря встревожилась: {check['ложных тревог']} раз "
                  f"(доля {check['доля ложных тревог']:.0%} спокойных моментов).",
                  "",
                  "Пропуски высоки не случайно: система реагирует на уже идущее "
                  "превышение и за сутки вперёд его не предсказывает — это названо в "
                  "README, раздел «Ограничения»."]

    blocks = [b for b in episodes(rows) if b["часов"] >= EPISODE_GAP_H]
    periods = data_periods()
    lines += ["", f"## Где система молчала: эпизоды отказов от {EPISODE_GAP_H:g} ч", "",
              "| С | По | Часов | Причина | Совпадает с неисправными данными |",
              "|---|---|---|---|---|"]
    for block in sorted(blocks, key=lambda b: -b["часов"])[:12]:
        lines.append(f"| {block['с']:%Y-%m-%d %H:%M} | {block['по']:%Y-%m-%d %H:%M} | "
                     f"{block['часов']:.0f} | {block['причина']} | {overlaps(block, periods)} |")
    if not periods:
        lines.append("")
        lines.append("Список неисправных периодов не собран: `python scripts/list_data_periods.py`.")

    chosen = pick(rows)
    lines += ["", "## Показательные моменты", "", "| Что | Момент | Карточка целиком |",
              "|---|---|---|"]
    for name, ts in chosen.items():
        if ts is not None:
            lines.append(f"| {name} | {ts:%Y-%m-%d %H:%M} | "
                         f"`python scripts/run_cycle.py --ts \"{ts:%Y-%m-%d %H:%M}\"` |")

    text = "\n".join(lines) + "\n"
    print(text)

    if args.cards:
        from nefte.config import load_config
        from nefte.pipeline import StateBuilder
        from scripts.run_cycle import build_system

        cfg = load_config()
        sb = StateBuilder(cfg)
        system = build_system(sb, cfg)
        system.log_runs = False
        for name, ts in chosen.items():
            if ts is None:
                continue
            system._last_action_ts = None          # как на дашборде: с чистого листа
            print(f"\n### {name}\n")
            print(system.run(sb.build(ts)).to_operator_text())

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Сохранено: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
