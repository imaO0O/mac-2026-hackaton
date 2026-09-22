"""Периоды, где данные неисправны: с какого по какое, почему и что делает система.

    python scripts/list_data_periods.py

Зачем. Эксперт на сессии 18.09 (`docs/transcripts/qa_2026-09-18.txt`, 08:55–09:45):
периоды, «когда поточник поломался, показывает неверные данные», из расчётов
исключать можно, «но даёте приписку, что исключены данные с такого-то по такое-то,
по таким-то причинам».

Мы такие периоды НЕ вырезаем: система в них работает и отказывается, называя
причину, а все метрики посчитаны с ними вместе. Но список с датами обязан быть —
чтобы проверяющий видел, где система молчит и почему, и мог сам их исключить.

Периоды находят те же детекторы, которыми пользуется система, а не ручная разметка:
* останов установки — расход сырья ниже 10 % медианы и реактор холоднее 150 °C
  (`ReliabilityAgent.is_unit_down`);
* полка оперативного анализатора Q21 — детектор с допуском (`cleaning.flat_mask`,
  `quality.q21_frozen_tolerance_mgkg`);
* полка ряда ПАК из файла — точное равенство (`cleaning.frozen_mask`);
* лаборатория молчит дольше трёх нормативных сроков (72 ч) — порог правила
  пригодности среза.

Результат: `docs/DATA_PERIODS.md` и `reports/data_periods.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.loaders import load_telemetry  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "data_periods.json"
DOC = ROOT / "docs" / "DATA_PERIODS.md"
MIN_HOURS = 6.0

WHAT_SYSTEM_DOES = {
    "останов установки": "отказ «установка не в работе»; совета по уставкам нет",
    "полка Q21 (оперативный анализатор)": (
        "прибор не используется как факт; решение по свежей лаборатории, при "
        "лаборатории старше 72 ч — отказ «данные недостоверны»"),
    "полка ПАК из файла": "второй прибор; примечание в карточке, в решении не участвует",
    "лаборатория молчит больше 72 ч": (
        "уверенность падает; при неисправном оперативном приборе — отказ"),
}


def intervals(mask: pd.Series, kind: str) -> list[dict]:
    """Непрерывные отрезки True не короче MIN_HOURS."""
    mask = mask.fillna(False).astype(bool)
    if not mask.any():
        return []
    group = (mask != mask.shift()).cumsum()
    idx = mask.index.to_series()
    out = []
    for _, block in idx[mask].groupby(group[mask]):
        start, end = block.iloc[0], block.iloc[-1]
        hours = (end - start).total_seconds() / 3600.0
        if hours >= MIN_HOURS:
            out.append({"что": kind, "с": start, "по": end, "часов": round(hours, 1)})
    return out


def lab_gaps(lab: pd.Series, hours: float) -> list[dict]:
    times = lab.dropna().sort_index().index
    out = []
    for before, after in zip(times[:-1], times[1:]):
        gap = (after - before).total_seconds() / 3600.0
        if gap > hours:
            out.append({"что": "лаборатория молчит больше 72 ч", "с": before, "по": after,
                        "часов": round(gap, 1)})
    return out


SPLITS = ("обучение", "валидация", "тест")


def split_of(ts: pd.Timestamp, cfg: dict) -> str:
    for name in ("train", "val", "test"):
        lo, hi = cfg["split"][name]
        if pd.Timestamp(lo) <= ts <= pd.Timestamp(hi) + pd.Timedelta(days=1):
            return {"train": "обучение", "val": "валидация", "test": "тест"}[name]
    return "вне разбиения"


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)

    # Останов — по СЫРОЙ телеметрии, как у агента надёжности: очистка вырезает
    # застывший на нуле расход сырья вместе с самим фактом останова, и по чистому
    # ряду тест выглядел бы вовсе без остановов.
    raw = load_telemetry("ht")
    hourly = raw.resample("1h").mean()
    down = ((hourly["F26"] < raw["F26"].median() * 0.1)
            & (hourly[["T5", "T6", "T11"]].mean(axis=1) < 150.0))
    rows = intervals(down, "останов установки")

    def shelf(mask: pd.Series) -> pd.Series:
        # полку рвут короткие пропуски данных — склеиваем через них, до 6 ч
        return (mask.astype(float).resample("1h").max().ffill(limit=6) > 0)

    rows += intervals(shelf(sb.q21_frozen), "полка Q21 (оперативный анализатор)")
    rows += intervals(shelf(sb.pak_frozen), "полка ПАК из файла")
    stale = float(cfg["quality"]["staleness_hours"]["lims"]) * 3
    rows += lab_gaps(sb.lims_sulfur, stale)
    rows.sort(key=lambda row: row["с"])
    for row in rows:
        row["выборка"] = split_of(row["с"], cfg)
        row["что делает система"] = WHAT_SYSTEM_DOES[row["что"]]

    summary: dict = {}
    for row in rows:
        key = (row["выборка"], row["что"])
        summary[key] = summary.get(key, 0.0) + row["часов"]
    print(f"периодов не короче {MIN_HOURS:g} ч: {len(rows)}")
    for (split, kind), hours in sorted(summary.items()):
        print(f"  {split:14s} {kind:38s} {hours:8.0f} ч")

    lines = [
        "# Периоды с неисправными данными",
        "",
        "Эксперт на сессии 18.09 (`docs/transcripts/qa_2026-09-18.txt`, 08:55–09:45):",
        "периоды, «когда поточник поломался, показывает неверные данные», исключать можно,",
        "«но даёте приписку, что исключены данные с такого-то по такое-то, по таким-то",
        "причинам». Мы их **не исключаем**: система в них работает и отказывается,",
        "называя причину, и все метрики посчитаны вместе с ними. Список ниже — чтобы",
        "было видно, где система молчит и почему; при желании любой из периодов можно",
        "исключить самим.",
        "",
        "Периоды находят те же детекторы, которыми пользуется система, а не ручная",
        f"разметка; показаны отрезки не короче {MIN_HOURS:g} ч. Собрано командой",
        "`python scripts/list_data_periods.py`.",
        "",
        "## Сколько часов по выборкам",
        "",
        "| Выборка | Что | Часов |",
        "|---|---|---|",
    ]
    # Выборки — в порядке времени, и пустая выборка названа: без строки читатель
    # решит, что её забыли проверить, а не что в ней чисто.
    for split in SPLITS:
        found = sorted((kind, hours) for (sp, kind), hours in summary.items() if sp == split)
        for kind, hours in found:
            lines.append(f"| {split} | {kind} | {hours:.0f} |")
        if not found:
            lines.append(f"| {split} | неисправных отрезков не короче {MIN_HOURS:g} ч нет | 0 |")
    lines += ["", "## Что делает система в каждом случае", "",
              "| Что | Что делает система |", "|---|---|"]
    for kind, action in WHAT_SYSTEM_DOES.items():
        lines.append(f"| {kind} | {action} |")
    long_rows = [row for row in rows if row["часов"] >= 24.0]
    lines += ["", f"## Отрезки не короче суток ({len(long_rows)})", "",
              "| С | По | Часов | Что | Выборка |", "|---|---|---|---|---|"]
    for row in long_rows:
        lines.append(f"| {row['с']:%Y-%m-%d %H:%M} | {row['по']:%Y-%m-%d %H:%M} | "
                     f"{row['часов']:.0f} | {row['что']} | {row['выборка']} |")
    lines += ["", "Короче суток — в `reports/data_periods.json`.", ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")

    REPORT.write_text(json.dumps({
        **report_provenance(cfg), "не короче, ч": MIN_HOURS,
        "часов по выборкам": [{"выборка": s, "что": k, "часов": round(h, 1)}
                              for (s, k), h in sorted(summary.items())],
        "периоды": [{**row, "с": str(row["с"]), "по": str(row["по"])} for row in rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nДокумент: {DOC.relative_to(ROOT)}; отчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
