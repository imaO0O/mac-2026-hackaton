"""Полки в телеметрии: неисправность прибора или честно ровный режим (участник 2).

    python scripts/check_shelves.py

Зачем. Полка Q21 оказалась неисправностью, и её период исключили из пригодности
(`scripts/check_q21_shelf.py`, участник 1). Полки есть и у других тегов — `avt:P51`,
`avt:F63`, `ht:P3`, `avt:F65`, `ht:W4`, `ht:Q20`, — а `ht_P3_std36` и
`ht_Q20_std6/std144` стоят в признаках модели. Значит, вердикт по ним меняет
признаки, и его надо вынести по правилу, а не на глаз.

Правило записано ДО счёта (docs/PLAN.md, коммит a113c1b) и исполняется как есть:

* кандидат — отрезок не короче 24 ч, где размах тега не выходит за допуск; допуск по
  тегу — его собственное разрешение, медиана ненулевых часовых изменений на
  работающей установке. Детектор общий — ``cleaning.flat_mask``;
* останов установки и известные заглушки (307, 313, 240, 252) отсеиваются заранее;
* на каждой полке мерится собственный шум тега (против шума за сутки до и после) и
  активность процесса по опорным тегам `F26`, `T6`, `P13` — каждый в долях своего
  обычного часового изменения;
* вердикт: **неисправность**, если шум тега упал ниже 5 % своего обычного И
  активность процесса не ниже 50 %; **ровный режим**, если активность ниже 20 %;
  между — **неопределённо**, такие полки никуда не идут.

Рядом с вердиктом, ОТДЕЛЬНО от него, сообщаются три пометки, найденные уже после
счёта: положение на шкале (ноль или край), нагрузка установки на полке в долях
обычной и число тегов, вставших в ту же минуту. Порогов правила они не меняют —
см. `scale_flag`, `load_flag` и `docs/SHELVES.md`.

Результат: reports/shelves.json и таблица в консоли.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.data.cleaning import flat_mask  # noqa: E402
from nefte.data.loaders import load_telemetry  # noqa: E402
from nefte.models.regime import FEED, outage_mask  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

MIN_HOURS = 24.0
STUBS = (307.0, 313.0, 240.0, 252.0)
# Чем меряется «установка жила» — по СВОЕЙ установке. В правиле записаны опорные
# теги гидроочистки; для тегов АВТ они мерят не ту установку, и это видно на данных:
# 16.06.2026 стояла АВТ (F7 ≈ 0, T33 50 °C вместо 338), а гидроочистка работала — по
# правилу десяток тегов АВТ попал бы в «неисправность». Поэтому теги АВТ судятся по
# опорным тегам АВТ, и это расширение области правила, а не смена его порогов.
REFERENCE = {"ht": ("F26", "T6", "P13"), "avt": ("F65", "T33", "T55")}
NOISE_DEAD = 0.05                       # шум тега на полке ниже этой доли — прибор молчит
ACTIVE = 0.50                           # процесс жил
QUIET = 0.20                            # процесс тоже стоял
SIDE_HOURS = 24                         # окно «до и после» для собственного шума
LOW_LOAD = 0.70                         # ниже этой доли обычной нагрузки — пуск, а не режим
FEEDS = {"ht": FEED, "avt": "F65"}      # чем мерить нагрузку своей установки
REPORT = ROOT / "reports" / "shelves.json"


def hourly_step(series: pd.Series) -> float:
    """Разрешение тега: медиана ненулевых часовых изменений (правило до счёта)."""
    hourly = series.resample("1h", label="right", closed="right").last()
    diff = hourly.diff().abs().dropna()
    nonzero = diff[diff > 0]
    return float(nonzero.median()) if len(nonzero) else 0.0


def noise(series: pd.Series) -> float:
    """Собственный шум: медиана абсолютных первых разностей."""
    diff = series.diff().abs().dropna()
    return float(diff.median()) if len(diff) else float("nan")


def intervals(mask: pd.Series, samples: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Отрезки, где маска истинна; начало сдвигается на длину окна назад."""
    blocks = (mask != mask.shift()).cumsum()
    out = []
    for _, part in mask.groupby(blocks):
        if not bool(part.iloc[0]):
            continue
        position = mask.index.get_loc(part.index[0])
        start = mask.index[max(position - samples + 1, 0)]
        out.append((start, part.index[-1]))
    return out


def activity(reference: pd.DataFrame, usual: pd.Series, lo, hi) -> float:
    """Во сколько раз процесс двигался на полке относительно обычного."""
    part = reference.loc[lo:hi]
    if len(part) < 6:
        return float("nan")
    ratios = [float(part[tag].diff().abs().median() / usual[tag])
              for tag in reference.columns if usual[tag] > 0]
    return float(np.median(ratios)) if ratios else float("nan")


def scale_flag(series: pd.Series, value: float, tolerance: float) -> str | None:
    """Залипание на нуле или на краю шкалы — отказ, которого критерий шума не видит.

    Найдено после счёта: `ht:P3` 313 ч показывает 0.002 МПа при рабочих 3.66 и
    давлении в реакторе 3.89, а `ht:Q20` стоит на 14 933–15 046 ppm при собственном
    p99 12 057. Шум у обоих на месте — они дрожат, просто не там. Признак идёт
    ОТДЕЛЬНО от вердикта правила, порогов правила не меняет.
    """
    usual = float(series.median())
    if abs(value) <= max(tolerance, abs(usual) * 0.01) and abs(usual) > 10 * tolerance:
        return "на нуле при рабочем уровне {:.4g}".format(usual)
    top, bottom = float(series.quantile(0.99)), float(series.quantile(0.01))
    if value >= top:
        return "на верхней границе шкалы (p99 {:.4g})".format(top)
    if value <= bottom:
        return "на нижней границе шкалы (p01 {:.4g})".format(bottom)
    return None


def load_flag(feed: pd.Series, usual: float, lo, hi) -> float | None:
    """Нагрузка установки на полке в долях обычной.

    Тоже найдено после счёта: 19.04.2024 полка началась у 20 тегов сразу, а
    гидроочистка шла на 160 т/ч вместо 256 и АВТ на 456 вместо 914 — это пуск после
    замены катализатора 17.04.2024, где `F14`, `F17`, `F22` стоят на нуле потому, что
    потока нет. Отсев останова такие часы не ловит (нагрузка выше половины обычной),
    а «неисправность» на них была бы ложной. Признак идёт ОТДЕЛЬНО от вердикта.
    """
    part = feed.loc[lo:hi].dropna()
    if part.empty or not usual > 0:
        return None
    return round(float(part.median()) / usual, 2)


def verdict(noise_ratio: float, process: float) -> str:
    if noise_ratio != noise_ratio or process != process:
        return "нет данных"
    if noise_ratio < NOISE_DEAD and process >= ACTIVE:
        return "неисправность"
    if process < QUIET:
        return "ровный режим"
    return "неопределённо"


def main() -> int:
    use_utf8_console()
    data = {unit: load_telemetry(unit) for unit in ("ht", "avt")}
    ht = data["ht"]
    down = outage_mask(ht[FEED], retrospective=True)
    work = ~down & (ht[FEED] > 0.5 * ht[FEED].median())
    step_h = float(pd.Series(ht.index).diff().median().total_seconds() / 3600)
    samples = int(round(MIN_HOURS / step_h))

    reference, usual = {}, {}
    for unit, tags in REFERENCE.items():
        frame = data[unit][list(tags)].where(work.reindex(data[unit].index, fill_value=False))
        reference[unit] = frame.resample("1h", label="right", closed="right").last()
        usual[unit] = reference[unit].diff().abs().median()

    feeds = {unit: frame[FEEDS[unit]] for unit, frame in data.items()}
    usual_load = {unit: float(series.where(work.reindex(series.index, fill_value=False)).median())
                  for unit, series in feeds.items()}

    rows, per_tag = [], {}
    for unit, frame in data.items():
        for tag in frame.columns:
            series = frame[tag].where(work.reindex(frame.index, fill_value=False))
            series = series.where(~series.round(1).isin(STUBS))
            tolerance = hourly_step(series)
            if not tolerance > 0:
                continue
            mask = flat_mask(series, min_samples=samples, tolerance=tolerance)
            found = intervals(mask, samples)
            shelves = []
            for lo, hi in found:
                hours = (hi - lo).total_seconds() / 3600
                if hours < MIN_HOURS:
                    continue
                inside = noise(series.loc[lo:hi])
                side = pd.concat([series.loc[lo - pd.Timedelta(hours=SIDE_HOURS):lo],
                                  series.loc[hi:hi + pd.Timedelta(hours=SIDE_HOURS)]])
                outside = noise(side)
                good = outside == outside and outside > 0
                ratio = inside / outside if good else float("nan")
                process = activity(reference[unit], usual[unit], lo, hi)
                value = float(series.loc[lo:hi].median())
                shelves.append({"начало": str(lo)[:16], "часов": round(hours, 1),
                                "значение": round(value, 4),
                                "шкала": scale_flag(series.dropna(), value, tolerance),
                                "нагрузка": load_flag(feeds[unit], usual_load[unit], lo, hi),
                                "шум на полке / вне": None if ratio != ratio else round(ratio, 3),
                                "активность процесса": (None if process != process
                                                        else round(process, 2)),
                                "вердикт": verdict(ratio, process)})
            if not shelves:
                continue
            per_tag[f"{unit}:{tag}"] = {"допуск": round(tolerance, 5),
                                        "полок": len(shelves),
                                        "часов всего": round(sum(s["часов"] for s in shelves), 1),
                                        "полки": shelves}
    # Сколько тегов встало в ту же минуту: одновременная полка у многих тегов — общая
    # причина (событие установки или заминка архива), а не отказ отдельного прибора.
    starts = pd.Series([s["начало"] for info in per_tag.values() for s in info["полки"]])
    together = starts.value_counts()
    for tag, info in per_tag.items():
        for shelf in info["полки"]:
            shelf["одновременно тегов"] = int(together[shelf["начало"]])
            rows.append({"тег": tag, **shelf})

    frame = pd.DataFrame(rows)
    print(f"Полок не короче {MIN_HOURS:.0f} ч: {len(frame)} у {len(per_tag)} тегов "
          f"(допуск по тегу — его разрешение)\n")
    if not frame.empty:
        counts = frame.groupby(["тег", "вердикт"])["часов"].agg(["size", "sum"]).reset_index()
        print(counts.rename(columns={"size": "полок", "sum": "часов"}).to_string(index=False))
        print("\nСамые длинные полки:")
        print(frame.sort_values("часов", ascending=False).head(15).to_string(index=False))
        print("\nИтог по вердиктам:")
        print(frame.groupby("вердикт")["часов"].agg(["size", "sum"]).rename(
            columns={"size": "полок", "sum": "часов"}).to_string())
        low = frame[frame["нагрузка"].notna() & (frame["нагрузка"] < LOW_LOAD)]
        many = frame[frame["одновременно тегов"] >= 3]
        print(f"\nПометки: на низкой нагрузке (ниже {LOW_LOAD:.0%} обычной) — {len(low)} полок, "
              f"{low['часов'].sum():.0f} ч, из них с вердиктом «неисправность» "
              f"{(low['вердикт'] == 'неисправность').sum()}; встали одновременно с 3+ тегами — "
              f"{len(many)} полок, {many['часов'].sum():.0f} ч")

    REPORT.write_text(json.dumps({
        "правило": {"минимум часов": MIN_HOURS, "шум на полке ниже": NOISE_DEAD,
                    "активность неисправности": ACTIVE, "активность ровного режима": QUIET,
                    "опорные теги": {k: list(v) for k, v in REFERENCE.items()}},
        "пометки после счёта (вердикт не меняют)": {
            "низкая нагрузка ниже": LOW_LOAD,
            "нагрузка мерится по": FEEDS,
            "обычная нагрузка": {k: round(v, 1) for k, v in usual_load.items()}},
        "обычное часовое изменение опорных": {unit: {k: round(float(v), 4) for k, v in row.items()}
                                              for unit, row in usual.items()},
        "теги": per_tag}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
