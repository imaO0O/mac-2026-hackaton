"""«Злые» кейсы: система обязана отказываться, а не фантазировать.

    python scripts/check_adversarial.py
    python scripts/check_adversarial.py --ts "2026-06-15 12:00"

Берём реальный срез состояния и портим его ровно так, как данные портятся на
установке: молчит поточный анализатор, зависает на полке, устаревает лаборатория,
в ключевые теги приходят заглушки 307. Для каждого случая ожидание записано
ЗАРАНЕЕ — скрипт его проверяет и возвращает ненулевой код, если система повела
себя не так. Это не демонстрация, а проверка.

Три случая берутся не из порчи, а из истории: реальный останов установки, самый
резкий скачок расхода сырья в тестовом периоде и реальный выброс в лаборатории.
Их портить не нужно — данные уже злые. Скачок сырья подменой одного значения в
срезе не проверить: скорость изменения режима считается по ряду, и в срезе её
просто нет.

Результат: таблица в консоли и reports/adversarial.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.agents.schemas import ProcessState, Recommendation  # noqa: E402
from nefte.provenance import reliability_provenance, report_provenance  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.models.regime import FEED  # noqa: E402
from nefte.pipeline import StateBuilder, is_state_usable  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

REPORT = ROOT / "reports" / "adversarial.json"

# Теги реакторного блока: именно по ним оптимизатор двигает режим. Если в них
# пришла заглушка, рекомендовать по ним что-либо нельзя.
KEY_TAGS = ["T5", "T6", "T11"]
# оба поточных анализатора серы: ряд из файла ПАК и тег Q21 телеметрии
ANALYZER_KEYS = ("pak_sulfur_ppm", "q21_sulfur_ppm")


# --------------------------------------------------------------------------- #
# порча среза
# --------------------------------------------------------------------------- #

def damage(state: ProcessState, cfg: dict, *, drop_pak: bool = False,
           freeze_pak: bool = False, drop_lims: bool = False,
           lims_age_hours: float | None = None,
           blank_ht: tuple[str, ...] = ()) -> ProcessState:
    """Копия среза с внесённой неисправностью.

    Пригодность пересчитывается тем же правилом, что и в рабочем цикле
    (``pipeline.is_state_usable``), а не повторяется здесь второй раз: иначе
    «злой» сценарий проверял бы не ту систему, которая пойдёт в демо.
    """
    s = state.model_copy(deep=True)

    # «ПАК» в сценариях — это поточный анализатор вообще, а их два: ряд из файла и
    # тег Q21 телеметрии, и оперативным выбран Q21 (`quality.analyzer_source`).
    # Раньше ломался только ряд из файла: сценарий «нет ни ЛИМС, ни ПАК» не
    # выполнялся, потому что система честно работала по живому Q21, а три других
    # сценария с ПАК проходили вхолостую — ломали не тот прибор, которым она
    # пользуется. Поломка применяется к обоим.
    for key in ANALYZER_KEYS:
        if drop_pak:
            s.quality.pop(key, None)
        analyzer = s.quality.get(key)
        if freeze_pak and analyzer is not None:
            analyzer.is_frozen = True
            analyzer.comment = "сигнал не менялся дольше порога"

    if drop_lims:
        s.quality.pop("lims_sulfur_mgkg", None)
    lims = s.quality.get("lims_sulfur_mgkg")
    if lims_age_hours is not None and lims is not None:
        lims.age_hours = lims_age_hours
        lims.is_stale = lims_age_hours > cfg["quality"]["staleness_hours"]["lims"]

    for tag in blank_ht:
        if tag in s.telemetry_ht:
            s.telemetry_ht[tag] = None       # заглушка 307 вычищается как пропуск

    total = len(s.telemetry_avt) + len(s.telemetry_ht)
    missing = sum(v is None for v in s.telemetry_avt.values()) + \
        sum(v is None for v in s.telemetry_ht.values())
    s.data_quality.missing_share = missing / total if total else 1.0
    if blank_ht:
        s.data_quality.sentinel_tags = sorted(set(s.data_quality.sentinel_tags)
                                              | {f"ht:{t}" for t in blank_ht})
        s.data_quality.notes.append(
            f"Значения-заглушки в тегах: 24-2000: {', '.join(sorted(blank_ht))}.")
    if freeze_pak:
        s.data_quality.frozen_tags = sorted(set(s.data_quality.frozen_tags) | {"pak_sulfur"})
        s.data_quality.notes.append("Поточный анализатор серы заморожен.")
    s.data_quality.stale_sources = [k for k, m in s.quality.items() if m.is_stale]
    # правило пригодности — то же, что в рабочем конвейере, а он смотрит на
    # зависание РЯДА ИЗ ФАЙЛА ПАК (`pipeline.StateBuilder.build`), даже когда
    # оперативным выбран Q21; расхождение разобрано в docs/PLAN.md
    pak = s.quality.get("pak_sulfur_ppm")
    s.data_quality.usable = is_state_usable(
        s.data_quality.missing_share,
        bool(pak is not None and pak.is_frozen),
        lims.age_hours if lims is not None else None, cfg)
    return s


# --------------------------------------------------------------------------- #
# ожидания
# --------------------------------------------------------------------------- #

def used_source(rec: Recommendation) -> str:
    return str(rec.state_summary.get("sulfur_source", "нет"))


def abstained(rec: Recommendation, keyword: str = "") -> bool:
    return rec.abstained and (keyword.lower() in rec.abstain_reason.lower() if keyword else True)


def touches(rec: Recommendation, tags: list[str]) -> list[str]:
    """Какие из тегов система реально двигает в рекомендации."""
    if rec.action is None:
        return []
    return sorted(t for t in tags if abs(rec.action.deltas.get(t, 0.0)) > 1e-6)


def feed_jump(sb: StateBuilder, cfg: dict) -> tuple[pd.Timestamp | None, float]:
    """Момент самого резкого часового изменения расхода сырья в тестовом периоде."""
    lo, hi = cfg["split"]["test"]
    if FEED not in sb.ht.columns:
        return None, 0.0
    step = pd.Series(sb.ht.index).diff().median()
    per_hour = max(int(pd.Timedelta("1h") / step), 1) if step else 6
    delta = sb.ht[FEED].loc[lo:hi].diff(per_hour).abs().dropna()
    if delta.empty:
        return None, 0.0
    return delta.idxmax(), float(delta.max())


# --------------------------------------------------------------------------- #

def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="2026-06-15 12:00",
                    help="опорный момент, срез которого портим")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    system = build_system(sb, cfg)
    system.log_runs = False          # «злые» кейсы не засоряют журнал прогонов

    base_ts = pd.Timestamp(args.ts)
    base = sb.build(base_ts)
    limit_lims = cfg["quality"]["lims_sulfur_outlier_above"]

    def run(state: ProcessState) -> Recommendation:
        # лимит частоты воздействий — состояние оркестратора; между независимыми
        # сценариями его надо сбрасывать, иначе второй кейс проверяет первый
        system._last_action_ts = None
        return system.run(state)

    cases: list[dict] = []

    def case(name: str, why: str, state: ProcessState, expect: str,
             ok_fn) -> None:
        rec = run(state)
        ok = bool(ok_fn(rec))
        cases.append({
            "сценарий": name,
            "что проверяем": why,
            "ожидание": expect,
            "исход": ("отказ: " + rec.abstain_reason[:120]) if rec.abstained else
                     ("действие: " + (", ".join(
                         f"{t} {d:+.2f}" for t, d in rec.action.deltas.items()
                         if abs(d) > 1e-6) or "держим режим") if rec.action else "нет действия"),
            "источник качества": used_source(rec),
            "выполнено": ok,
        })

    # 1. Поточный анализатор молчит, лаборатория свежая ------------------- #
    case("обрыв ПАК", "приборный сигнал пропал, но лабораторный анализ свежий",
         damage(base, cfg, drop_pak=True, lims_age_hours=3.0),
         "работает на ЛИМС, отказ не нужен",
         lambda rec: not rec.abstained and used_source(rec) == "lims")

    # 2. Анализатор завис на полке, лаборатория свежая -------------------- #
    case("завис ПАК + свежий ЛИМС", "случай 15–27.04.2026: прибор врёт, лаборатория в норме",
         damage(base, cfg, freeze_pak=True, lims_age_hours=3.0),
         "зависшее значение не считается фактом, решение по ЛИМС",
         lambda rec: not rec.abstained and used_source(rec) == "lims")

    # 3. Анализатор завис, лаборатория устарела --------------------------- #
    case("завис ПАК + ЛИМС 96 ч", "оба источника качества недостоверны",
         damage(base, cfg, freeze_pak=True, lims_age_hours=96.0),
         "отказ от рекомендации",
         lambda rec: abstained(rec))

    # 4. Источников качества нет вообще ----------------------------------- #
    # Без обученной модели прогнозировать нечем — только отказ. С моделью это
    # третий приоритет по ТЗ, виртуальный анализатор: работать можно, но система
    # обязана назвать источник и снизить уверенность, а не молча выдать решение
    # «как обычно». Раньше она показывала источником ЛИМС — от серы СЫРЬЯ.
    baseline_confidence = run(base).confidence
    case("нет ни ЛИМС, ни ПАК", "качество продукта известно только модели",
         damage(base, cfg, drop_pak=True, drop_lims=True),
         "отказ либо честно помеченный ВАК с пониженной уверенностью",
         lambda rec: abstained(rec) or (used_source(rec) == "vak"
                                        and rec.confidence < baseline_confidence))

    # 5. Заглушки 307 в ключевых тегах ------------------------------------ #
    case("заглушки 307 в реакторных тегах", "уставки, которыми управляем, не видны",
         damage(base, cfg, blank_ht=tuple(KEY_TAGS)),
         "система не даёт рекомендаций по тегам, значений которых не видит",
         lambda rec: rec.abstained or not touches(rec, KEY_TAGS))

    # 6. Скачок расхода сырья --------------------------------------------- #
    # Момент берём из истории, а не выдумываем: скорость изменения режима
    # считается по ряду, и подмена одного значения в срезе её не сдвинет —
    # проверять пришлось бы не ту величину, которая работает в цикле.
    jump_ts, jump_value = feed_jump(sb, cfg)
    if jump_ts is not None:
        before = system.reliability.assess(sb.build(jump_ts - pd.Timedelta("12h")))
        after = system.reliability.assess(sb.build(jump_ts))
        case("реальный скачок сырья",
             f"момент из данных: {jump_ts:%Y-%m-%d %H:%M}, расход за час изменился "
             f"на {jump_value:.1f} ед.",
             sb.build(jump_ts),
             "тяжесть режима растёт, подъём температур в переходном режиме не предлагается",
             lambda rec: (after.severity_index >= before.severity_index
                          and (rec.abstained or not any(
                              rec.action.deltas.get(t, 0.0) > 1e-6 for t in KEY_TAGS))))
        cases[-1]["severity спокойно/скачок"] = [round(before.severity_index, 3),
                                                 round(after.severity_index, 3)]
    else:
        cases.append({"сценарий": "реальный скачок сырья",
                      "что проверяем": "в тестовом периоде скачков не найдено",
                      "ожидание": "—", "исход": "нет данных", "выполнено": True})

    # 7. Реальный останов установки --------------------------------------- #
    down = system.reliability.down_series
    down_ts = None
    if down is not None:
        test_lo = cfg["split"]["test"][0]
        stopped = down.loc[test_lo:]
        stopped = stopped[stopped]
        down_ts = stopped.index[0] if len(stopped) else None
    if down_ts is not None:
        case("реальный останов установки", f"момент из данных: {down_ts:%Y-%m-%d %H:%M}",
             sb.build(down_ts),
             "отказ с причиной «установка остановлена», а не советом греть реактор",
             lambda rec: abstained(rec, "остановлена"))
    else:
        cases.append({"сценарий": "реальный останов установки",
                      "что проверяем": "в тестовом периоде остановов не найдено",
                      "ожидание": "—", "исход": "нет данных", "выполнено": True})

    # 8. Реальный выброс в лаборатории ------------------------------------ #
    lo, hi = cfg["demo_windows"]["bad_data_lims_outlier"]
    raw = sb.lims_sulfur.loc[lo:hi]
    outlier_ts = pd.Timestamp(hi)
    state = sb.build(outlier_ts)
    used = state.quality.get("lims_sulfur_mgkg")
    rec = run(state)
    cases.append({
        "сценарий": "выброс в лаборатории",
        "что проверяем": f"окно {lo}…{hi}: в сырых данных есть значение выше "
                         f"{limit_lims:.0f} мг/кг",
        "ожидание": "выброс в карантине, в решение попадает достоверное значение",
        "исход": (f"использовано {used.value:.2f} мг/кг" if used and used.value is not None
                  else "значения нет"),
        "источник качества": used_source(rec),
        "выполнено": bool(used is None or used.value is None or used.value <= limit_lims),
        "макс. в окне после очистки": None if raw.empty else round(float(raw.max()), 2),
    })

    # ------------------------------------------------------------------ #
    frame = pd.DataFrame(cases)
    columns = [c for c in ["сценарий", "ожидание", "исход", "источник качества", "выполнено"]
               if c in frame.columns]
    print(f"\n«Злые» кейсы на срезе {base_ts:%Y-%m-%d %H:%M}\n")
    print(frame[columns].to_string(index=False))

    failed = [c["сценарий"] for c in cases if not c["выполнено"]]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {**report_provenance(), **reliability_provenance(),
         "ts": str(base_ts), "cases": cases, "failed": failed},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")

    if failed:
        print(f"\nНЕ ВЫПОЛНЕНО: {', '.join(failed)}")
        return 1
    print(f"\nВсе {len(cases)} сценариев отработаны как ожидалось.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
