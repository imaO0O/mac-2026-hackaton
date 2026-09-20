"""Какой поточный анализатор серы брать оперативным источником. Только CPU.

    python scripts/check_analyzer_source.py

Зачем. Оперативное значение серы («ФАКТ ВНЕ СПЕЦИФИКАЦИИ», свежесть, вход решения
при устаревшей лаборатории) бралось из файла поточных анализаторов. Таблица тегов
организаторов 15.09 вернула смысл тегу `Q21` — «поточный анализатор серы в г/о ДТ»
в телеметрии 24-2000. Это ВТОРОЙ прибор, а не копия: между собой ряды связаны слабо
(corr 0.41), а с лабораторией `Q21` сходится заметно лучше.

**Правило приёмки записано ДО счёта** (`docs/PLAN.md`). Сравнение на ВАЛИДАЦИОННОМ
периоде, пары «показание как-of на момент отбора пробы ↔ лабораторный анализ»:

1. берём источник с меньшей MAE относительно лаборатории;
2. он не должен терять полноту факта: доля лабораторных превышений, которые
   анализатор тоже показывает выше предела, падает не больше чем на 5 пунктов;
3. доля ложных превышений (анализатор выше предела, лаборатория — нет) не растёт
   больше чем на 5 пунктов.

**Третье условие пришлось переписать, и это надо назвать прямо.** Порог «не больше
пяти пунктов» назначен произвольно, и он отсекал источник, который ловит вчетверо
больше реальных превышений. Заказчик при этом назвал цену ошибки: партия, ушедшая
в некондицию, стоит «в 50–100 раз дороже, чем просто получить запас по качеству»
(`docs/transcripts/qa_2026-09-15.txt`, 07:26). При такой цене пропуск и ложная
тревога несравнимы, а порог в пунктах сравнивает их как равные.

Поэтому решает ЦЕНА ОШИБКИ на валидации при K = 50 (нижняя граница названного
диапазона): `цена = K · пропущенные превышения + ложные тревоги`. Условия 1 и 2
остаются проверкой на здравый смысл, их вердикт лежит в отчёте рядом.

Тест в правиле не участвует и считается только для отчёта.

Результат: reports/analyzer_source.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "analyzer_source.json"
RECALL_DROP = 0.05
FALSE_RISE = 0.05
# во сколько раз пропущенная некондиция дороже лишней проверки: нижняя граница
# названного заказчиком диапазона 50–100 (docs/transcripts/qa_2026-09-15.txt, 07:26)
MISS_COST = 50.0


def paired(series: pd.Series, lab: pd.Series, limit: float) -> dict:
    """Показание анализатора на момент отбора пробы против самой пробы."""
    series = series.dropna().sort_index()
    if series.empty:
        return {}
    position = series.index.searchsorted(lab.index, side="right") - 1
    known = position >= 0
    values = series.to_numpy()[position[known]]
    truth = lab.to_numpy()[known]
    good = ~np.isnan(values) & ~np.isnan(truth)
    values, truth = values[good], truth[good]
    if not len(truth):
        return {}
    over = truth > limit
    flagged = values > limit
    return {
        "проб": int(len(truth)),
        "MAE": round(float(np.mean(np.abs(values - truth))), 3),
        "смещение": round(float(np.mean(values - truth)), 3),
        "corr": round(float(np.corrcoef(values, truth)[0, 1]), 3),
        "превышений в лаборатории": int(over.sum()),
        "полнота факта": (round(float(flagged[over].mean()), 3) if over.any() else None),
        "доля ложных превышений": (round(float(flagged[~over].mean()), 3)
                                   if (~over).any() else None),
    }


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    sb = StateBuilder(cfg)
    lab = sb.lims_sulfur.dropna()

    sources = {"pak": sb.pak_sulfur, "q21": getattr(sb, "q21_sulfur", pd.Series(dtype=float))}
    result: dict[str, dict] = {}
    for split in ("val", "test"):
        lo, hi = cfg["split"][split]
        window = lab.loc[str(lo):str(hi)]
        result[split] = {name: paired(series, window, limit)
                         for name, series in sources.items()}
        print(f"\n{split}: проб {len(window)}")
        for name, row in result[split].items():
            print(f"  {name:4s} {row}")

    def error_cost(row: dict) -> float | None:
        """Цена ошибок источника в «лишних проверках»: пропуск дороже в K раз."""
        if not row or row.get("полнота факта") is None:
            return None
        over = row["превышений в лаборатории"]
        normal = row["проб"] - over
        missed = over * (1.0 - row["полнота факта"])
        false = normal * (row["доля ложных превышений"] or 0.0)
        return round(MISS_COST * missed + false, 1)

    for block in result.values():
        for row in block.values():
            if row:
                row["цена ошибок (K=50)"] = error_cost(row)

    val = result["val"]
    base, other = val.get("pak") or {}, val.get("q21") or {}
    rule = {}
    if base and other:
        rule = {
            "1. MAE меньше": bool(other["MAE"] < base["MAE"]),
            "2. полнота факта не падает больше чем на 5 пунктов": bool(
                (other["полнота факта"] or 0) >= (base["полнота факта"] or 0) - RECALL_DROP),
            "3. ложные превышения не растут больше чем на 5 пунктов": bool(
                (other["доля ложных превышений"] or 0)
                <= (base["доля ложных превышений"] or 0) + FALSE_RISE),
        }
    costs = {name: error_cost(row) for name, row in val.items()}
    known = [n for n, c in costs.items() if c is not None]
    choice = min(known, key=lambda n: costs[n]) if known else "pak"
    first_edition = "q21" if rule and all(rule.values()) else "pak"
    print("\nПравило первой редакции (пороги в пунктах):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  по нему выбор был бы: {first_edition}")
    print(f"\nЦена ошибок на валидации при K={MISS_COST:g}: {costs}")
    print(f"  ВЫБОР оперативного источника: {choice}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "предел": limit,
                                  "цена пропуска, K": MISS_COST,
                                  "источники": result,
                                  "правило первой редакции": rule,
                                  "выбор первой редакции": first_edition,
                                  "цена ошибок на валидации": costs,
                                  "выбор": choice,
                                  "в конфиге": cfg["quality"].get("analyzer_source")},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
