"""Отказ — это решение. Проверяем, в каких местах система молчит. Только CPU.

    python scripts/check_refusal_quality.py

Зачем. Система отказывается решать в четверти моментов теста, и на защите это
первое, за что зацепятся: «значит, она не работает». Ответ зависит от того, ГДЕ
она молчит. Молчать там, где испортились данные, — ровно то, ради чего отказ
задуман. Молчать там, где процесс уходит за предел, — провал: система отворачивается
именно тогда, когда нужна.

Считаем по уже посчитанному прогону, разбивая отказы по причинам:

* **установка остановлена** — рекомендаций быть не должно (ответ эксперта 11.09,
  32:49: если система в эти моменты молчит, «это не страшно»);
* **данные недостоверны** — отказ по недоверию к измерению;
* **нет допустимых вариантов** — жёсткие ограничения не проходит ни один вариант.

Для каждой группы: доля моментов, доля превышений в лаборатории за 24 ч и ошибка
оперативного значения против лаборатории. Последнее и отличает «молчим, потому что
не знаем» от «молчим, потому что не хотим знать».

**Правило записано ДО счёта.** Отказы признаются здоровыми, если на ТЕСТЕ:

1. ошибка оперативного значения в отказах по недостоверности данных ВЫШЕ, чем в
   моментах с решением, — то есть система молчит там, где приборы врут;
2. доля превышений в отказах не выше, чем в моментах с решением, плюс 5 п.п. —
   иначе система систематически отворачивается от проблем.
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

REPORT = ROOT / "reports" / "refusal_quality.json"
RUNS = {"валидация": "val_period_step1h_lock4.json",
        "тест": "test_period_step1h.json"}
HOURS = 24.0
TOLERANCE = 0.05

REASONS = {
    "установка остановлена": "остановлена",
    "данные недостоверны": "недостоверн",
    "нет допустимых вариантов": "допустимых",
}


def asof(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    series = series.dropna().sort_index()
    pos = series.index.searchsorted(index, side="right") - 1
    return np.where(pos >= 0, series.to_numpy()[pos.clip(min=0)], np.nan)


def nearest_lab(lab: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Ближайший СЛЕДУЮЩИЙ анализ: с ним и сравнивается оперативное значение."""
    times = lab.index.to_numpy()
    pos = np.searchsorted(times, index.to_numpy(), side="right")
    ok = pos < len(times)
    return np.where(ok, lab.to_numpy()[np.clip(pos, 0, len(times) - 1)], np.nan)


def block(path: Path, sb: StateBuilder) -> dict:
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    index = pd.DatetimeIndex(rows["ts"])
    lab = sb.lims_sulfur.dropna().sort_index()
    truth = nearest_lab(lab, index)
    operational = asof(sb.q21_sulfur, index)
    error = np.abs(operational - truth)
    column = f"превышение за {HOURS:.0f} ч"
    known = rows[column].notna().to_numpy()
    over = np.where(known, rows[column].to_numpy() == True, False)  # noqa: E712
    reason = rows["причина отказа"].fillna("").astype(str)
    refused = rows["исход"].eq("отказ").to_numpy()

    def describe(mask: np.ndarray) -> dict:
        if mask.sum() < 5:
            return {}
        with_fact = mask & known
        with_error = mask & ~np.isnan(error)
        return {
            "моментов": int(mask.sum()),
            "доля моментов": round(float(mask.mean()), 3),
            "превышений за 24 ч": (round(float(over[with_fact].mean()), 3)
                                   if with_fact.any() else None),
            "ошибка оперативного значения, мг/кг": (
                round(float(error[with_error].mean()), 2) if with_error.any() else None),
        }

    out = {"всего моментов": int(len(rows)), "с решением": describe(~refused)}
    for name, needle in REASONS.items():
        out[f"отказ: {name}"] = describe(
            refused & reason.str.contains(needle, regex=False).to_numpy())
    return out


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)

    out = {}
    for name, filename in RUNS.items():
        path = ROOT / "reports" / filename
        if not path.exists():
            print(f"нет прогона {filename}")
            continue
        out[name] = block(path, sb)
        print(f"\n{name}:")
        for key, value in out[name].items():
            print(f"  {key}: {value}")

    test = out.get("тест", {})
    decided = test.get("с решением") or {}
    data_refusal = test.get("отказ: данные недостоверны") or {}
    rule = {
        "1. в отказах по данным оперативное значение врёт сильнее": bool(
            (data_refusal.get("ошибка оперативного значения, мг/кг") or 0)
            > (decided.get("ошибка оперативного значения, мг/кг") or 0)),
        "2. отказы не прячут превышения": bool(
            (data_refusal.get("превышений за 24 ч") or 0)
            <= (decided.get("превышений за 24 ч") or 0) + TOLERANCE),
    }
    accepted = all(rule.values())
    print("\nПравило приёмки (тест):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: отказы {'здоровые' if accepted else 'НЕЗДОРОВЫЕ — разбираться'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "окно факта, ч": HOURS,
                                  "выборки": out, "правило": rule,
                                  "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
