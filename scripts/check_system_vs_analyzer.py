"""Система против простого правила «Q21 выше порога». Самый неудобный вопрос защиты.

    python scripts/check_system_vs_analyzer.py

Зачем. В телеметрии есть поточный анализатор серы (`Q21`), и он неплох: с
лабораторией corr 0.66 на валидации. Значит, на защите спросят прямо: зачем модель
и пять агентов, если можно смотреть на прибор и вмешиваться, когда он показывает
много. Ответ должен быть измеренным, а не риторическим.

Сравниваем НА ОДНОЙ И ТОЙ ЖЕ строгости — доле ложных тревог, — как и модели между
собой (`scripts/compare_decision_curves.py`):

* **система**: риск из отчёта прогона по тесту, порог перебирается;
* **правило по анализатору**: тревога, когда показание `Q21` на момент решения выше
  порога; порог перебирается по тем же моментам.

Мера одна и та же: доля лабораторных превышений за 24 ч, которые остались без
вмешательства («пропуски»), при заданной доле ложных тревог.

**Правило записано ДО счёта**: система обязана давать меньше пропусков при равной
доле ложных тревог хотя бы на двух уровнях строгости из трёх (≤25 %, ≤30 %, ≤40 %).
Не выполняется — это записывается как измеренное ограничение, и ценность системы на
защите формулируется через то, чего у анализатора нет: прогноз эффекта уставок,
интервал неопределённости, отказ на недостоверных данных и разбор причины.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import reliability_provenance, report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "system_vs_analyzer.json"
RUN = "test_period_step1h.json"
HOURS = 24.0
LEVELS = [0.25, 0.30, 0.40]


def curve(flags_by_threshold: dict[float, np.ndarray], over: np.ndarray) -> list[dict]:
    """Доля пропусков и доля ложных тревог по каждому порогу."""
    out = []
    n_over, n_calm = max(int(over.sum()), 1), max(int((~over).sum()), 1)
    for threshold, acts in sorted(flags_by_threshold.items()):
        missed = int((over & ~acts).sum())
        false = int((~over & acts).sum())
        out.append({"порог": round(float(threshold), 4),
                    "доля пропусков": round(missed / n_over, 3),
                    "доля ложных тревог": round(false / n_calm, 3)})
    return out


def misses_at(curve_rows: list[dict], level: float) -> float | None:
    """Наименьшая доля пропусков среди порогов, укладывающихся в долю ложных."""
    affordable = [r["доля пропусков"] for r in curve_rows
                  if r["доля ложных тревог"] <= level + 1e-9]
    return min(affordable) if affordable else None


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    data = json.loads((ROOT / "reports" / RUN).read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    column = f"превышение за {HOURS:.0f} ч"
    known = rows[rows[column].notna() & rows["риск"].notna()].set_index("ts")
    over = known[column].astype(bool).to_numpy()
    print(f"{RUN}: моментов {len(known)}, из них перед превышением {int(over.sum())}")

    sb = StateBuilder(cfg)
    analyzer = sb.q21_sulfur.dropna().sort_index()
    position = analyzer.index.searchsorted(known.index, side="right") - 1
    values = np.where(position >= 0, analyzer.to_numpy()[position.clip(min=0)], np.nan)
    live = ~np.isnan(values)
    print(f"моментов с показанием Q21: {int(live.sum())}")

    risk = known["риск"].to_numpy()
    system_flags = {float(t): (risk >= t) & live
                    for t in np.round(np.linspace(0.02, 0.9, 45), 4)}
    analyzer_flags = {float(t): (values >= t) & live
                      for t in np.round(np.quantile(values[live], np.linspace(0.05, 0.99, 45)), 3)}

    over_live = over & live
    curves = {"система": curve(system_flags, over_live),
              "правило по Q21": curve(analyzer_flags, over_live)}

    compare = {}
    for level in LEVELS:
        compare[f"ложных ≤{level:.0%}"] = {
            name: misses_at(rowset, level) for name, rowset in curves.items()}
        print(f"\nпри доле ложных ≤{level:.0%}: {compare[f'ложных ≤{level:.0%}']}")

    # Вторая половина вопроса: что делает правило по прибору, когда прибор врёт.
    # Окно «плохих данных» выбрано не нами — это демо-сценарий ТЗ.
    lo, hi = cfg["demo_windows"]["bad_data_frozen_pak"]
    # берём ВСЕ моменты окна, а не только те, у которых есть лабораторный факт за
    # 24 ч: оператор в это окно смотрит целиком, и молчание лаборатории — часть
    # картины, а не повод выкинуть моменты из счёта
    window = rows.set_index("ts").loc[str(lo):str(hi)]
    idx = pd.DatetimeIndex(window.index)
    pos = analyzer.index.searchsorted(idx, side="right") - 1
    live_window = np.where(pos >= 0, analyzer.to_numpy()[pos.clip(min=0)], np.nan)
    lab_window = sb.lims_sulfur.loc[str(lo):str(hi)].dropna()
    outcomes = window["исход"].value_counts().to_dict()
    bad_data = {
        "окно": [str(lo), str(hi)],
        "моментов": int(len(window)),
        "исходы системы": {str(k): int(v) for k, v in outcomes.items()},
        "доля отказов системы": round(float(window["исход"].eq("отказ").mean()), 3),
        "правило по Q21: доля тревог": round(float(np.nanmean(live_window > limit)), 3),
        "медиана показания Q21": round(float(np.nanmedian(live_window)), 1),
        "лаборатория в окне": [round(float(x), 1) for x in lab_window.values],
        "проб выше предела": int((lab_window > limit).sum()),
    }
    print(f"\nокно плохих данных {lo} … {hi}: "
          f"{json.dumps(bad_data, ensure_ascii=False)}")

    wins = 0
    for block in compare.values():
        ours, theirs = block.get("система"), block.get("правило по Q21")
        if ours is not None and theirs is not None and ours < theirs:
            wins += 1
    accepted = wins >= 2
    print(f"\nсистема лучше на {wins} уровнях строгости из {len(LEVELS)}")
    print("  ВЫВОД: преимущество системы в решениях "
          f"{'ПОДТВЕРЖДЕНО' if accepted else 'НЕ подтверждено'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), **reliability_provenance(cfg),
                                  "прогон": RUN, "окно факта, ч": HOURS,
                                  "моментов": int(len(known)),
                                  "перед превышением": int(over_live.sum()),
                                  "кривые": curves, "сравнение": compare,
                                  "окно плохих данных": bad_data,
                                  "уровней в пользу системы": wins,
                                  "подтверждено": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
