"""Цетановое число: что о нём говорят выданные данные.

    python scripts/check_cetane.py

Организаторы назвали цетановое число одним из трёх обязательных показателей
качества — наравне с серой и Т95. Скрипт отвечает на четыре вопроса, и все ответы
получаются из данных, а не из общих соображений:

1. **Сколько его вообще меряют?** От этого зависит, можно ли строить модель.
2. **Работает ли цетановый индекс?** Классический ASTM D976 считается по плотности
   и Т50, которые у нас есть каждый день. Если он воспроизводит лабораторию, у нас
   есть виртуальный анализатор цетанового числа. Если нет — надо честно сказать,
   что его нет, и не выдавать индекс за прогноз.
3. **Куда он движется?** Показатель может выйти за норматив не рывком, а сползанием.
4. **Во что обходится присадка?** Организаторы дали дозу до 3 % и цену 100× за
   тонну. Отсюда считается, сколько стоит вытянуть недостающие единицы.

Результат: reports/cetane.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.data.loaders import lims_series, load_lims  # noqa: E402
from nefte.models.cetane import (  # noqa: E402
    CETANE_SPEC_MIN,
    IMPROVER_MAX_PCT,
    IMPROVER_PRICE_RATIO,
    cetane_index_d976,
    dose_for_deficit,
    improver_cost_share,
    improver_effect,
    trend_per_year,
)
from nefte.utils import use_utf8_console  # noqa: E402


def matched(cetane: pd.Series, others: dict[str, pd.Series],
            tolerance: str = "48h") -> pd.DataFrame:
    """Анализы цетанового числа с ближайшими по времени плотностью и Т50.

    Точного совпадения меток нет: разные показатели меряют в разное время. Берём
    ближайший анализ в пределах допуска — иначе сопоставлять было бы нечего.
    """
    base = cetane.rename("cn").reset_index()
    base.columns = ["ts", "cn"]
    base = base.sort_values("ts")
    for name, series in others.items():
        right = series.rename(name).reset_index()
        right.columns = ["ts", name]
        base = pd.merge_asof(base, right.sort_values("ts"), on="ts",
                             direction="nearest",
                             tolerance=pd.Timedelta(tolerance))
    return base.dropna()


def main() -> int:
    use_utf8_console()
    lims = load_lims()
    cetane = lims_series("Гидроочистка|2|CetaneNumber", lims).sort_index()
    d15 = lims_series("Гидроочистка|2|D15", lims).sort_index()
    t50 = lims_series("Гидроочистка|2|50%.T", lims).sort_index()

    print("1. Сколько меряют")
    gaps = cetane.index.to_series().diff().dt.total_seconds() / 86400
    print(f"   анализов: {len(cetane)} за "
          f"{(cetane.index[-1] - cetane.index[0]).days} суток; "
          f"интервал медиана {gaps.median():.0f} сут, максимум {gaps.max():.0f} сут")
    print(f"   значения: min {cetane.min():.1f}, среднее {cetane.mean():.2f}, "
          f"max {cetane.max():.1f}, норматив {CETANE_SPEC_MIN}")
    below = int((cetane < CETANE_SPEC_MIN).sum())
    print(f"   ниже норматива: {below} из {len(cetane)}; "
          f"последний анализ {cetane.iloc[-1]:.1f} от "
          f"{cetane.index[-1]:%d.%m.%Y}")

    print("\n2. Работает ли цетановый индекс ASTM D976")
    pairs = matched(cetane, {"d15": d15, "t50": t50})
    index = pairs.apply(lambda r: cetane_index_d976(r["d15"], r["t50"]), axis=1)
    ok = index.notna()
    pairs, index = pairs[ok], index[ok].astype("float64")
    err = index.to_numpy() - pairs["cn"].to_numpy()
    corr = float(np.corrcoef(index, pairs["cn"])[0, 1])
    # честный ориентир: насколько ошибётся тот, кто вообще ничего не считает
    naive = float(np.abs(pairs["cn"] - pairs["cn"].mean()).mean())
    print(f"   сопоставлено пар: {len(pairs)}")
    print(f"   корреляция индекса с лабораторией: {corr:+.3f}")
    print(f"   разброс: индекс {index.std():.2f}, лаборатория {pairs['cn'].std():.2f}")
    print(f"   MAE индекса {np.abs(err).mean():.2f}; "
          f"MAE «просто среднее по истории» {naive:.2f}")
    works = abs(corr) > 0.3 and np.abs(err).mean() < naive
    print("   ВЫВОД: индекс " + ("воспроизводит лабораторию — годится как "
                                 "виртуальный анализатор."
                                 if works else
                                 "лабораторию НЕ воспроизводит и хуже среднего. "
                                 "Прогноза цетанового числа у нас нет, и выдавать "
                                 "индекс за прогноз нельзя."))

    print("\n3. Куда движется")
    by_year = cetane.groupby(cetane.index.year).agg(["count", "mean", "min"])
    print(by_year.round(2).to_string())
    slope = trend_per_year(cetane)
    print(f"   наклон тренда: {slope:+.2f} единиц в год")
    if slope < 0:
        margin = float(cetane.iloc[-1]) - CETANE_SPEC_MIN
        print(f"   запас последнего анализа до норматива: {margin:+.1f} единиц; "
              + ("запаса нет — норматив уже нарушен." if margin < 0 else
                 f"при таком наклоне он кончится через "
                 f"{margin / -slope * 12:.0f} мес."))

    print("\n4. Чего стоит присадка")
    print(f"   дозировка до {IMPROVER_MAX_PCT} % массы, цена {IMPROVER_PRICE_RATIO:.0f}× "
          "цены дизеля за тонну (оба числа — ответ организаторов)")
    rows = []
    for dose in (0.05, 0.1, 0.2, 0.5, 1.0, 3.0):
        rows.append({"доза, % масс.": dose,
                     "прибавка ЦЧ": round(improver_effect(dose), 2),
                     "цена, долей цены ДТ": round(improver_cost_share(dose), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    deficit = max(0.0, CETANE_SPEC_MIN - float(cetane.iloc[-1]))
    need = dose_for_deficit(deficit)
    if deficit <= 0:
        verdict = "последний анализ норматив проходит, присадка не нужна"
    elif need is None:
        verdict = (f"нехватку {deficit:.1f} единиц присадкой не закрыть даже при "
                   f"{IMPROVER_MAX_PCT} % — нужен другой компонент смешения")
    else:
        verdict = (f"чтобы закрыть нехватку {deficit:.1f} единиц, нужно "
                   f"{need:.3f} % присадки — это {improver_cost_share(need) * 100:.1f} % "
                   "стоимости тонны топлива")
    print(f"   На последнем анализе: {verdict}")
    print("\n   Отклик на дозу — ДОПУЩЕНИЕ (паспорта присадки в пакете нет), "
          "насыщающаяся кривая. Предел дозы и цена — не допущение.")

    out = ROOT / "reports" / "cetane.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "анализов": int(len(cetane)),
        "интервал_суток_медиана": float(gaps.median()),
        "последний": {"значение": float(cetane.iloc[-1]),
                      "дата": str(cetane.index[-1])},
        "ниже_норматива": below,
        "норматив": CETANE_SPEC_MIN,
        "по_годам": {str(k): {kk: float(vv) for kk, vv in v.items()}
                     for k, v in by_year.to_dict("index").items()},
        "тренд_единиц_в_год": slope,
        "индекс_d976": {"пар": int(len(pairs)), "корреляция": corr,
                        "MAE": float(np.abs(err).mean()), "MAE_среднего": naive,
                        "годится": bool(works)},
        "присадка": {"предел_%": IMPROVER_MAX_PCT, "цена_к_дизелю": IMPROVER_PRICE_RATIO,
                     "таблица": rows, "вердикт": verdict},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
