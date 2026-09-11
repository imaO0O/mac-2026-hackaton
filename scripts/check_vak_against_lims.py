"""Сверка формул справочника с лабораторией: тот ли поток они описывают.

    python scripts/check_vak_against_lims.py
    python scripts/check_vak_against_lims.py --min-pairs 50

Проверка правдоподобия (`models/vak._plausible_share`) отвечает на вопрос «похоже
ли это вообще на физическую величину» и отсеивает бессмыслицу вроде 198 000 °C. Но
она **не отличает правильную величину от чужой**: плотность 869 кг/м³ выглядит
совершенно нормально, даже если на самом деле этот поток имеет 847.

Дыру нашёл участник 2, когда восстановил привязку блоков справочника к точкам
отбора ЛИМС. Оказалось, что `vak_AVT6_240_350_D15` — признак, который стоит в
рабочей модели горизонта 0 и занимает 16-е место по важности, — завышает плотность
на 22 кг/м³ относительно лаборатории. Проверку правдоподобия он проходит на 100 %.

Что делает этот скрипт
----------------------
Для каждой считающейся формулы ищет в ЛИМС ряд ТОГО ЖЕ показателя и сравнивает:

* **смещение** — систематический сдвиг уровня. Для градиентного бустинга
  постоянный сдвиг безвреден (дерево делит по порогу, а не по значению), но для
  ОБЪЯСНЕНИЯ оператору он смертелен: показывать «плотность 869» там, где
  лаборатория меряет 847, нельзя;
* **корреляцию** — а ту ли величину формула вообще отслеживает. Смещение
  лечится калибровкой, отсутствие корреляции — нет;
* **лучшую точку отбора** — с какой из точек ЛИМС формула согласуется сильнее
  всего. Привязку не задаём руками, а выводим из данных и показываем: если
  «своя» точка проигрывает чужой, это и есть ошибка привязки.

Осознанное ограничение: сверить можно только те формулы, у которых в ЛИМС есть
одноимённый показатель. Для вязкости, отгона 350 и части разгонок точки отбора
нет, и они остаются непроверенными — это честнее, чем молча их пропустить.

Пересечение с работой участника 2. Его `scripts/check_avt_formulas.py` разбирает
блок АВТ глубже: по слагаемым, с проверкой гипотез об ошибке записи. Этот скрипт
шире, но мельче — он покрывает оба блока и отвечает на один вопрос: можно ли
доверять признаку, который РЕАЛЬНО стоит в модели качества. Привязку блока
240-350 к точке АВТ|3 мы вывели независимо и получили одно и то же (+21.8 против
его +23.8 кг/м³ на тех же двух сотнях анализов), так что привязке можно верить.

Результат: reports/vak_vs_lims.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.data.loaders import lims_series, load_lims  # noqa: E402
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.models.vak import compile_formulas  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

# Показатель формулы → как он называется в ЛИМС. Только то, что реально есть в
# выданных данных; всё остальное сверить не с чем, и мы об этом говорим прямо.
PARAM_TO_LIMS = {
    "D15": "D15",
    "T50": "50%.T",
    "T90": "90%.T",
    "T95": "95%.T",
    "95%.T": "95%.T",
    "EBP": "EBP.T",
    "IBP": "IBP.T",
    "CloudPoint": "CloudPoint",
    "CFPP": "CFPP",
    "I250": "I250",
    "I350": "I350",
}

# Точки отбора по УСТАНОВКАМ. Установка формулы известна из её блока и догадкой
# не является: `AVT6:*` считается по тегам АВТ, `24-2000:*` — по тегам гидроочистки.
# Сравнивать формулу АВТ с продуктом гидроочистки бессмысленно, а именно это и
# выходило, пока кандидаты не были разделены: у Гидроочистки|2 просто больше всего
# анализов, и она выигрывала любой отбор «по объёму».
POINTS_BY_UNIT = {
    "avt": ["АВТ|1", "АВТ|2", "АВТ|2.1", "АВТ|3"],
    "ht": ["Гидроочистка|1", "Гидроочистка|2"],
}

# Штатная точка отбора для блока формулы — там, где она известна без догадок.
# Блок `24-2000:GODT` это продукт гидроочистки, и его точка в ЛИМС одна.
# Для блоков АВТ соответствие «срез в справочнике → точка отбора» НЕ установлено:
# 240-350 не совпадает ни с одним выданным срезом (в ЛИМС есть 150-250 и 290-350).
# Поэтому для них штатной точки не задаём и показываем все — догадка тут была бы
# ровно той ошибкой, которую этот скрипт и должен ловить.
NOMINAL_POINT = {"24-2000:GODT": "Гидроочистка|2"}

# Насколько меньше пар допускается у «лучшей» точки по сравнению с самой полной.
# Без этого правила победителем становится любая мелкая выборка: у 24-2000:GODT:T90
# корреляция с АВТ|2 равна 0.44 на 204 парах против 0.14 со своей точкой на 1251 —
# и первое выглядит убедительнее, хотя означает лишь, что на коротком куске
# совпало. Сравнивать надо сопоставимые по объёму ряды.
MIN_PAIRS_SHARE = 0.5

# Физически осмысленные границы, чтобы в сравнение не попадали заглушки ЛИМС
# (в разгонке встречается 0, в плотности — единицы).
SANE = {"D15": (700.0, 1000.0), "50%.T": (100.0, 400.0), "90%.T": (150.0, 450.0),
        "95%.T": (150.0, 450.0), "EBP.T": (150.0, 500.0), "IBP.T": (50.0, 350.0),
        "CloudPoint": (-60.0, 40.0), "CFPP": (-60.0, 40.0),
        "I250": (0.0, 100.0), "I350": (0.0, 100.0)}


def compare(feature: pd.Series, lab: pd.Series) -> dict | None:
    """Смещение и корреляция признака относительно лабораторного ряда.

    Лабораторный анализ сопоставляется с ПОСЛЕДНИМ предшествующим значением
    признака: сравнивать надо то, что система знала на момент отбора пробы.
    """
    lab = lab[~lab.index.duplicated(keep="last")].sort_index()
    feature = feature.dropna()
    if feature.empty or lab.empty:
        return None
    pos = feature.index.searchsorted(lab.index, side="right") - 1
    ok = pos >= 0
    if ok.sum() < 5:
        return None
    matched = feature.to_numpy()[pos[ok]]
    values = lab.to_numpy()[ok]
    good = ~np.isnan(matched) & ~np.isnan(values)
    if good.sum() < 5:
        return None
    diff = matched[good] - values[good]
    corr = (float(np.corrcoef(matched[good], values[good])[0, 1])
            if np.std(matched[good]) > 0 and np.std(values[good]) > 0 else float("nan"))
    return {
        "пар": int(good.sum()),
        "среднее лаб.": round(float(values[good].mean()), 2),
        "среднее форм.": round(float(matched[good].mean()), 2),
        "смещение": round(float(diff.mean()), 2),
        "MAE": round(float(np.abs(diff).mean()), 2),
        "корреляция": None if corr != corr else round(corr, 3),
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pairs", type=int, default=20,
                    help="меньше пар — сравнению верить нельзя")
    args = ap.parse_args()

    feats = build_feature_matrix()
    lims = load_lims()
    usable, _ = compile_formulas()

    rows, unchecked = [], []
    for item in usable:
        column = item["column"]
        if column not in feats.columns:
            unchecked.append({"формула": item["target"],
                              "причина": "не прошла проверку правдоподобия"})
            continue
        param = item["target"].rsplit(":", 1)[-1]
        lims_param = PARAM_TO_LIMS.get(param)
        if lims_param is None:
            unchecked.append({"формула": item["target"],
                              "причина": f"показателя «{param}» нет в ЛИМС"})
            continue

        results = {}
        for point in POINTS_BY_UNIT.get(item["unit"], []):
            try:
                lab = lims_series(f"{point}|{lims_param}", lims)
            except KeyError:
                continue
            lo, hi = SANE.get(lims_param, (-1e9, 1e9))
            lab = lab[(lab > lo) & (lab < hi)]
            stats = compare(feats[column], lab)
            if stats is None or stats["пар"] < args.min_pairs:
                continue
            results[point] = stats

        if not results:
            unchecked.append({"формула": item["target"],
                              "причина": "нет точки отбора с достаточным числом пар"})
            continue

        block = item["target"].rsplit(":", 1)[0]
        nominal = NOMINAL_POINT.get(block)
        most_pairs = max(results, key=lambda k: results[k]["пар"])
        # Порог по объёму: на короткой выборке совпасть может что угодно.
        floor = MIN_PAIRS_SHARE * results[most_pairs]["пар"]
        comparable = [k for k in results if results[k]["пар"] >= floor]

        if nominal in results:
            # Установка одна, точка отбора известна — сверяем с ней, и точка.
            reference, derived = nominal, False
        else:
            # Привязка среза справочника к точке отбора НЕ установлена (срез
            # 240-350 не совпадает ни с одним выданным). Выводим её по данным:
            # берём точку, с которой формула согласуется лучше всего среди
            # сопоставимых по объёму. Это гипотеза, и она помечена как гипотеза.
            reference = max(comparable,
                            key=lambda k: abs(results[k]["корреляция"] or 0.0))
            derived = True

        rival = max((k for k in comparable if k != reference),
                    key=lambda k: abs(results[k]["корреляция"] or 0.0), default=None)

        stats = results[reference]
        rows.append({
            "формула": item["target"],
            "исправлена": "да" if item["corrected"] else "нет",
            "точка ЛИМС": reference + (" (по данным)" if derived else ""),
            **stats,
            "ближайшая чужая": (
                "—" if rival is None else
                f"{rival} {results[rival]['корреляция']:+.2f}"),
        })

    frame = pd.DataFrame(rows)
    print("\nФормулы справочника против лаборатории "
          f"(точки отбора только своей установки, минимум {args.min_pairs} пар)\n")
    print(frame.to_string(index=False))

    if unchecked:
        print(f"\nНе сверялись — {len(unchecked)}:")
        for item in unchecked:
            print(f"   {item['формула']:26s} {item['причина']}")

    # Два разных диагноза, и путать их нельзя.
    no_signal = frame[(frame["корреляция"].isna()) | (frame["корреляция"].abs() < 0.2)]
    biased = frame[(frame["смещение"].abs() > 0.05 * frame["среднее лаб."].abs())
                   & (~frame.index.isin(no_signal.index))]

    print("\nЧто из этого следует.")
    if len(no_signal):
        print(f"  НЕ ОТСЛЕЖИВАЮТ величину (|корреляция| < 0.2): "
              + ", ".join(no_signal["формула"]) + ".")
        print("  Это тяжёлый диагноз: смещение лечится калибровкой, отсутствие "
              "связи — нет. Такой признак несёт что угодно, только не заявленный "
              "показатель.")
    if len(biased):
        print(f"  СМЕЩЕНЫ больше чем на 5 % уровня: "
              + ", ".join(f"{r['формула']} ({r['смещение']:+g})"
                          for _, r in biased.iterrows()) + ".")
        print("  Для бустинга постоянный сдвиг безвреден — дерево делит по порогу. "
              "А вот показывать такое значение оператору как измеренную величину "
              "нельзя, и в объяснении рекомендации его быть не должно.")
    if not len(no_signal) and not len(biased):
        print("  Все сверенные формулы согласуются с лабораторией по уровню и "
              "направлению.")

    out = ROOT / "reports" / "vak_vs_lims.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "сверено": rows, "не_сверялось": unchecked,
        "без_связи": list(no_signal["формула"]),
        "смещённые": list(biased["формула"]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
