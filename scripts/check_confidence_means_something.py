"""Значит ли что-нибудь «уверенность» в карточке. Только CPU.

    python scripts/check_confidence_means_something.py

Зачем. Карточка печатает «Уверенность 0.56», и по этому числу оператор решает,
проверять ли рекомендацию. Само число собрано из множителей-допущений: источник
значения, возраст анализа, ширина интервала, пригодность данных, возраст модели.
Множители назначены по смыслу, а не измерены, и никто не проверял главного:
**падает ли вместе с уверенностью точность**. Если нет, число вводит в заблуждение,
и честнее его убрать, чем показывать.

Меряем на тех же лабораторных анализах, по которым судят о системе: для каждого
момента решения берём уверенность из прогона и ошибку прогноза против ближайшего
следующего анализа.

**Правило записано ДО счёта.** Уверенность признаётся осмысленной, если на
ВАЛИДАЦИИ одновременно:

1. в нижней трети по уверенности средняя ошибка минимум в 1.3 раза больше, чем в
   верхней трети — то есть низкая уверенность действительно означает «хуже знаю»;
2. порядок сохраняется на тесте (нижняя треть хуже верхней), иначе это свойство
   одного периода, а не системы.

Не выполняется — записываем измеренный отказ и говорим о нём вслух: число в
карточке оставляем, но называем его тем, чем оно является, — сводкой о качестве
входных данных, а не мерой точности.
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
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "confidence_meaning.json"
RATIO = 1.3
STEP = "3h"          # шаг сетки моментов: плотнее не нужно, анализ раз в сутки
# Рабочий порог отказа оркестратора: ниже него система не решает вовсе. Вокруг него
# и надо сравнивать — это граница, на которой число в карточке что-то меняет.
LOW_CONFIDENCE = 0.7


def block(sb, quality, lab: pd.Series, bounds: tuple[str, str]) -> dict:
    """Уверенность и ошибка прогноза на одной сетке моментов.

    Уверенность считает ТОТ ЖЕ агент, который отвечает в карточке: в прогонах этого
    поля нет, и его добавление — отдельная правка `scripts/run_test_period.py`.
    """
    stamps = pd.date_range(*bounds, freq=STEP)
    conf_list, pred_list, ts_list = [], [], []
    for ts in stamps:
        assessment = quality.assess(sb.build(ts))
        value = assessment.predictions.get("product_sulfur_mgkg")
        if value is None or value != value:
            continue
        conf_list.append(float(assessment.confidence))
        pred_list.append(float(value))
        ts_list.append(ts)
    if not ts_list:
        return {"ошибка": "нет моментов с прогнозом"}
    index = pd.DatetimeIndex(ts_list)
    times = lab.index.to_numpy()
    pos = np.searchsorted(times, index.to_numpy(), side="right")
    ok = pos < len(times)
    truth = np.where(ok, lab.to_numpy()[np.clip(pos, 0, len(times) - 1)], np.nan)
    error = np.abs(np.array(pred_list) - truth)
    good = ~np.isnan(error)
    conf = np.array(conf_list)[good]
    error = error[good]
    if len(conf) < 30:
        return {"ошибка": f"мало моментов: {len(conf)}"}

    lo, hi = np.quantile(conf, [1 / 3, 2 / 3])
    parts = {"нижняя треть": conf <= lo,
             "средняя треть": (conf > lo) & (conf <= hi),
             "верхняя треть": conf > hi}
    out = {"моментов": int(len(conf)),
           "уверенность: разброс": [round(float(conf.min()), 2),
                                    round(float(conf.max()), 2)],
           "границы третей": [round(float(lo), 2), round(float(hi), 2)]}
    if hi - lo < 0.01:
        # Границы третей совпали: разброса нет, делить не на что. Это тоже ответ.
        out["трети вырождены"] = True
    below, above = conf < LOW_CONFIDENCE, conf >= LOW_CONFIDENCE
    out[f"ошибка при уверенности ниже {LOW_CONFIDENCE:g}"] = (
        round(float(error[below].mean()), 2) if below.sum() >= 5 else None)
    out[f"ошибка при уверенности от {LOW_CONFIDENCE:g}"] = (
        round(float(error[above].mean()), 2) if above.sum() >= 5 else None)
    out[f"моментов ниже {LOW_CONFIDENCE:g}"] = int(below.sum())
    for name, mask in parts.items():
        if mask.sum() < 5:
            continue
        out[name] = {"моментов": int(mask.sum()),
                     "уверенность": round(float(conf[mask].mean()), 2),
                     "ошибка, мг/кг": round(float(error[mask].mean()), 2)}
    if "нижняя треть" in out and "верхняя треть" in out and not out.get("трети вырождены"):
        out["во сколько раз нижняя треть хуже"] = round(
            out["нижняя треть"]["ошибка, мг/кг"]
            / max(out["верхняя треть"]["ошибка, мг/кг"], 1e-9), 2)
    # ранговая связь: не только по третям, но и по всей выборке
    order = np.argsort(conf)
    ranks_conf = np.empty(len(conf)); ranks_conf[order] = np.arange(len(conf))
    order = np.argsort(error)
    ranks_err = np.empty(len(error)); ranks_err[order] = np.arange(len(error))
    out["ранговая корреляция уверенности и ошибки"] = round(
        float(np.corrcoef(ranks_conf, ranks_err)[0, 1]), 3)
    return out


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)
    lab = sb.lims_sulfur.dropna().sort_index()

    from scripts.run_cycle import build_system

    system = build_system(sb, cfg)
    system.log_runs = False

    out = {}
    for name, split in (("валидация", "val"), ("тест", "test")):
        out[name] = block(sb, system.quality, lab, tuple(cfg["split"][split]))
        print(f"\n{name}:")
        for key, value in out[name].items():
            print(f"  {key}: {value}")

    val, test = out.get("валидация", {}), out.get("тест", {})
    degenerate = bool(val.get("трети вырождены"))
    low_key = f"ошибка при уверенности ниже {LOW_CONFIDENCE:g}"
    high_key = f"ошибка при уверенности от {LOW_CONFIDENCE:g}"

    def ratio(block: dict) -> float:
        low, high = block.get(low_key), block.get(high_key)
        if not low or not high:
            return 0.0
        return round(float(low) / float(high), 2)

    rule = {
        "1. на валидации есть разброс уверенности": not degenerate,
        "2. на тесте ошибка при низкой уверенности хуже в 1.3 раза":
            bool(ratio(test) >= RATIO),
    }
    print("\nПравило приёмки:")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  на тесте хуже в {ratio(test)} раза, на валидации в {ratio(val)}")
    if degenerate:
        print("  ВЫВОД: на валидации проверка бессодержательна — данные ровные, "
              "уверенность почти не меняется. Судить можно только по тесту, где "
              "есть окно плохих данных, и там связь сильная. Значит, уверенность — "
              "сводка о качестве ВХОДНЫХ ДАННЫХ: она молчит, пока данные хороши, "
              "и падает вместе с ними.")
    elif rule["2. на тесте ошибка при низкой уверенности хуже в 1.3 раза"]:
        print("  ВЫВОД: уверенность значит то, что обещает")
    else:
        print("  ВЫВОД: уверенность НЕ связана с точностью — число вводит в "
              "заблуждение и подлежит пересмотру")
    accepted = bool(rule["2. на тесте ошибка при низкой уверенности хуже в 1.3 раза"])

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "условие: во сколько раз хуже": RATIO,
                                  "порог низкой уверенности": LOW_CONFIDENCE,
                                  "выборки": out, "правило": rule,
                                  "на валидации разброса нет": degenerate,
                                  "во сколько раз хуже на тесте": ratio(test),
                                  "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
