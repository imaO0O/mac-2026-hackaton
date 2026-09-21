"""Можно ли обещать оператору СЛЕДУЮЩИЙ лабораторный анализ. Только CPU.

    python scripts/check_next_lab.py

Зачем. Карточка сообщает прогноз и интервал НА СЕЙЧАС, а оператор ждёт другого:
что придёт из лаборатории и когда. Лаборатория — контрольный факт по ТЗ, по ней
судят о работе системы, и между решением и анализом проходят часы. Если интервал
модели честен не только «на сейчас», но и на момент следующего анализа, карточка
может называть диапазон ожидаемого результата — это ровно то, что оператор потом
сверит с бумагой.

Меряем на тех же данных два обещания по отдельности.

**Правило записано ДО счёта.** На ВАЛИДАЦИИ:

1. **диапазон** называем, если фактический следующий анализ попадает в интервал
   80 % в 70–90 % случаев. Ниже — обещание врёт; выше — интервал настолько широк,
   что ничего не обещает;
2. **время** называем, если медиана ошибки предсказания момента следующего анализа
   не больше 6 часов, то есть попадаем в пределах смены. Иначе называем только
   диапазон, без времени.

Не выполняется ни то, ни другое — измеренный отказ, карточка остаётся как есть.

**Вариант Б, правило тоже записано до счёта.** Узкий интервал — ещё не приговор
обещанию: его можно расширить ровно настолько, насколько дальше от «сейчас» лежит
анализ. Множитель к сигме подбирается на ОБУЧАЮЩЕМ периоде до покрытия 80 % и
проверяется на валидации тем же условием 70–90 %. Подбирать на валидации значило бы
подгонять под то, чем потом хвалимся, поэтому обучение.

Отдельно считаем, как обещание портится с ростом ожидания (0–6, 6–12, 12–24, >24 ч):
если диапазон держится только на коротком плече, это и должно быть сказано вслух,
а не спрятано в среднем.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.models.quality_model import Z90, SulfurModel  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "next_lab.json"
COVERAGE_BAND = (0.70, 0.90)
MAX_TIME_ERROR_H = 6.0
BUCKETS = [(0.0, 6.0), (6.0, 12.0), (12.0, 24.0), (24.0, 1e9)]


def next_lab(lab: pd.Series, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Значение и момент ПЕРВОГО анализа строго позже каждого момента решения."""
    times = lab.index.to_numpy()
    pos = np.searchsorted(times, index.to_numpy(), side="right")
    ok = pos < len(times)
    value = np.where(ok, lab.to_numpy()[np.clip(pos, 0, len(times) - 1)], np.nan)
    when = np.where(ok, times[np.clip(pos, 0, len(times) - 1)],
                    np.datetime64("NaT"))
    return value, when


def last_lab_time(lab: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    times = lab.index.to_numpy()
    pos = np.searchsorted(times, index.to_numpy(), side="right") - 1
    return np.where(pos >= 0, times[np.clip(pos, 0, len(times) - 1)],
                    np.datetime64("NaT"))


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    model = SulfurModel.load(SulfurModel.default_path(0.0, "sulfur"))
    features = build_feature_matrix()
    masks = time_split(features.index, cfg)
    sb = StateBuilder(cfg)
    lab = sb.lims_sulfur.dropna().sort_index()

    # Ритм анализов берём с ОБУЧАЮЩЕГО периода: предсказывать момент по будущему
    # нельзя, это была бы утечка в ту самую величину, которую обещаем.
    train_lo, train_hi = cfg["split"]["train"]
    train_lab = lab.loc[str(train_lo):str(train_hi)]
    cadence_h = float(pd.Series(train_lab.index).diff().dt.total_seconds().median() / 3600)
    print(f"ритм анализов на обучении: медиана {cadence_h:.1f} ч, проб {len(train_lab)}")

    # Множитель к сигме для обещания «на следующий анализ» — с ОБУЧАЮЩЕГО периода.
    Xtr = features[masks["train"].to_numpy()][model.features].dropna()
    vtr, _ = next_lab(lab, pd.DatetimeIndex(Xtr.index))
    ptr = model.predict_frame(Xtr)
    resid = np.abs(vtr - ptr["q50"].to_numpy()) / np.maximum(ptr["sigma"].to_numpy(), 1e-6)
    resid = resid[~np.isnan(resid)]
    widen = float(np.quantile(resid, 0.80)) if len(resid) else float("nan")
    print(f"множитель к сигме по обучению (покрытие 80 %): {widen:.2f}")

    out: dict = {"ритм анализов на обучении, ч": round(cadence_h, 2),
                 "множитель к сигме по обучению": round(widen, 3)}
    for split in ("val", "test"):
        X = features[masks[split].to_numpy()][model.features].dropna()
        if X.empty:
            continue
        index = pd.DatetimeIndex(X.index)
        value, when = next_lab(lab, index)
        pred = model.predict_frame(X)
        wait_h = (when - index.to_numpy()) / np.timedelta64(1, "h")
        # интервал 80 % в том же определении, что и везде в проекте
        half = Z90 * pred["sigma"].to_numpy()
        inside = (np.abs(value - pred["q50"].to_numpy()) <= half)
        wide = np.abs(value - pred["q50"].to_numpy()) <= widen * pred["sigma"].to_numpy()
        good = ~np.isnan(value)

        previous = last_lab_time(lab, index)
        guess = previous + np.timedelta64(int(cadence_h * 3600), "s")
        time_error = np.abs((guess - when) / np.timedelta64(1, "h"))

        rows = []
        for lo, hi in BUCKETS:
            sel = good & (wait_h >= lo) & (wait_h < hi)
            if sel.sum() < 20:
                continue
            rows.append({
                "ожидание, ч": f"{lo:g}–{hi:g}" if hi < 1e9 else f">{lo:g}",
                "моментов": int(sel.sum()),
                "покрытие интервалом 80 %": round(float(inside[sel].mean()), 3),
                "покрытие расширенным интервалом": round(float(wide[sel].mean()), 3),
                "средняя ошибка прогноза, мг/кг": round(
                    float(np.mean(np.abs(pred["q50"].to_numpy()[sel] - value[sel]))), 2),
            })
        out[split] = {
            "моментов": int(good.sum()),
            "медиана ожидания, ч": round(float(np.nanmedian(wait_h[good])), 1),
            "покрытие интервалом 80 %": round(float(inside[good].mean()), 3),
            "покрытие расширенным интервалом": round(float(wide[good].mean()), 3),
            "полуширина расширенного интервала, мг/кг": round(
                float(np.median(widen * pred["sigma"].to_numpy()[good])), 2),
            "MAE до следующего анализа, мг/кг": round(
                float(np.mean(np.abs(pred["q50"].to_numpy()[good] - value[good]))), 2),
            "медиана ошибки момента, ч": round(float(np.nanmedian(time_error[good])), 1),
            "доля моментов с ошибкой времени до 6 ч": round(
                float(np.nanmean(time_error[good] <= MAX_TIME_ERROR_H)), 3),
            "по ожиданию": rows,
        }
        print(f"\n{split}: моментов {int(good.sum())}, медиана ожидания "
              f"{out[split]['медиана ожидания, ч']} ч, покрытие "
              f"{out[split]['покрытие интервалом 80 %']:.1%}, MAE "
              f"{out[split]['MAE до следующего анализа, мг/кг']}, ошибка момента "
              f"{out[split]['медиана ошибки момента, ч']} ч")
        print(f"    расширенный интервал покрывает "
              f"{out[split]['покрытие расширенным интервалом']:.1%}, полуширина "
              f"{out[split]['полуширина расширенного интервала, мг/кг']} мг/кг")
        for row in rows:
            print(f"    {row}")

    val = out.get("val", {})
    coverage = float(val.get("покрытие интервалом 80 %", 0.0))
    time_error_h = float(val.get("медиана ошибки момента, ч", 99.0))
    wide_coverage = float(val.get("покрытие расширенным интервалом", 0.0))
    rule = {
        "1. покрытие следующего анализа в 70–90 %": bool(
            COVERAGE_BAND[0] <= coverage <= COVERAGE_BAND[1]),
        "2. момент анализа угадан в пределах смены (6 ч)": bool(
            time_error_h <= MAX_TIME_ERROR_H),
        "Б. расширенный интервал попадает в 70–90 %": bool(
            COVERAGE_BAND[0] <= wide_coverage <= COVERAGE_BAND[1]),
    }
    print("\nПравило приёмки (валидация):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    narrow_ok = rule["1. покрытие следующего анализа в 70–90 %"]
    wide_ok = rule["Б. расширенный интервал попадает в 70–90 %"]
    time_ok = rule["2. момент анализа угадан в пределах смены (6 ч)"]
    if narrow_ok or wide_ok:
        verdict = ("интервал модели" if narrow_ok else "расширенный интервал")
        verdict += (" и время анализа" if time_ok else ", без времени")
    else:
        verdict = ("только время анализа" if time_ok
                   else "ничего: обещание не подтверждено")
    print(f"  ВЫВОД: в карточке называем — {verdict}")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "условие: покрытие в диапазоне": list(COVERAGE_BAND),
                                  "условие: ошибка момента не больше, ч": MAX_TIME_ERROR_H,
                                  **out, "правило": rule, "вывод": verdict},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
