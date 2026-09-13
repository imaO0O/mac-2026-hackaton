"""Стареет ли модель и как быстро — по возрасту относительно конца обучения.

    python scripts/check_drift.py
    python scripts/check_drift.py --horizon 0 --freq QE

Вопрос, который на защите зададут обязательно: **как часто её переобучать?**
Отвечать «раз в месяц, наверное» нельзя — это надо измерить.

Повод не теоретический. Модель Т95 выиграла валидацию и проиграла тест
(`docs/HARD_CHECKS.md` §8.8) именно из-за дрейфа: показатель растёт год от года,
модель обучена на прошлом и систематически недотягивает. Заранее записанное
правило «победитель выбирается по валидации» от подгонки под тест защищает, а от
дрейфа — нет. Значит, дрейф надо мерить отдельно.

Как устроено измерение
----------------------
Обучение кончается 30.06.2025. Дальше идут валидация (0–6 месяцев после) и тест
(6–19 месяцев после), и это естественная шкала ВОЗРАСТА модели. Разбиваем всё, что
после обучения, на интервалы и смотрим, как метрики меняются с возрастом.

Разделяем два разных «дрейфа», иначе вывод получится бессмысленным:

* **дрейф данных** — меняются сами условия: средний уровень показателя, частота
  превышений. Модель тут ни при чём, но её рабочая точка едет;
* **дрейф модели** — растёт ошибка и смещение прогноза. Вот это и есть старение.

Отдельно считается **смещение** (не MAE): именно оно выдаёт старение раньше всего.
MAE растёт и от шума, а систематический уход прогноза в одну сторону — признак
того, что мир уехал, а модель осталась.

Базой сравнения служит персистенция (предыдущий анализ). Она не стареет по
построению: ей нечего помнить. Если ошибка модели растёт, а персистенции — нет,
это старение модели, а не усложнение периода.

Результат: reports/drift_h<горизонт>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.models.dataset import (  # noqa: E402
    QUALITY_TARGETS,
    build_feature_matrix,
    build_training_table,
)
from nefte.models.quality_model import Z90, SulfurModel  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

MIN_PER_BUCKET = 25          # меньше — метрики шумят сильнее, чем меняются


def bucket_metrics(pred: pd.DataFrame, y: pd.Series, risk: pd.Series,
                   prev: pd.Series, limit: float) -> dict:
    """Метрики одного интервала. None там, где считать не из чего."""
    err = pred["q50"] - y
    over = (y > limit).astype(int)
    # Покрытие считаем ТОЙ ЖЕ формулой, что и сама модель в interval_metrics:
    # через q50 ± z·sigma с конформной поправкой, а не по сырым q10/q90. Иначе
    # число в этой таблице не сходится с числом в отчёте модели, и читатель решает,
    # что одно из них врёт.
    half = Z90 * pred["sigma"]
    inside = ((y >= pred["q50"] - half) & (y <= pred["q50"] + half)).mean()

    auc = None
    if over.nunique() > 1:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(over, risk))

    prev_mae = None
    valid = prev.notna()
    if valid.sum() >= 5:
        prev_mae = float((prev[valid] - y[valid]).abs().mean())

    return {
        "анализов": int(len(y)),
        "среднее факта": round(float(y.mean()), 2),
        "превышений": int(over.sum()),
        "частота превышений": round(float(over.mean()), 3),
        "MAE": round(float(err.abs().mean()), 3),
        "смещение": round(float(err.mean()), 3),
        "покрытие 80%": round(float(inside), 3),
        "ROC-AUC": None if auc is None else round(auc, 3),
        "заявленный риск": round(float(risk.mean()), 3),
        "MAE персистенции": None if prev_mae is None else round(prev_mae, 3),
        # Разрыв с персистенцией — то, ради чего модель вообще нужна. Абсолютная
        # ошибка зависит от того, трудный период или лёгкий, а разрыв — нет.
        "выигрыш у персистенции": (None if prev_mae is None
                                   else round(prev_mae - float(err.abs().mean()), 3)),
    }


def age_or_level(err: pd.Series, age: pd.Series, prev: pd.Series) -> dict:
    """Смещение от ВОЗРАСТА модели или от УРОВНЯ серы? Они спутаны.

    По кварталам уровень серы падает с возрастом (9.28 → 8.06), и поквартальное
    смещение одинаково хорошо ложится и на возраст, и на уровень. Если причина —
    уровень, то «модель стареет» неверно, а верно «модель занижает, когда серы
    много», и переобучение этого не лечит.

    Разделяем на уровне отдельных анализов. Уровень берём по ПРЕДЫДУЩЕМУ анализу:
    сама ошибка ``прогноз − факт`` связана с фактом по построению, а с прошлым
    анализом механической связи у неё нет.

    Два устойчивых способа, потому что первый, обычная регрессия, обманул: выбросы
    до 46 мг/кг расплющили наклон по уровню до t = −0.8, и уровень выглядел
    непричастным. Медианы в таблице 2×2 и регрессия по рангу уровня с обрезанной
    ошибкой выбросов не боятся.
    """
    ok = prev.notna() & err.notna()
    err, age, prev = err[ok], age[ok], prev[ok]
    if len(err) < 40:
        return {}
    e = err.clip(*err.quantile([0.01, 0.99]))
    rank = prev.rank(pct=True)
    A = np.column_stack([np.ones(len(e)), age.to_numpy(), rank.to_numpy()])
    beta, *_ = np.linalg.lstsq(A, e.to_numpy(), rcond=None)
    resid = e.to_numpy() - A @ beta
    cov = resid @ resid / (len(e) - 3) * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))

    old, high = age >= age.median(), prev >= prev.median()
    cells = {}
    for a_name, am in (("моложе", ~old), ("старше", old)):
        for l_name, lm in (("низкий уровень", ~high), ("высокий уровень", high)):
            m = am & lm
            cells[f"{a_name}, {l_name}"] = {"медиана смещения": round(float(err[m].median()), 3),
                                            "анализов": int(m.sum())}
    return {
        "анализов": int(len(e)),
        "граница возраста, мес": round(float(age.median()), 1),
        "граница уровня, мг/кг": round(float(prev.median()), 2),
        "таблица 2×2": cells,
        "возраст: наклон в месяц": round(float(beta[1]), 4),
        "возраст: t": round(float(beta[1] / se[1]), 1),
        "уровень: наклон на ранг": round(float(beta[2]), 3),
        "уровень: t": round(float(beta[2] / se[2]), 1),
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=0.0)
    ap.add_argument("--target", default="sulfur", choices=sorted(QUALITY_TARGETS))
    ap.add_argument("--freq", default="QE",
                    help="интервал разбиения: QE — кварталы, ME — месяцы")
    args = ap.parse_args()

    cfg = load_config()
    model = SulfurModel.load(SulfurModel.default_path(args.horizon, args.target))
    limit = model.limit
    train_end = pd.Timestamp(cfg["split"]["train"][1]) + pd.Timedelta(days=1)

    X, y = build_training_table(horizon_hours=args.horizon,
                                features=build_feature_matrix(),
                                train_bounds=tuple(cfg["split"]["train"]),
                                target=args.target)
    # всё, что ПОСЛЕ обучения: валидация и тест на одной оси возраста модели
    after = X.index >= train_end
    X, y = X[after], y[after]
    print(f"модель: горизонт {args.horizon:g} ч, обучение кончается "
          f"{train_end - pd.Timedelta(days=1):%d.%m.%Y}")
    print(f"после обучения: {len(y)} анализов, "
          f"{y.index[0]:%d.%m.%Y} … {y.index[-1]:%d.%m.%Y}")

    pred = model.predict_frame(X)
    risk = model.predict_risk(X)
    prev = y.shift(1)

    rows = []
    for period, group in y.groupby(pd.Grouper(freq=args.freq)):
        if len(group) < MIN_PER_BUCKET:
            continue
        idx = pd.DatetimeIndex(group.index)
        age = (idx.to_series().mean() - train_end).days / 30.44
        row = {"интервал": f"{period:%Y-%m}", "возраст, мес": round(age, 1)}
        row.update(bucket_metrics(pred.loc[idx], y.loc[idx], risk.loc[idx],
                                  prev.loc[idx], limit))
        rows.append(row)

    frame = pd.DataFrame(rows)
    print(f"\nПо интервалам (возраст — сколько месяцев прошло с конца обучения)\n")
    print(frame.to_string(index=False))

    # Тренды считаем по возрасту, а не по номеру интервала: интервалы разной
    # наполненности, и «на третьем стало хуже» ничего не значит без шкалы времени.
    trends = {}
    for column in ("MAE", "смещение", "покрытие 80%", "частота превышений",
                   "MAE персистенции", "выигрыш у персистенции", "среднее факта"):
        sub = frame[["возраст, мес", column]].dropna()
        if len(sub) >= 3:
            slope = float(np.polyfit(sub["возраст, мес"], sub[column], 1)[0])
            trends[column] = round(slope, 4)

    print("\nНаклон по возрасту (единиц в месяц):")
    for key, slope in trends.items():
        print(f"   {key:22s} {slope:+.4f}")

    print("\nВывод.")
    verdict = []
    model_slope = trends.get("MAE")
    gain_slope = trends.get("выигрыш у персистенции")

    if model_slope is not None:
        verdict.append(
            f"Абсолютная ошибка модели с возрастом почти не меняется "
            f"({model_slope:+.4f} мг/кг в месяц)."
            if abs(model_slope) <= 0.005 else
            f"Абсолютная ошибка модели растёт на {model_slope:+.4f} в месяц.")

    # Абсолютная ошибка обманчива: периоды бывают лёгкие и трудные, и на лёгком
    # стареющая модель выглядит здоровой. Смотреть надо на РАЗРЫВ с персистенцией —
    # на то, ради чего модель вообще нужна.
    if gain_slope is not None:
        first_gain = float(frame["выигрыш у персистенции"].dropna().iloc[0])
        if gain_slope < -0.005:
            months = abs(first_gain / gain_slope) if gain_slope else None
            verdict.append(
                f"А вот ВЫИГРЫШ у персистенции тает: {gain_slope:+.4f} мг/кг в "
                f"месяц при начальном {first_gain:+.2f}. "
                + (f"Такими темпами модель сравняется с «взять прошлый анализ» "
                   f"примерно через {months:.0f} мес. после обучения. "
                   if months else "")
                + "Это и есть практический ответ про переобучение: смотреть надо "
                  "не на рост ошибки, а на исчезновение преимущества.")
        else:
            verdict.append(
                f"Выигрыш у персистенции с возрастом не тает ({gain_slope:+.4f} в "
                f"месяц) — на доступном горизонте переобучать незачем.")

    bias_slope = trends.get("смещение")
    level_slope = trends.get("среднее факта")
    if bias_slope is not None and abs(bias_slope) > 0.01:
        line = (f"Смещение уезжает на {bias_slope:+.4f} мг/кг в месяц — это ранний "
                "признак старения: MAE ещё молчит, а прогноз уже систематически "
                "сдвинут.")
        if level_slope is not None and abs(level_slope) > 0.01:
            share = abs(bias_slope / level_slope)
            line += (f" Причина видна рядом: сам уровень серы едет на "
                     f"{level_slope:+.4f} в месяц, а модель отслеживает лишь "
                     f"{max(0.0, 1 - share):.0%} этого движения — остальное "
                     "превращается в смещение.")
        verdict.append(line)

    cov_slope = trends.get("покрытие 80%")
    if cov_slope is not None and cov_slope < -0.003:
        verdict.append(
            f"Покрытие интервала падает на {cov_slope:+.4f} в месяц. Интервал — "
            "это то, из чего считается вероятность нарушения, так что стареет не "
            "только точка прогноза, но и мера уверенности в ней.")
    rate_slope = trends.get("частота превышений")
    if rate_slope is not None and abs(rate_slope) > 0.002:
        verdict.append(
            f"Частота превышений сама меняется на {rate_slope:+.4f} в месяц. Это "
            "дрейф ДАННЫХ, а не модели, но он двигает рабочую точку: порог, "
            "подобранный на одном окне, на другом означает не то же самое.")
    # Ради чего всё считалось: назвать срок, а не сказать «дрейф есть».
    #
    # Срок берём по СМЕЩЕНИЮ, а не по MAE, и вот почему. MAE тут почти не растёт,
    # но смещение вниз опаснее роста ошибки: систематически заниженный прогноз
    # серы означает, что продукт выглядит чище, чем он есть, и система реже
    # вмешивается. Ровно этим уже отличились монотонные ограничения
    # (docs/HARD_CHECKS.md §8.7), только там смещение было постоянным, а здесь
    # накапливается со временем.
    #
    # Допустимое смещение — 5 % предела. Это НАШ выбор, не измерение: в пакете нет
    # требования к точности виртуального анализатора. Число видно здесь, и на
    # защите его надо называть допущением.
    tolerance = 0.05 * limit
    shelf: dict = {}
    if bias_slope and abs(bias_slope) > 1e-6 and len(rows) >= 2:
        ages = np.array([r["возраст, мес"] for r in rows], dtype=float)
        biases = np.array([r["смещение"] for r in rows], dtype=float)
        slope, intercept = np.polyfit(ages, biases, 1)

        # ТУТ БЫЛА ОШИБКА, и её стоит назвать. Срок считался как
        # tolerance / |наклон| — время, за которое смещение проходит допустимые
        # 0.5 мг/кг ОТ НУЛЯ. Но в нуле оно не начинается: подгонка даёт
        # пересечение около -0.84, то есть на молодой модели прогноз ЗАНИЖЕН, а с
        # возрастом смещение идёт вверх, проходит ноль и только потом уходит в
        # плюс. По модулю оно сначала УБЫВАЕТ.
        #
        # Из-за пропущенного пересечения срок годности оказывался втрое короче
        # реального, и карточка оператора резала уверенность сильнее всего ровно
        # там, где измеренное смещение наименьшее.
        def crossing(level: float) -> float | None:
            if abs(slope) < 1e-9:
                return None
            return float((level - intercept) / slope)

        up, down = crossing(tolerance), crossing(-tolerance)
        inside = [t for t in (up, down) if t is not None and t > ages.max()]
        measured_to = float(ages.max())
        shelf = {
            "подгонка": {"пересечение": float(intercept), "наклон": float(slope)},
            "смещение_по_модулю_убывает": bool(abs(biases[-1]) < abs(biases[0])),
            "измерено_до_мес": measured_to,
            "выход_за_допуск_вверх_мес": up,
            "выход_за_допуск_вниз_мес": down,
            "допуск": float(tolerance),
        }
        verdict.append(
            f"Смещение подгоняется прямой {intercept:+.3f} {slope:+.4f}*возраст. "
            f"Допустимые ±{tolerance:.2f} мг/кг (5 % предела, ДОПУЩЕНИЕ) она "
            f"пересекает вверх на {up:.1f} мес и вниз на {down:.1f} мес; "
            f"внутри допуска модель с {max(down, 0):.1f} по {up:.1f} мес.")
        verdict.append(
            f"Считать срок как {tolerance:.2f}/наклон было бы ошибкой: это время "
            f"прохождения допуска ОТ НУЛЯ, а смещение начинается с "
            f"{intercept:+.3f}. По модулю оно сначала убывает "
            f"({abs(biases[0]):.3f} на {ages.min():.1f} мес против "
            f"{abs(biases[-1]):.3f} на {ages.max():.1f} мес).")
        verdict.append(
            f"Поэтому срок годности берём по КРАЮ ИЗМЕРЕННОГО, а не по "
            f"экстраполяции: проверено до {measured_to:.1f} мес, и на всём этом "
            f"промежутке качество решений не ухудшается. Дальше мы не знаем — и "
            f"уверенность должна падать именно от незнания, а не от измеренной "
            f"порчи." + (f" Экстраполяция обещала бы {inside[0]:.1f} мес."
                         if inside else ""))

    split = age_or_level(pred["q50"] - y,
                         pd.Series((y.index - train_end).days / 30.44, index=y.index),
                         prev)
    if split:
        print(chr(10) + "Возраст или уровень (медиана смещения, в скобках анализов):")
        for key, cell in split["таблица 2×2"].items():
            print(f"   {key:32s} {cell['медиана смещения']:+.3f} ({cell['анализов']})")
        print(f"   регрессия: возраст {split['возраст: наклон в месяц']:+.4f}/мес "
              f"(t={split['возраст: t']:+.1f}), ранг уровня "
              f"{split['уровень: наклон на ранг']:+.3f} (t={split['уровень: t']:+.1f})")
        age_real = abs(split["возраст: t"]) >= 2
        level_real = abs(split["уровень: t"]) >= 2
        verdict.append(
            ("Подозрение, что «старение» — это спутанный с возрастом уровень серы, "
             "проверено на отдельных анализах. "
             + ("Возраст устоял: " if age_real else "Возраст НЕ устоял: ")
             + f"t={split['возраст: t']:+.1f} при учёте уровня. ")
            + ("Но и уровень оказался отдельной причиной "
               f"(t={split['уровень: t']:+.1f}): модель занижает, когда серы много, "
               "независимо от возраста, и переобучение этого не лечит."
               if level_real else "Уровень отдельной причиной не оказался."))

    for line in verdict:
        print("  " + line)

    out = ROOT / "reports" / f"drift_h{args.horizon:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        **report_provenance(cfg),
        "горизонт": args.horizon, "показатель": args.target, "предел": limit,
        "конец_обучения": str(train_end - pd.Timedelta(days=1)),
        "интервалы": rows, "наклоны_в_месяц": trends, "срок_годности": shelf,
        "возраст_или_уровень": split,
        "вывод": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
