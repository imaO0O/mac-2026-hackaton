"""Транспортное запаздывание «уставка → сера» — по данным, а не по допущению.

    python scripts/find_delays.py
    python scripts/find_delays.py --max-lag-hours 3 --smooth 3

Организаторы ответили, что величина запаздывания между управляющим воздействием и
откликом качества в пакете не задана, лежит в пределах 0–3 часов и найти её надо
эмпирически. До сих пор она у нас была допущением: постоянная времени отклика в
имитационной среде стояла 4 часа «на глаз», а признаки строились скользящими
окнами 1/6/24 ч без всякой опоры на измеренное запаздывание.

Метод
-----
1. Отклик берём с ПОТОЧНОГО анализатора (ПАК, шаг 10 мин), а не с ЛИМС: у ЛИМС
   один анализ в сутки, на такой сетке трёхчасовое запаздывание неразличимо в
   принципе.
2. Считаем на ПЕРВЫХ РАЗНОСТЯХ. Уровни и уставки, и серы дрейфуют месяцами
   (кампания катализатора, сезон), их корреляция говорит про общий тренд, а не
   про отклик. Разности убирают тренд — это стандартное «отбеливание» перед
   взаимной корреляцией.
3. Считаем ТОЛЬКО на train. Запаздывание — такой же параметр модели, как веса, и
   подбирать его на тесте нельзя.
4. Выкидываем остановы и «замороженный» ПАК: на остановленной установке отклик
   не наблюдается, а залипший анализатор даёт нулевые разности и завышает
   корреляцию на всех лагах одинаково.
5. Ищем лаг в ОБЕ стороны, а не только вперёд. Установка управляется оператором:
   он поднимает температуру, УВИДЕВ рост серы. Такая обратная причинность даёт
   максимум корреляции на ОТРИЦАТЕЛЬНОМ лаге, и если её не проверить, обратную
   связь оператора легко принять за отклик процесса. Ищем в обе стороны и прямо
   говорим, какая из двух картин сильнее.

Что получаем
------------
* ``lag_hours`` — чистое запаздывание по максимуму |корреляции| разностей;
* ``tau_hours`` — постоянная времени по авторегрессии первого порядка
  ``s[t] = a·s[t-1] + b·u[t-L] + c``, откуда ``tau = -Δt / ln a``. Это то самое
  число, которое имитационная среда использует как скорость отклика. ВАЖНО: это
  постоянная времени КАНАЛА СЕРЫ (реактор + смешение + анализатор), а не каждого
  входа по отдельности. Коэффициент ``a`` определяется инерцией самого ряда серы,
  поэтому по всем тегам получается практически одно и то же число — это ОДНА
  оценка, а не семь независимых подтверждений;
* знак реакции: рост реакторной температуры обязан СНИЖАТЬ серу. Если знак
  обратный, тег опознан неверно, и это надо увидеть, а не спрятать.

Результат: reports/delays.json.
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
from nefte.data.cleaning import frozen_mask  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.data.loaders import load_pak, load_telemetry  # noqa: E402
from nefte.data.validity import SignalValidity  # noqa: E402
from nefte.models.regime import FEED, outage_mask  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

STEP_MINUTES = 10

# Управляющие теги, которые оптимизатор реально двигает, плюс расход сырья:
# именно про них организаторы спрашивали про запаздывание.
CONTROLS = {
    "T5":  "Р-201: температура ГСС на выходе",
    "T6":  "Р-202: температура",
    "T11": "Реакторный блок: температура",
    "P13": "Давление в реакторном блоке",
    "F26": "Расход сырья на установку",
    "F15": "Расход квенча в Р-202",
    "P24": "Расход свежего ВСГ",
}

# Ожидаемый знак реакции серы на РОСТ уставки. None — знание отсутствует.
EXPECTED_SIGN = {"T5": -1, "T6": -1, "T11": -1, "P13": -1, "F26": +1,
                 "F15": +1, "P24": -1}


def prepare(smooth_steps: int) -> tuple[pd.DataFrame, pd.Series]:
    """Телеметрия и сера ПАК на общей 10-минутной сетке, только train, без брака."""
    cfg = load_config()
    # чистим тем же единственным путём, что и признаки: две реализации одного
    # правила разъезжаются, и в прошлый раз разъехались на 0.9 % значений
    ht = SignalValidity.build(load_telemetry("ht"), unit="ht").clean
    pak = load_pak()["sulfur_ppm"].sort_index()

    grid = ht.index
    pak = pak.reindex(grid).astype("float64")

    # брак: останов установки и залипший анализатор
    bad = pd.Series(False, index=grid)
    if FEED in ht.columns:
        bad |= outage_mask(ht[FEED]).reindex(grid).fillna(True)
    bad |= frozen_mask(pak).reindex(grid).fillna(True)

    masks = time_split(grid, cfg)
    keep = masks["train"].to_numpy() & ~bad.to_numpy()

    if smooth_steps > 1:
        # лёгкое сглаживание: шум анализатора на 10-минутной сетке сопоставим с
        # полезным сигналом, без него максимум взаимной корреляции размазан
        ht = ht.rolling(smooth_steps, min_periods=smooth_steps).mean()
        pak = pak.rolling(smooth_steps, min_periods=smooth_steps).mean()

    ht = ht.where(pd.Series(keep, index=grid), np.nan)
    pak = pak.where(pd.Series(keep, index=grid), np.nan)
    return ht, pak


def cross_correlation(control: pd.Series, response: pd.Series,
                      max_lag: int) -> pd.Series:
    """corr(Δu[t], Δs[t+lag]) для lag от -max_lag до +max_lag шагов.

    Отрицательный лаг означает «сера изменилась РАНЬШЕ уставки», то есть реакцию
    оператора, а не реакцию процесса. Без этой половины оси одно легко принять за
    другое.
    """
    du = control.diff()
    ds = response.diff()
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = ds.shift(-lag)
        both = du.notna() & shifted.notna()
        if both.sum() < 500:
            continue
        out[lag] = float(np.corrcoef(du[both], shifted[both])[0, 1])
    return pd.Series(out, dtype="float64")


def first_order_fit(control: pd.Series, response: pd.Series,
                    lag: int) -> tuple[float, float]:
    """``s[t] = a·s[t-1] + b·u[t-lag] + c`` методом наименьших квадратов.

    Возвращает (a, tau_hours). Постоянная времени ``tau = -Δt/ln a`` — время, за
    которое отклик проходит 63% пути к новому равновесию. Если a вне (0, 1),
    процесс на этих данных не описывается затухающим звеном первого порядка, и
    возвращается NaN, а не подогнанное число.
    """
    frame = pd.DataFrame({
        "s": response,
        "s_prev": response.shift(1),
        "u": control.shift(lag),
    }).dropna()
    if len(frame) < 500:
        return float("nan"), float("nan")

    design = np.column_stack([frame["s_prev"], frame["u"], np.ones(len(frame))])
    coef, *_ = np.linalg.lstsq(design, frame["s"].to_numpy(), rcond=None)
    a = float(coef[0])
    if not 0.0 < a < 1.0:
        return a, float("nan")
    return a, float(-(STEP_MINUTES / 60.0) / np.log(a))


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lag-hours", type=float, default=3.0,
                    help="верхняя граница поиска; организаторы назвали 0–3 ч")
    ap.add_argument("--smooth", type=int, default=3,
                    help="сглаживание входов, шагов по 10 мин")
    args = ap.parse_args()

    max_lag = int(round(args.max_lag_hours * 60 / STEP_MINUTES))
    ht, pak = prepare(args.smooth)
    usable = int(pak.notna().sum())
    print(f"Пригодных 10-минутных отсчётов на train: {usable}")
    if usable < 1000:
        print("Данных мало — результату верить нельзя.")
        return 1

    rows, curves = [], {}
    for tag, name in CONTROLS.items():
        if tag not in ht.columns:
            continue
        curve = cross_correlation(ht[tag], pak, max_lag)
        if curve.empty:
            continue
        forward = curve[curve.index >= 0]
        backward = curve[curve.index < 0]
        best = int(forward.abs().idxmax())
        corr = float(forward.loc[best])
        # обратная причинность: если «сера раньше уставки» связана сильнее, чем
        # «уставка раньше серы», это контур оператора, а не отклик установки
        back_peak = (float(backward.abs().max()) if len(backward) else 0.0)
        causality = "процесс" if abs(corr) >= back_peak else "оператор"
        a, tau = first_order_fit(ht[tag], pak, max(best, 0))
        expected = EXPECTED_SIGN.get(tag)
        sign_ok = (expected is None or corr == 0
                   or np.sign(corr) == np.sign(expected))
        curves[tag] = {str(round(k * STEP_MINUTES / 60, 2)): round(v, 4)
                       for k, v in curve.items()}
        rows.append({
            "тег": tag,
            "что это": name[:34],
            "лаг, ч": round(best * STEP_MINUTES / 60, 2),
            "corr": round(corr, 4),
            "знак": "ожидаемый" if sign_ok else "ОБРАТНЫЙ",
            "кто первый": causality,
            "tau, ч": None if tau != tau else round(tau, 2),
        })

    frame = pd.DataFrame(rows).sort_values("corr", key=lambda s: s.abs(),
                                           ascending=False)
    print("\nЗапаздывание отклика серы на изменение уставки (train, ПАК)\n")
    print(frame.to_string(index=False))

    strong = frame[frame["corr"].abs() >= 0.05]
    if len(strong):
        lag_hours = float(np.median(strong["лаг, ч"]))
        taus = [t for t in strong["tau, ч"] if t is not None]
        tau_hours = float(np.median(taus)) if taus else float("nan")
    else:
        lag_hours, tau_hours = float("nan"), float("nan")

    print(f"\nМедиана по тегам с заметной реакцией (|corr| >= 0.05): "
          f"запаздывание {lag_hours:.2f} ч, постоянная времени "
          f"{tau_hours:.2f} ч." if lag_hours == lag_hours else
          "\nНи один тег не дал заметной реакции: запаздывание по этим данным "
          "не определяется.")

    feedback = frame[(frame["кто первый"] == "оператор")
                     & (frame["corr"].abs() >= 0.05)]
    if len(feedback):
        print("\nУ этих тегов связь «сера раньше уставки» сильнее обратной: "
              + ", ".join(feedback["тег"]) + ". Это контур оператора, а не отклик "
              "процесса, и лаг по ним трактовать как транспортное запаздывание "
              "нельзя.")

    print("\nОдна оговорка про tau. Он оценён по инерции самого ряда серы, "
          "поэтому по всем входам выходит почти одинаковым. Это одна оценка "
          "постоянной времени канала, а не семь независимых.")

    wrong = frame[frame["знак"] == "ОБРАТНЫЙ"]
    if len(wrong):
        print("\nЗнак реакции обратный ожидаемому у: "
              + ", ".join(wrong["тег"]) + ". Либо тег опознан неверно, либо "
              "уставкой управлял оператор в ответ на серу, а не наоборот "
              "(обратная причинность). На защите об этом надо сказать прямо.")

    out = ROOT / "reports" / "delays.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "метод": "взаимная корреляция первых разностей, ПАК, только train",
        "шаг_минут": STEP_MINUTES,
        "сглаживание_шагов": args.smooth,
        "отсчётов": usable,
        "оговорка_tau": ("постоянная времени канала серы; по всем входам одна и "
                         "та же, потому что определяется инерцией самого ряда"),
        "по_тегам": rows,
        "кривые_корреляции": curves,
        "итог": {"лаг_часов": None if lag_hours != lag_hours else lag_hours,
                 "tau_часов": None if tau_hours != tau_hours else tau_hours},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
