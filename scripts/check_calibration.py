"""Честна ли вероятность нарушения, которой система принимает решения.

    python scripts/check_calibration.py
    python scripts/check_calibration.py --horizon 2 --bins 5

Оркестратор сравнивает вероятность превышения с порогом и по результату решает,
вмешиваться или нет. Бюджет тревог тоже задан в терминах вероятности. То есть всё
решающее правило стоит на предположении, что **0.2 действительно означает
«случится примерно в одном случае из пяти»**.

Это предположение до сих пор никто не проверял. В отчёте модели есть ROC-AUC,
PR-AUC и покрытие интервала — все три меряют РАЗЛИЧЕНИЕ, то есть умение
упорядочить моменты по опасности. Калибровка — другое свойство: попадает ли
заявленная вероятность в наблюдаемую частоту. Модель может прекрасно ранжировать
и при этом систематически завышать вероятность вдвое; ROC-AUC этого не заметит,
а оператор заметит — по числу ложных тревог.

Поправка Платта у нас подбирается на ВАЛИДАЦИИ. Обучающая процедура делает её
честной именно там, а на тесте она может разъехаться — ровно так же, как
разъехалась модель Т95 из-за дрейфа (docs/HARD_CHECKS.md §8.8).

Что считаем
-----------
* **кривая надёжности** — предсказанная вероятность против наблюдаемой частоты по
  бинам. Главное здесь не число, а форма: систематический перекос вверх или вниз;
* **Brier** — средний квадрат ошибки вероятности. Сравнивается с «всегда базовая
  частота»: модель, проигравшая константе, вероятностью не является;
* **ECE** — средний по бинам разрыв между заявленным и наблюдённым, взвешенный
  числом точек. Одно число, чтобы следить за регрессом;
* то же самое **без поправки Платта** — чтобы видеть, что она вообще делает.

Результат: reports/calibration_h<горизонт>.json.
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
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import (  # noqa: E402
    QUALITY_TARGETS,
    build_feature_matrix,
    build_training_table,
)
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402


def reliability_curve(prob: pd.Series, actual: pd.Series, bins: int) -> pd.DataFrame:
    """Заявленная вероятность против наблюдаемой частоты.

    Бины по квантилям, а не равной ширины: событий мало, и при равной ширине
    верхние бины оказываются пустыми, а таблица — красивой и бессмысленной.
    """
    frame = pd.DataFrame({"p": prob.to_numpy(), "y": actual.to_numpy()})
    try:
        frame["bin"] = pd.qcut(frame["p"], bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    out = frame.groupby("bin", observed=True).agg(
        моментов=("y", "size"),
        заявлено=("p", "mean"),
        наблюдалось=("y", "mean"),
    ).reset_index(drop=True)
    out["разрыв"] = out["наблюдалось"] - out["заявлено"]
    # Доверительный интервал Вилсона. Без него таблицу читают как точную, а в
    # бине полсотни точек и три события: наблюдаемая частота 0.06 совместима со
    # значениями от 0.01 до 0.17, и «разрыв −0.11» может оказаться ничем.
    lo, hi = [], []
    for _, row in out.iterrows():
        a, b = wilson(row["наблюдалось"], int(row["моментов"]))
        lo.append(a)
        hi.append(b)
    out["набл. от"], out["набл. до"] = lo, hi
    # заявленное попало в интервал наблюдаемого — расхождения не доказано
    out["согласуется"] = (out["заявлено"] >= out["набл. от"]) & (out["заявлено"] <= out["набл. до"])
    return out


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % интервал Вилсона для доли. Корректен при малых n и p у края."""
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def expected_calibration_error(curve: pd.DataFrame) -> float:
    """Средний разрыв, взвешенный числом точек в бине."""
    if not len(curve):
        return float("nan")
    weight = curve["моментов"] / curve["моментов"].sum()
    return float((weight * curve["разрыв"].abs()).sum())


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=0.0)
    ap.add_argument("--target", default="sulfur", choices=sorted(QUALITY_TARGETS))
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config()
    model = SulfurModel.load(SulfurModel.default_path(args.horizon, args.target))
    limit = model.limit
    print(f"модель: горизонт {args.horizon:g} ч, предел {limit:g}, "
          f"источник риска «{model.risk_source}», "
          f"поправка Платта {'есть' if model.risk_calibration else 'НЕТ'}")

    X, y = build_training_table(horizon_hours=args.horizon,
                                features=build_feature_matrix(),
                                train_bounds=tuple(cfg["split"]["train"]),
                                target=args.target)
    masks = time_split(X.index, cfg)

    report: dict = {"горизонт": args.horizon, "показатель": args.target,
                    "предел": limit, "источник_риска": model.risk_source,
                    "выборки": {}}

    for split in ("val", "test"):
        mask = masks[split].to_numpy()
        Xp, yp = X[mask], y[mask]
        if not len(yp) or (yp > limit).nunique() < 2:
            continue
        actual = (yp > limit).astype(int)
        base = float(actual.mean())

        block = {"n": int(len(yp)), "базовая_частота": round(base, 4)}
        print(f"\n=== {split}: {len(yp)} анализов, превышений "
              f"{int(actual.sum())} ({base:.1%})")

        for name, raw in (("с поправкой Платта", False), ("без поправки", True)):
            prob = model.predict_risk(Xp, raw=raw)
            brier = float(((prob - actual) ** 2).mean())
            brier_const = float(((base - actual) ** 2).mean())
            curve = reliability_curve(prob, actual, args.bins)
            ece = expected_calibration_error(curve)

            print(f"\n  {name}:")
            if len(curve):
                shown = curve.copy()
                for col in ("заявлено", "наблюдалось", "разрыв",
                            "набл. от", "набл. до"):
                    shown[col] = shown[col].round(3)
                print(shown.to_string(index=False))
                disagree = curve[~curve["согласуется"]]
                if len(disagree):
                    print(f"    Расхождение статистически значимо в "
                          f"{len(disagree)} бине(ах) из {len(curve)}.")
                else:
                    print("    Ни в одном бине расхождение не выходит за "
                          "доверительный интервал — форма кривой не опровергнута.")
            print(f"    Brier {brier:.4f} против {brier_const:.4f} у «всегда базовая "
                  f"частота» — {'лучше' if brier < brier_const else 'ХУЖЕ'}")
            print(f"    ECE {ece:.4f}; средняя заявленная {prob.mean():.3f} "
                  f"против наблюдаемой {base:.3f}")

            block["платт" if not raw else "сырой"] = {
                "Brier": round(brier, 5),
                "Brier_константы": round(brier_const, 5),
                "лучше_константы": bool(brier < brier_const),
                "ECE": None if ece != ece else round(ece, 5),
                "средняя_заявленная": round(float(prob.mean()), 4),
                "кривая": [] if not len(curve) else [
                    {k: (int(v) if k == "моментов"
                         else bool(v) if k == "согласуется"
                         else round(float(v), 4))
                     for k, v in row.items()}
                    for row in curve.to_dict("records")],
            }
        report["выборки"][split] = block

    # Главный вопрос: сохраняется ли калибровка за пределами валидации, на которой
    # её подбирали. Если нет — порог и бюджет тревог означают не то, что написано.
    val, test = report["выборки"].get("val"), report["выборки"].get("test")
    if val and test:
        print("\nВывод.")
        drift = test["платт"]["средняя_заявленная"] - test["базовая_частота"]
        print(f"  На валидации, где подбиралась поправка: заявленная "
              f"{val['платт']['средняя_заявленная']:.3f} при частоте "
              f"{val['базовая_частота']:.3f}.")
        print(f"  На тесте: заявленная {test['платт']['средняя_заявленная']:.3f} "
              f"при частоте {test['базовая_частота']:.3f} "
              f"(перекос {drift:+.3f}).")
        if abs(drift) > 0.05:
            print("  Перекос заметный: порог вмешательства и бюджет тревог означают "
                  "на тесте не то, что на валидации. Это надо назвать на защите и "
                  "не выдавать вероятность за точную.")
        else:
            print("  Перекос в пределах 0.05: вероятность можно показывать "
                  "оператору как вероятность.")
        if not test["платт"]["лучше_константы"]:
            print("  ВНИМАНИЕ: на тесте модель проигрывает по Brier константе "
                  "«всегда базовая частота». Как ВЕРОЯТНОСТЬ такой выход "
                  "использовать нельзя, даже если он хорошо ранжирует.")

    out = ROOT / "reports" / f"calibration_h{args.horizon:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
