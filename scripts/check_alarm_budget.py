# -*- coding: utf-8 -*-
"""Держит ли модель обещанный бюджет тревог — и что с этим делает скользящий порог.

    python scripts/check_alarm_budget.py --horizon 2
    python scripts/check_alarm_budget.py --horizon 0 --model seq

Порог тревоги подбирается по бюджету НА ВАЛИДАЦИИ: берётся квантиль риска такой,
чтобы система вмешивалась не чаще заданной доли моментов. Приём молчаливо
предполагает, что распределение риска на тесте будет тем же самым. Для бустинга
это так, для нейросети — нет: её средний риск между валидацией и тестом
удваивается, и тот же порог означает вмешательство в две трети спокойных моментов
вместо трети.

Лечится скользящим порогом: тот же квантиль, но по собственному выходу модели за
прошедшие 30 суток. Меток не требует — лаборатория приходит раз в сутки с
задержкой, а риск известен сразу, — поэтому приём применим в эксплуатации.

Скрипт существует потому, что числа из этой таблицы стояли в документации, не
будучи воспроизводимыми ни одним счётом. В проекте это была единственная такая
таблица, и разница между «померили один раз» и «меряется командой» вся в том,
переживёт ли вывод следующий пересчёт матрицы.

Результат: reports/alarm_budget_<модель>_h<горизонт>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from nefte.config import ROOT, load_config                      # noqa: E402
from nefte.data.features import time_split                      # noqa: E402
from nefte.models.dataset import (                              # noqa: E402
    QUALITY_TARGETS,
    build_feature_matrix,
    build_training_table,
)
from nefte.models.quality_model import (                        # noqa: E402
    SulfurModel,
    rolling_budget_threshold,
)
from nefte.models.sequence import SulfurSequenceModel            # noqa: E402
from nefte.utils import use_utf8_console                        # noqa: E402


def outcomes(risk: pd.Series, actual: pd.Series,
             threshold: pd.Series | float) -> dict:
    """Доля тревог, точность и полнота при таком пороге.

    Порог может быть числом (фиксированный) или рядом (скользящий). Моменты, где
    скользящий порог ещё не определён — первые недели окна, — исключаются из
    ОБОИХ вариантов, иначе сравнение идёт на разных выборках и разница в доле
    тревог окажется артефактом длины ряда, а не свойством порога.
    """
    if isinstance(threshold, pd.Series):
        usable = threshold.notna()
        thr = threshold[usable]
    else:
        usable = pd.Series(True, index=risk.index)
        thr = float(threshold)
    r, a = risk[usable], actual[usable]
    alarm = r > thr
    hits = int((alarm & (a == 1)).sum())
    return {
        "n": int(len(r)),
        "доля_тревог": float(alarm.mean()) if len(r) else float("nan"),
        "precision": float(hits / alarm.sum()) if alarm.sum() else float("nan"),
        "recall": float(hits / (a == 1).sum()) if (a == 1).sum() else float("nan"),
        "превышений": int((a == 1).sum()),
    }


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--target", default="sulfur", choices=sorted(QUALITY_TARGETS))
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--min-periods", type=int, default=20)
    ap.add_argument("--model", default="boost", choices=("boost", "seq"),
                    help="какая модель даёт риск: бустинг или сеть")
    ap.add_argument("--arch", default="tcn", help="архитектура сети (для --model seq)")
    ap.add_argument("--window", type=int, default=48, help="окно сети")
    ap.add_argument("--pretrain", action="store_true", help="сеть с предобучением")
    args = ap.parse_args()

    cfg = load_config()
    budget = float(cfg["quality"]["alarm_budget"])
    features = build_feature_matrix()
    if args.model == "seq":
        path = SulfurSequenceModel.default_path(args.horizon, args.arch,
                                                args.window, args.pretrain)
        model = SulfurSequenceModel.load(path).attach(features)
    else:
        model = SulfurModel.load(SulfurModel.default_path(args.horizon, args.target))
    limit = model.limit

    X, y = build_training_table(horizon_hours=args.horizon,
                                features=features,
                                train_bounds=tuple(cfg["split"]["train"]),
                                target=args.target)
    masks = time_split(X.index, cfg)

    name = ("сеть %s%d%s" % (args.arch, args.window, " + предобучение" if args.pretrain else "")
            if args.model == "seq" else "бустинг")
    print(f"модель: {name}, горизонт {args.horizon:g} ч, предел {limit:g}, "
          f"бюджет тревог {budget:.0%}, фиксированный порог "
          f"{model.alarm_threshold:.4f} (подобран на валидации)")

    report = {
        "модель": name,
        "горизонт": args.horizon, "показатель": args.target,
        "бюджет": budget, "фиксированный_порог": float(model.alarm_threshold),
        "окно_суток": args.window_days, "выборки": {},
    }

    for split in ("val", "test"):
        mask = masks[split].to_numpy()
        Xp, yp = X[mask], y[mask]
        if not len(yp) or (yp > limit).nunique() < 2:
            continue
        actual = (yp > limit).astype(int)
        # У сети вход — индекс: окно она собирает сама из присоединённой матрицы.
        risk = (model.predict_risk(Xp.index) if args.model == "seq"
                else model.predict_risk(Xp))
        risk = pd.Series(np.asarray(risk, dtype=float), index=Xp.index)
        keep = risk.notna()
        risk, actual = risk[keep], actual[keep]

        rolling = rolling_budget_threshold(risk, budget,
                                           window_days=args.window_days,
                                           min_periods=args.min_periods)
        fixed_all = outcomes(risk, actual, model.alarm_threshold)
        # то же сравнение на общей выборке: там, где скользящий порог определён
        common = rolling.notna()
        fixed_common = outcomes(risk[common], actual[common], model.alarm_threshold)
        roll = outcomes(risk, actual, rolling)

        block = {
            "средний_риск": float(risk.mean()),
            "фиксированный": fixed_all,
            "фиксированный_на_общей": fixed_common,
            "скользящий": roll,
        }
        report["выборки"][split] = block

        print(f"\n=== {split}: {len(yp)} анализов, превышений "
              f"{int(actual.sum())}, средний риск {risk.mean():.3f}")
        print("  %-28s %10s %10s %10s %6s" % ("", "тревог", "precision", "recall", "n"))
        for name, block_key in (("фиксированный порог", "фиксированный"),
                                ("он же на общей выборке", "фиксированный_на_общей"),
                                ("скользящий порог", "скользящий")):
            b = block[block_key]
            print("  %-28s %9.1f%% %10.3f %10.3f %6d"
                  % (name, 100 * b["доля_тревог"], b["precision"], b["recall"], b["n"]))

    val = report["выборки"].get("val", {})
    test = report["выборки"].get("test", {})
    if val and test:
        shift = test["средний_риск"] / max(val["средний_риск"], 1e-9)
        report["сдвиг_среднего_риска"] = float(shift)
        kept = test["фиксированный"]["доля_тревог"] / budget
        report["перерасход_бюджета"] = float(kept)
        print(f"\nсредний риск тест/валидация: {shift:.2f}")
        print(f"фактическая доля тревог на тесте против обещанной: "
              f"{test['фиксированный']['доля_тревог']:.1%} против {budget:.0%} "
              f"({kept:.2f}x)")
        if kept > 1.3:
            print("  Бюджет НЕ выдержан фиксированным порогом: распределение риска "
                  "уехало. Скользящий порог здесь и нужен.")
        else:
            print("  Бюджет выдержан: распределение риска стабильно, скользящий "
                  "порог не требуется и добавляет шума.")

    report["feature_version"] = __import__("nefte.models.dataset",
                                           fromlist=["FEATURE_VERSION"]).FEATURE_VERSION
    suffix = f"_{args.arch}{args.window}" + ("_pre" if args.pretrain else "")         if args.model == "seq" else ""
    out = ROOT / "reports" / f"alarm_budget_{args.model}{suffix}_h{args.horizon:g}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nзаписано:", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
