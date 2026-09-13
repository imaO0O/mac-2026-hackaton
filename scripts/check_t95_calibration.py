"""Калибровка вероятности нарушения по Т95 и поправка к ней. Только CPU.

    python scripts/check_t95_calibration.py

Вероятность нарушения Т95 агент качества считает в нормальном приближении:
оценка Т95 (последний анализ плюс приращение по формуле справочника) и σ по
возрасту анализа. Её видит оператор — в карточке качества, в карточках вариантов
оптимизатора и в тексте тревоги.

Мера — момент отбора каждой пробы Т95. Срез на этот момент знает только
ОПУБЛИКОВАННЫЕ анализы, так что сама проба в нём ещё не видна: это честный
прогноз того, что покажет лаборатория.

Что выяснилось при первом счёте: вероятность завышена в 2–4 раза на всех трёх
периодах и проигрывает по Brier константе «всегда базовая частота» — то есть как
вероятность её показывать нельзя. Упорядочивает она при этом хорошо (ROC-AUC
0.77–0.78). Такое лечится монотонной поправкой.

Правило принятия поправки записано ДО счёта:

* поправка Платта подбирается на ОБУЧАЮЩЕМ периоде;
* принимается, только если на ВАЛИДАЦИИ улучшает Brier и приближает наклон
  калибровки к единице;
* тест — только проверка.

Результат: reports/t95_risk_calibration.json. Агент качества и оптимизатор
читают поправку оттуда, и только если она принята.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.quality import QualityAgent, spec_risk_normal, t95_sigma  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.models.quality_model import calibration_slope  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "t95_risk_calibration.json"


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_platt(risk, over) -> tuple[float, float]:
    """``logit p' = a + b·logit p`` методом Ньютона."""
    z = logit(risk)
    o = np.asarray(over, dtype=float)
    A = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(100):
        q = 1.0 / (1.0 + np.exp(-(A @ w)))
        step = np.linalg.solve((A * (q * (1 - q))[:, None]).T @ A, A.T @ (o - q))
        w = w + step
        if np.abs(step).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def apply_platt(risk, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a + b * logit(risk))))


def summary(risk: np.ndarray, over: np.ndarray) -> dict:
    s = calibration_slope(pd.Series(risk), pd.Series(over), n_boot=1000)
    return {
        "Brier": round(float(((risk - over) ** 2).mean()), 5),
        "Brier_константы": round(float(((over.mean() - over) ** 2).mean()), 5),
        "средняя_заявленная": round(float(risk.mean()), 4),
        "частота": round(float(over.mean()), 4),
        "наклон": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()},
        "анализов": int(len(risk)),
    }


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    sb = StateBuilder(cfg)
    agent = QualityAgent(model=None, cfg=cfg)
    limit = agent.t95_limit

    data: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        lo, hi = cfg["split"][split]
        fact = sb.lims_t95.loc[lo:hi]
        rows = []
        for ts, value in fact.items():
            state = sb.build(ts)
            estimate = agent.t95_fn(state, {}) if agent.t95_fn else None
            if estimate is None or estimate != estimate:
                continue
            meas = state.quality.get("lims_t95_c")
            # СЫРАЯ вероятность — тем же приближением, что и в агенте, но без
            # поправки: иначе после встраивания поправки скрипт подбирал бы её
            # к уже поправленным числам
            raw = spec_risk_normal(float(estimate),
                                   t95_sigma(meas.age_hours if meas else None), limit)
            rows.append({"ts": ts, "риск": raw, "факт": float(value)})
        frame = pd.DataFrame(rows)
        frame["превышение"] = (frame["факт"] > limit).astype(int)
        data[split] = frame
        print(f"{split:5s}: {len(frame)} анализов, превышений {frame['превышение'].sum()} "
              f"({frame['превышение'].mean():.1%}), средняя заявленная {frame['риск'].mean():.3f}")

    a, b = fit_platt(data["train"]["риск"], data["train"]["превышение"])
    print(f"\nПоправка по обучению: logit p' = {a:+.3f} + {b:.3f}·logit p")

    report: dict = {"предел": limit, "поправка": {"a": a, "b": b, "подобрана_на": "train"},
                    "выборки": {}}
    for split in ("train", "val", "test"):
        f = data[split]
        raw = f["риск"].to_numpy()
        fixed = apply_platt(raw, a, b)
        over = f["превышение"].to_numpy()
        report["выборки"][split] = {"как_есть": summary(raw, over),
                                    "с_поправкой": summary(fixed, over)}
        for name, block in report["выборки"][split].items():
            s = block["наклон"]
            print(f"  {split:5s} {name:12s} Brier {block['Brier']:.4f} (константа "
                  f"{block['Brier_константы']:.4f}), средняя {block['средняя_заявленная']:.3f} "
                  f"при частоте {block['частота']:.3f}, наклон {s.get('b', float('nan')):.2f} "
                  f"[{s.get('b_от', float('nan')):.2f}…{s.get('b_до', float('nan')):.2f}]")

    val = report["выборки"]["val"]
    before, after = val["как_есть"], val["с_поправкой"]
    brier_better = after["Brier"] < before["Brier"]
    slope_closer = abs(after["наклон"]["b"] - 1) < abs(before["наклон"]["b"] - 1)
    accepted = bool(brier_better and slope_closer)
    report["принята"] = accepted
    report["правило"] = ("подбор на обучении; принимается, если на валидации Brier лучше "
                         "и наклон калибровки ближе к единице; тест — проверка")
    # Порог заметки в карточке качества был 0.2 по СЫРОЙ вероятности. Поправка
    # монотонна, поэтому тот же набор моментов даёт порог, пересчитанный через неё:
    # заметка появляется там же, где раньше, но с честным числом.
    report["порог_заметки"] = float(apply_platt([0.2], a, b)[0])
    report["частота_на_обучении"] = float(data["train"]["превышение"].mean())

    test = report["выборки"]["test"]
    print(f"\nВывод: поправка {'ПРИНЯТА' if accepted else 'НЕ принята'} "
          f"(валидация: Brier {'лучше' if brier_better else 'не лучше'}, наклон "
          f"{'ближе' if slope_closer else 'не ближе'} к единице).")
    print(f"  Проверка на тесте: Brier {test['как_есть']['Brier']:.4f} → "
          f"{test['с_поправкой']['Brier']:.4f} при константе "
          f"{test['как_есть']['Brier_константы']:.4f}.")
    print(f"  Заметка в карточке: порог {report['порог_заметки']:.3f} по поправленной "
          f"вероятности — те же моменты, что 0.2 по сырой.")

    REPORT.write_text(json.dumps({**report_provenance(cfg), **report}, ensure_ascii=False,
                                 indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
