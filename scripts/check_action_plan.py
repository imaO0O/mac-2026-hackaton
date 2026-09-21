"""Хватает ли одного шага: нужен ли в карточке план из нескольких. Только CPU.

    python scripts/check_action_plan.py

Зачем. Карточка даёт ОДИН шаг и иногда добавляет «может понадобиться ещё». Это
половина ответа: оператор не знает, сколько всего шагов его ждёт, когда проверять
результат и по чему судить. Отклик серы на уставку запаздывает (τ около 4.7 ч),
лаборатория приходит раз в сутки — то есть цикл «подвинул — проверил» длиннее
одного решения по построению.

Считаем по УЖЕ посчитанному прогону валидации, эпизодами: действия подряд с
промежутком не больше 14 ч (3τ) — для оператора это одно вмешательство, а не
десять (так же, как в `scripts/check_intervention_episodes.py`).

**Правило записано ДО счёта.** План показываем, если на ВАЛИДАЦИИ одновременно:

1. одного шага обычно НЕ хватает: доля эпизодов ровно из одного действия меньше
   половины — иначе план обещал бы продолжение, которого чаще всего не будет;
2. длина эпизода предсказуема: разброс числа шагов не настолько дик, чтобы
   называть число, — межквартильный размах не больше медианы.

Не выполняется — измеренный отказ: карточка остаётся с одним шагом и честной
оговоркой «может понадобиться ещё».
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "action_plan.json"
RUNS = {"валидация": "val_period_step1h_lock4.json",
        "тест": "test_period_step1h.json"}
EPISODE_GAP_H = 14.0


def episodes(times: pd.DatetimeIndex, gap_h: float) -> list[list[pd.Timestamp]]:
    """Действия подряд с промежутком не больше gap_h — один эпизод."""
    out: list[list[pd.Timestamp]] = []
    for ts in times:
        if out and (ts - out[-1][-1]).total_seconds() / 3600.0 <= gap_h:
            out[-1].append(ts)
        else:
            out.append([ts])
    return out


def block(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["rows"])
    rows["ts"] = pd.to_datetime(rows["ts"])
    acts = pd.DatetimeIndex(rows.loc[rows["исход"].eq("меняем уставки"), "ts"])
    chains = episodes(acts, EPISODE_GAP_H)
    steps = np.array([len(c) for c in chains])
    length_h = np.array([(c[-1] - c[0]).total_seconds() / 3600.0 for c in chains])
    q1, q3 = np.percentile(steps, [25, 75])
    return {
        "прогон": path.name,
        "действий": int(len(acts)),
        "эпизодов": int(len(chains)),
        "доля эпизодов из одного действия": round(float((steps == 1).mean()), 3),
        "шагов в эпизоде: медиана": float(np.median(steps)),
        "шагов в эпизоде: квартили": [float(q1), float(q3)],
        "межквартильный размах": float(q3 - q1),
        "длина эпизода, ч: медиана": round(float(np.median(length_h)), 1),
        "длина эпизода, ч: квартили": [round(float(x), 1)
                                       for x in np.percentile(length_h, [25, 75])],
        "часов между шагами: медиана": round(float(np.median(np.concatenate(
            [np.diff([t.value for t in c]) / 3.6e12 for c in chains if len(c) > 1]
            or [np.array([np.nan])]))), 1),
    }


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    out = {}
    for name, filename in RUNS.items():
        path = ROOT / "reports" / filename
        if not path.exists():
            print(f"нет прогона {filename} — пропуск")
            continue
        out[name] = block(path)
        print(f"\n{name}:")
        for key, value in out[name].items():
            print(f"  {key}: {value}")

    val = out.get("валидация", {})
    single = float(val.get("доля эпизодов из одного действия", 1.0))
    median = float(val.get("шагов в эпизоде: медиана", 1.0))
    spread = float(val.get("межквартильный размах", 99.0))
    rule = {
        "1. одного шага обычно не хватает (эпизодов из одного действия < 50 %)":
            bool(single < 0.50),
        "2. длина эпизода предсказуема (размах не больше медианы)":
            bool(spread <= median),
    }
    accepted = all(rule.values())
    print("\nПравило приёмки (валидация):")
    for name, ok in rule.items():
        print(f"  {'да ' if ok else 'НЕТ'} {name}")
    print(f"  ВЫВОД: план из нескольких шагов в карточке "
          f"{'ПРИНЯТ' if accepted else 'НЕ принят'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "промежуток внутри эпизода, ч": EPISODE_GAP_H,
                                  "выборки": out, "правило": rule,
                                  "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
