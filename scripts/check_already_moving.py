"""Режим уже едет: не советует ли система то, что и так делается. Только CPU.

    python scripts/check_already_moving.py

Зачем. Отклик серы на уставку запаздывает: постоянная времени 4.6 ч. Значит, если
оператор (или регулятор) поднял температуру час назад, эффект ещё НЕ виден ни в
анализаторе, ни в признаках модели — и система, глядя на высокую серу, посоветует
поднять ещё. Два действия сложатся, а ждать придётся оба. Это не гипотеза: ровно от
этого в конфигурации стоит запрет двигать уставку чаще чем раз в 4 часа, но запрет
знает только о СВОИХ действиях, а не о чужих.

Меряем, как часто режим уже едет к моменту решения. Порог движения не выбран на
глаз: это максимальный шаг одного цикла рекомендации (`limits.max_step_per_cycle`,
2 °C). Если за время отклика режим уже прошёл целый такой шаг, следующий совет
удваивает воздействие.

**Правило записано ДО счёта.** Строку в карточку добавляем, если на ВАЛИДАЦИИ доля
моментов с уже идущим движением лежит между 5 % и 50 %: реже — оператор её никогда
не увидит, чаще — она перестанет что-либо выделять. Отдельно смотрим моменты с
высоким риском: там совет «поднять» наиболее вероятен, и именно там сложение
опаснее всего.
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
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "already_moving.json"
WABT = "reg_wabt"
RESPONSE_H = 4.6                 # постоянная времени канала серы, reports/delays.json
RUNS = {"val": "val_period_step1h_lock4.json", "test": "test_period_step1h.json"}
BAND = (0.05, 0.50)


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    step_c = float(cfg["limits"]["max_step_per_cycle"]["temperature_c"])
    features = build_feature_matrix()
    masks = time_split(features.index, cfg)
    print(f"шаг одного цикла: {step_c:g} °C, время отклика {RESPONSE_H:g} ч")

    out: dict = {"порог движения, °C": step_c, "окно, ч": RESPONSE_H}
    for split in ("val", "test"):
        frame = features[masks[split].to_numpy()]
        if frame.empty or WABT not in frame:
            continue
        wabt = frame[WABT].astype(float)
        # шаг сетки признаков считаем по самим данным, а не предполагаем часовой
        step_h = float(pd.Series(frame.index).diff().dt.total_seconds().median() / 3600)
        lag = max(int(round(RESPONSE_H / step_h)), 1)
        change = wabt - wabt.shift(lag)
        moving = change.abs() >= step_c
        rising = change >= step_c

        block = {
            "шаг сетки, ч": round(step_h, 2),
            "моментов": int(change.notna().sum()),
            "доля моментов: режим уже едет": round(float(moving.mean()), 3),
            "доля моментов: режим уже поднимают": round(float(rising.mean()), 3),
            "медиана хода за окно, °C": round(float(change.abs().median()), 2),
        }

        path = ROOT / "reports" / RUNS[split]
        if path.exists():
            rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
            rows["ts"] = pd.to_datetime(rows["ts"])
            rows = rows.set_index("ts")
            joined = rows.join(pd.DataFrame({"ход": change}), how="inner")
            high = joined["риск"] >= joined["риск"].quantile(0.75)
            acts = joined["исход"].eq("меняем уставки")
            block.update({
                "решений сверено": int(len(joined)),
                "доля среди четверти самых рисковых": round(
                    float((joined.loc[high, "ход"].abs() >= step_c).mean()), 3),
                "доля среди вмешательств": round(
                    float((joined.loc[acts, "ход"].abs() >= step_c).mean()), 3),
                "доля вмешательств, где режим уже поднимают": round(
                    float((joined.loc[acts, "ход"] >= step_c).mean()), 3),
            })
        out[split] = block
        print(f"\n{split}:")
        for key, value in block.items():
            print(f"  {key}: {value}")

    val = out.get("val", {})
    share = float(val.get("доля моментов: режим уже едет", 0.0))
    rule = {"доля моментов с уже идущим движением в 5–50 %":
            bool(BAND[0] <= share <= BAND[1])}
    accepted = all(rule.values())
    print(f"\nПравило приёмки (валидация): {'да ' if accepted else 'НЕТ'} "
          f"доля {share:.1%} в полосе 5–50 %")
    print(f"  ВЫВОД: предупреждение «режим уже едет» "
          f"{'ПРИНЯТО' if accepted else 'НЕ принято'}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), **out,
                                  "условие: доля в полосе": list(BAND),
                                  "правило": rule, "принят": accepted},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
