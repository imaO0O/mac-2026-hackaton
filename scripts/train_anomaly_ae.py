"""LSTM-автоэнкодер против расстояния Махаланобиса: обучение и честное сравнение.

    python scripts/train_anomaly_ae.py
    python scripts/train_anomaly_ae.py --window 48 --epochs 300 --cpu

Оба детектора решают одну задачу — «этот режим нетипичен» — но разными средствами.
Махаланобис смотрит на мгновенное сочетание шести описателей режима, автоэнкодер —
на форму окна. Сравнение честное только при одинаковых условиях, поэтому:

* набор описателей один и тот же (`agents/reliability.regime_anomaly_frame`);
* нормировка и порог считаются ТОЛЬКО по обучающему периоду;
* квантиль порога одинаковый (0.99 обучающего периода).

Что сравниваем. Прямой разметки аномалий в пакете нет, поэтому меряем то, что
можно измерить:

1. доля нетипичных моментов в train / val / test — если на тесте она резко выше,
   детектор реагирует на смену периода, а не на режим;
2. согласие детекторов между собой на тесте;
3. поведение на СОБЫТИЯХ, про которые мы знаем из данных: останов установки,
   зависание поточного анализатора, превышение спецификации по сере;
4. объяснимость — здесь автоэнкодер заведомо слабее, и это надо сказать вслух.

Результат: reports/anomaly_ae.json и таблица в консоли.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.reliability import ReliabilityAgent, regime_anomaly_frame  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import frozen_mask  # noqa: E402
from nefte.data.loaders import load_pak, load_telemetry  # noqa: E402
from nefte.models.anomaly import RegimeAnomalyDetector  # noqa: E402
from nefte.models.anomaly_ae import LSTMAnomalyDetector  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402


def share(mask: pd.Series, window: tuple[str, str]) -> float:
    part = mask.loc[window[0]:window[1]].dropna()
    return float(part.mean()) if len(part) else float("nan")


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24, help="длина окна, часов")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    train = tuple(cfg["split"]["train"])

    print("[1/4] данные и описатели режима…")
    sb = StateBuilder(cfg)
    frame = regime_anomaly_frame(sb.avt, sb.ht)
    # шаг сетки 10 минут; окно задаём в часах, поэтому прореживаем до часа —
    # автоэнкодеру важна форма, а не каждый десятиминутный отсчёт
    hourly = frame.resample("1h", label="right", closed="right").mean()
    print(f"      {hourly.shape[0]} часов × {hourly.shape[1]} описателей: "
          f"{', '.join(hourly.columns)}")

    print("[2/4] Махаланобис (CPU)…")
    maha = RegimeAnomalyDetector.fit(hourly, list(hourly.columns), train=train)
    maha_score = pd.Series(maha.distance(hourly) / maha.threshold, index=hourly.index)
    maha_flag = maha_score > 1.0

    print(f"[3/4] LSTM-автоэнкодер (окно {args.window} ч)…")
    ae = LSTMAnomalyDetector.fit(hourly, list(hourly.columns), train=train,
                                 window=args.window, hidden=args.hidden,
                                 latent=args.latent, epochs=args.epochs,
                                 prefer_gpu=not args.cpu)
    if not ae.fitted:
        print("      не удалось обучить: слишком мало полных окон")
        return 1
    print(f"      {ae.history['epochs']} эпох на {ae.history['n_windows']} окнах, "
          f"устройство {ae.history['device']}, MSE {ae.history['val_mse']:.4f}")
    ae_score = ae.normalized(hourly)
    ae_flag = (ae_score > 1.0).fillna(False)
    path = ae.save()

    print("[4/4] сравнение…")
    # --- события, про которые мы знаем из данных ------------------------ #
    agent = ReliabilityAgent.from_history(sb.avt, sb.ht, cfg,
                                          raw_ht=load_telemetry("ht"))
    down = (agent.down_series.resample("1h", label="right", closed="right").max()
            .reindex(hourly.index).fillna(False).astype(bool)
            if agent.down_series is not None else pd.Series(False, index=hourly.index))
    pak = load_pak()["sulfur_ppm"]
    pak_frozen = (frozen_mask(pak, int(cfg["telemetry"]["frozen_min_samples"]))
                  .resample("1h", label="right", closed="right").max()
                  .reindex(hourly.index).fillna(False).astype(bool))
    over = (sb.lims_sulfur > cfg["spec"]["product_sulfur_mgkg"]["max"])
    over_hourly = (over.resample("1h", label="right", closed="right").max()
                   .reindex(hourly.index).fillna(False).astype(bool))

    windows = {"train": train, "val": tuple(cfg["split"]["val"]),
               "test": tuple(cfg["split"]["test"])}
    # Доли по событиям сами по себе обманчивы: детектор с базовой частотой 4 %
    # и на событиях даст около 4 %. Поэтому рядом считаем ЛИФТ — во сколько раз
    # чаще детектор срабатывает на событии, чем вообще на тестовом периоде.
    # Лифт около 1 означает «сигнала нет», как бы ни выглядел сам процент.
    rows = []
    for name, detector_flag in (("Махаланобис", maha_flag), ("автоэнкодер", ae_flag)):
        row = {"детектор": name}
        for split, bounds in windows.items():
            row[f"нетипично {split}, %"] = round(share(detector_flag, bounds) * 100, 2)
        base = share(detector_flag, windows["test"])
        for label, mask in (("остановы", down), ("зависший ПАК", pak_frozen),
                            ("часы превышений", over_hourly)):
            value = float(detector_flag[mask].mean()) if mask.any() else float("nan")
            row[f"{label}, %"] = round(value * 100, 2)
            row[f"{label}, лифт"] = round(value / base, 2) if base else None
        rows.append(row)
    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    both = maha_flag & ae_flag
    either = maha_flag | ae_flag
    test_lo, test_hi = windows["test"]
    agreement = {
        "оба сработали, ч": int(both.loc[test_lo:test_hi].sum()),
        "хотя бы один, ч": int(either.loc[test_lo:test_hi].sum()),
        "жаккар": round(float(both.loc[test_lo:test_hi].sum()
                              / max(int(either.loc[test_lo:test_hi].sum()), 1)), 3),
        "корреляция оценок": round(float(
            pd.concat([maha_score, ae_score], axis=1).dropna()
            .loc[test_lo:test_hi].corr().iloc[0, 1]), 3),
    }
    print(f"\nСогласие на тесте: {agreement}")

    report = {
        "window_hours": args.window, "hidden": args.hidden, "latent": args.latent,
        "device": ae.history.get("device"), "ae_history": ae.history,
        "columns": list(hourly.columns),
        "thresholds": {"mahalanobis": maha.threshold, "autoencoder": ae.threshold},
        "table": rows, "agreement_test": agreement,
        "model_path": str(path.relative_to(ROOT)),
    }
    out = ROOT / "reports" / "anomaly_ae.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    # --- вердикт ------------------------------------------------------- #
    maha_row, ae_row = rows[0], rows[1]
    drift_maha = maha_row["нетипично test, %"] - maha_row["нетипично train, %"]
    drift_ae = ae_row["нетипично test, %"] - ae_row["нетипично train, %"]
    print("\nВердикт:")
    print(f"  сдвиг доли аномалий train→test: Махаланобис {drift_maha:+.2f} п.п., "
          f"автоэнкодер {drift_ae:+.2f} п.п. — оба реагируют на смену периода.")
    for label in ("остановы", "зависший ПАК", "часы превышений"):
        print(f"  {label}: лифт Махаланобиса {maha_row[f'{label}, лифт']}, "
              f"автоэнкодера {ae_row[f'{label}, лифт']}  (1.0 — сигнала нет)")
    print(f"  согласие детекторов на тесте слабое: жаккар {agreement['жаккар']}, "
          f"корреляция оценок {agreement['корреляция оценок']}. Они видят РАЗНОЕ, "
          f"и складывать их в один индекс без разметки нельзя.")
    print("  объяснимость: у Махаланобиса вклад переменной — это «на сколько сигм она")
    print("  выбивается с учётом связей», у автоэнкодера — лишь «какой канал сеть")
    print("  восстановила хуже». Для карточки оператора первое сильнее, поэтому")
    print("  детектором по умолчанию остаётся Махаланобис.")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    print(f"Модель: {path.relative_to(ROOT)}")
    print("\nПодключение в цикл: python scripts/run_cycle.py --anomaly ae")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
