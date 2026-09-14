"""Сквозной прогон цикла МАС на одном моменте времени или на демо-окне.

    python scripts/run_cycle.py --ts "2026-04-20 12:00"
    python scripts/run_cycle.py --window bad_data_frozen_pak --every 6h

Это «скелет» демонстрации: агенты пока на базовых реализациях, но полный путь
данные → 4 агента → рекомендация оператору работает и логируется в reports/runs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.blending import BlendingAgent, components_from_data  # noqa: E402
from nefte.agents.optimizer import OptimizerAgent, linear_surrogate  # noqa: E402
from nefte.agents.orchestrator import Orchestrator  # noqa: E402
from nefte.agents.quality import QualityAgent  # noqa: E402
from nefte.agents.reliability import ReliabilityAgent, regime_anomaly_frame  # noqa: E402
from nefte.data.loaders import load_lims, load_telemetry  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.models.kinetics import make_kinetic_surrogate  # noqa: E402
from nefte.models.quality_model import (  # noqa: E402
    MODELS_DIR,
    SulfurModel,
    controllable_features,
    make_model_surrogate,
)
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

# Кандидатные управляющие воздействия для базового прогона (см. configs/config.yaml).
CONTROL_TAGS = ["T5", "T11", "F26", "P13"]

# Линейные чувствительности серы к уставкам — ЗАГЛУШКА до обученной модели.
# Знак взят из физики и подтверждён корреляциями (рост температуры → падение серы).
SENSITIVITIES = {"T5": -0.15, "T11": -0.15, "F26": 0.01, "P13": -0.5}


def load_quality_model(horizon: float | None = None,
                       path: Path | None = None) -> SulfurModel | None:
    """Обученная модель, если она есть. Иначе система работает на персистенции.

    По умолчанию берём nowcast (h=0): виртуальный анализатор приводит показания
    ПАК к лабораторной шкале и работает заметно точнее прогноза на 2 часа.
    ``path`` — конкретная модель, например сидовый вариант для проверки
    устойчивости решений.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = ([SulfurModel.default_path(horizon)] if horizon is not None
                      else [SulfurModel.default_path(0), SulfurModel.default_path(2)])
    for candidate in candidates:
        if (candidate / "meta.json").exists():
            model = SulfurModel.load(candidate)
            model.attach(build_feature_matrix())
            # какой артефакт реально попал в прогон — отчёт обязан это назвать
            model.source_path = candidate
            return model
    if path is not None:
        raise FileNotFoundError(f"нет обученной модели в {path}")
    return None


def load_sequence_model(horizon: float | None = None, path: Path | None = None):
    """Нейросетевой виртуальный анализатор, если он обучен и torch установлен.

    Инференс идёт на CPU: демо обязано работать на машине без видеокарты
    (docs/GPU_SETUP.md §6). Обучение при этом было на GPU — см. train_sequence.py.
    """
    try:
        from nefte.models.sequence import SulfurSequenceModel
    except ImportError:
        print("[модель] torch не установлен — нейросетевую модель не подключить")
        return None

    # Варианты с другим базовым сидом (``_s100``, ``_s200``) — это проверка
    # устойчивости, а не кандидаты: выбирать среди них лучший по валидации значило
    # бы подбирать сид. Раньше они и не могли сюда попасть отдельно — они затирали
    # основную модель, и загрузчик молча брал сидовый вариант.
    if path is not None:
        # Явно указанная модель — для проверки разброса по сидам на уровне решений.
        # Выбор по валидации тут не нужен и был бы вреден: сравнивается ровно то,
        # что попросили.
        path = Path(path)
        model = SulfurSequenceModel.load(path).attach(build_feature_matrix())
        model.source_path = path
        print(f"[модель] последовательность (указана явно): {path.name}")
        return model

    trained = [p for p in sorted((ROOT / "models").glob("sulfur_seq_*"))
               if (p / "meta.json").exists()
               and not re.search(r"_s\d+_h[\d.]+$", p.name)]
    # Горизонты не смешиваем: сравнивать MAE прогноза на 0 и на 2 часа бессмысленно.
    # По умолчанию берём nowcast, как и у бустинга, и только если его нет — h=2.
    candidates: list[Path] = []
    for wanted in ([horizon] if horizon is not None else [0, 2]):
        candidates = [p for p in trained if p.name.endswith(f"_h{wanted:g}")]
        if candidates:
            break
    if not candidates:
        print("[модель] обученной нейросетевой модели нет "
              "(запустите scripts/train_sequence.py)")
        return None

    def val_mae(path: Path) -> float:
        """Выбираем конфигурацию по ВАЛИДАЦИИ, а не по алфавиту имени файла."""
        try:
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            return float(meta["metrics"]["splits"]["val"]["model"]["MAE"])
        except (KeyError, ValueError, OSError):
            return float("inf")

    path = min(candidates, key=val_mae)
    model = SulfurSequenceModel.load(path).attach(build_feature_matrix())
    # какой артефакт реально попал в прогон — отчёт обязан это назвать
    model.source_path = path
    print(f"[модель] последовательность: {path.name}, "
          f"{len(model.channels)} каналов, окно {model.window} ч, "
          f"порог тревоги {model.alarm_threshold:.2f} "
          f"(выбрана по MAE на валидации из {len(candidates)} обученных)")
    return model


def attach_autoencoder(reliability, sb: StateBuilder, cfg: dict) -> bool:
    """Подменяет детектор аномалий на LSTM-автоэнкодер, если он обучен.

    Автоэнкодер работает на ОКНЕ, а не на срезе, поэтому его оценки считаются
    заранее по всей истории и передаются агенту рядом — тем же способом, что и
    скорость изменения режима.
    """
    try:
        from nefte.models.anomaly_ae import LSTMAnomalyDetector
    except ImportError:
        print("[аномалии] torch не установлен — остаётся Махаланобис")
        return False

    path = LSTMAnomalyDetector.default_path()
    if not (path / "meta.json").exists():
        print("[аномалии] автоэнкодер не обучен (scripts/train_anomaly_ae.py) — "
              "остаётся Махаланобис")
        return False

    detector = LSTMAnomalyDetector.load(path)
    frame = regime_anomaly_frame(sb.avt, sb.ht)
    hourly = frame.resample("1h", label="right", closed="right").mean()
    reliability.anomaly_series = detector.normalized(hourly)
    reliability.anomaly_parts = detector.contributions_frame(hourly)
    print(f"[аномалии] LSTM-автоэнкодер: окно {detector.window} ч, "
          f"порог {detector.threshold:.4f}")
    return True


def build_system(sb: StateBuilder, cfg: dict, model_kind: str = "boost",
                 anomaly_kind: str = "maha",
                 seq_horizon: float | None = None,
                 seq_path: Path | None = None,
                 quality_path: Path | None = None) -> Orchestrator:
    # нормировка тяжести режима — только по обучающему периоду, без заглядывания вперёд
    # сырая телеметрия нужна агенту, чтобы увидеть остановы: очистка убирает
    # замороженный на нуле расход сырья вместе с самим фактом останова
    reliability = ReliabilityAgent.from_history(sb.avt, sb.ht, cfg,
                                               raw_ht=load_telemetry("ht"))
    bounds = sb.model_bounds(CONTROL_TAGS, unit="ht")

    if anomaly_kind == "ae":
        attach_autoencoder(reliability, sb, cfg)

    model = (load_sequence_model(seq_horizon, path=seq_path) if model_kind == "seq"
             else load_quality_model(path=quality_path))
    if model_kind == "seq" and model is not None:
        # У сети нет табличного суррогата: она читает окно, а не строку признаков.
        # Кинетика берёт у модели только уровень серы, и этого достаточно —
        # так прямо написано в контракте docs/GPU_SETUP.md §3.
        surrogate = make_kinetic_surrogate(model)
        optimizer = OptimizerAgent(bounds=bounds, surrogate=surrogate, cfg=cfg,
                               reliability_agent=reliability)
        lims = load_lims()
        return Orchestrator(
            QualityAgent(model=model, cfg=cfg), reliability, optimizer, cfg=cfg,
            blending=BlendingAgent(cfg),
            components_fn=lambda ts: components_from_data(sb, ts, lims=lims),
        )

    if model is not None:
        seen = controllable_features(model, CONTROL_TAGS)
        print(f"[модель] sulfur: {len(model.features)} признаков, порог тревоги "
              f"{model.alarm_threshold:.2f}, управляющие теги в модели: {seen or 'нет'}")
        # уровень серы даёт модель, отклик на изменение уставок — кинетика:
        # в истории связи «температура → сера» почти нет, и чисто статистический
        # суррогат оставил бы оптимизатор без градиента (docs/QUALITY_AGENT.md)
        surrogate = make_kinetic_surrogate(model, base_surrogate=make_model_surrogate(model))
    else:
        print("[модель] обученной модели нет — персистенция и линейная заглушка "
              "(запустите scripts/train_quality.py)")
        surrogate = linear_surrogate(SENSITIVITIES)

    robust_order = (cfg.get("optimization") or {}).get("robust_kinetic_order")
    robust = (make_kinetic_surrogate(model, base_surrogate=make_model_surrogate(model),
                                     order=float(robust_order))
              if model is not None and robust_order else None)
    optimizer = OptimizerAgent(bounds=bounds, surrogate=surrogate, cfg=cfg,
                               reliability_agent=reliability, robust_surrogate=robust)

    # ЛИМС читается один раз: компоненты смешения собираются на каждом такте,
    # перечитывать книгу на каждый момент времени незачем
    lims = load_lims()
    return Orchestrator(
        QualityAgent(model=model, cfg=cfg), reliability, optimizer, cfg=cfg,
        blending=BlendingAgent(cfg),
        components_fn=lambda ts: components_from_data(sb, ts, lims=lims),
    )


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", help="момент времени, например '2026-04-20 12:00'")
    ap.add_argument("--window", help="имя окна из configs/config.yaml: demo_windows")
    ap.add_argument("--every", default="12h", help="шаг обхода окна")
    ap.add_argument("--model", choices=("boost", "seq"), default="boost",
                    help="какой виртуальный анализатор: CatBoost или нейросеть")
    ap.add_argument("--anomaly", choices=("maha", "ae"), default="maha",
                    help="детектор аномалий режима: Махаланобис или автоэнкодер")
    ap.add_argument("--seq-horizon", type=float, default=None,
                    help="горизонт нейросетевой модели: 0 — виртуальный анализатор, "
                         "2 — прогноз на 2 часа")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder(cfg)
    system = build_system(sb, cfg, model_kind=args.model, anomaly_kind=args.anomaly,
                          seq_horizon=args.seq_horizon)

    if args.window:
        lo, hi = cfg["demo_windows"][args.window]
        stamps = pd.date_range(lo, hi, freq=args.every)
    else:
        stamps = [pd.Timestamp(args.ts or "2026-04-20 12:00")]

    for ts in stamps:
        rec = system.run(sb.build(ts))
        print(rec.to_operator_text())
        print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
