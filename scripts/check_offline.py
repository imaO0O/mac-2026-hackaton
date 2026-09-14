"""Сквозной прогон на настоящих данных без сети. Только CPU.

    python scripts/check_offline.py
    python scripts/check_offline.py --ts "2026-06-11 00:00" "2026-04-20 12:00"

Организаторы на сессии 11.09 назвали критерием оценки запуск в закрытой среде:
технологическая сеть предприятия с интернетом не связана. Тест
``tests/test_runs_offline.py`` проверяет это на синтетическом срезе; здесь — то же
на выданных данных и обученных моделях: сборка системы (агенты, модель качества,
детектор аномалий, смешение), цикл решения на нескольких моментах и импорт
библиотек дашборда идут внутри ``nefte.offline.forbid_network``. Любая попытка
разрешить внешнее имя или открыть соединение наружу — падение с именем адреса.

Что НЕ проверяется: установка зависимостей. ``pip install`` без сети требует
заранее собранного каталога колёс — порядок в README, «Закрытый контур».
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.offline import forbid_network  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

# останов, риск по сере ниже порога с Т95 за пределом, действие
DEFAULT_STAMPS = ("2026-04-20 12:00", "2026-06-11 00:00", "2026-02-28 00:00")


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", nargs="+", default=list(DEFAULT_STAMPS))
    args = ap.parse_args()

    attempts: list = []
    started = time.perf_counter()
    with forbid_network(attempts):
        from nefte.config import load_config
        from nefte.pipeline import StateBuilder
        from scripts.run_cycle import build_system

        cfg = load_config()
        sb = StateBuilder(cfg)
        system = build_system(sb, cfg)
        system.log_runs = False
        outcomes = []
        for ts in args.ts:
            rec = system.run(sb.build(pd.Timestamp(ts)))
            outcomes.append((ts, rec.outcome()))
        # дашборд: сам сервер здесь не поднимается, но его библиотеки грузятся
        import plotly.graph_objects  # noqa: F401
        import streamlit  # noqa: F401

    print(f"\nСистема собрана и отработала без сети за {time.perf_counter() - started:.0f} с:")
    for ts, verdict in outcomes:
        print(f"  {ts}: {verdict}")
    print(f"Попыток выйти в сеть: {len(attempts)}")
    return 0 if not attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
