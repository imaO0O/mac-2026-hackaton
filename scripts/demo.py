"""Сценарий показа: четыре окна, блендинг и «злые» кейсы одной командой.

    python scripts/demo.py                 # весь сценарий, ~3 минуты
    python scripts/demo.py --step stable   # одна сцена
    python scripts/demo.py --list          # что вообще есть

Демонстрация — отдельный критерий ТЗ, и репетировать её надо не набором команд из
README, а одним предсказуемым прогоном. Скрипт печатает сцену, говорит, ЧТО именно
в ней надо показать, и выводит настоящий ответ системы — без заранее записанных
результатов.

Правило демо не меняется: всё на CPU, без интернета и внешних сервисов.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.agents.blending import BlendingAgent, components_from_data  # noqa: E402
from nefte.config import load_config  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.run_cycle import build_system  # noqa: E402

# Сцены сценария. Порядок не случайный: сначала система показывает, что умеет
# молчать, и только потом — что умеет советовать. Это и есть главный тезис защиты.
SCENES = [
    {
        "key": "unit_down",
        "title": "Установка не в работе",
        "ts": "2026-04-20 12:00",
        "point": "Поточный анализатор в этот момент завис на 18.4 мг/кг, но это "
                 "СЛЕДСТВИЕ останова, а не поломка прибора. Система называет "
                 "оператору причину, а не следствие, и не советует греть холодный "
                 "реактор. Половина всех отказов на тестовом периоде — это он.",
    },
    {
        "key": "bad_data_frozen_pak",
        "title": "Недостоверные данные: прибор врёт, лаборатория устарела",
        "window": "bad_data_frozen_pak",
        "every": "4D",
        "point": "Оба источника качества недостоверны — система отказывается от "
                 "рекомендации и перечисляет причины по каждому тегу. Отказ — это "
                 "требование ТЗ, а не отговорка.",
    },
    {
        "key": "stable",
        "title": "Устойчивый режим",
        "window": "stable",
        "every": "20D",
        "point": "Риск ниже порога вмешательства — система держит режим и прямо "
                 "говорит, что менять уставки не нужно. Лишние воздействия в "
                 "спокойном режиме запрещены ТЗ.",
    },
    {
        "key": "quality_risk",
        "title": "Риск по качеству: рекомендация с объяснением",
        # конкретный момент, а не обход окна: на защите нужна сцена, где система
        # именно СОВЕТУЕТ, а не держит режим
        "ts": "2026-06-15 12:00",
        "point": "Здесь видно всё, что требует п.5 ТЗ: проблема, действие, эффект "
                 "относительно бездействия, проверенные ограничения, уверенность и "
                 "объяснение. Плюс рецептура смешения на прогнозной сере.",
    },
    {
        "key": "quality_watch",
        "title": "Тот же месяц, но риск ниже порога",
        "window": "quality_risk",
        "every": "10D",
        "point": "Соседние моменты того же окна: риск есть, но ниже порога "
                 "вмешательства — система держит режим под наблюдением и говорит "
                 "об этом прямо. Порог подобран на валидации, цена его сдвига "
                 "измерена (docs/HARD_CHECKS.md §3).",
    },
]


def show_blending(sb: StateBuilder, cfg: dict, ts: str = "2026-02-28 00:00") -> None:
    """Отдельная сцена: почему смешение не спасает по сере."""
    agent = BlendingAgent(cfg)
    components = components_from_data(sb, pd.Timestamp(ts))
    base = min(components, key=lambda c: c.sulfur_mgkg)
    print(f"\nПредельная доля компонентов при разбавлении «{base.name}» "
          f"(сера {base.sulfur_mgkg:.1f} мг/кг):")
    for component in components:
        if component is base:
            continue
        print(f"  {component.name:28s} не более "
              f"{agent.max_share_of(base, component) * 100:.4f} %")
    print("  Вывод: единственный рычаг по сере — глубина гидроочистки, "
          "а не рецептура.")


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", help="показать одну сцену по ключу")
    ap.add_argument("--list", action="store_true", help="перечислить сцены")
    args = ap.parse_args()

    if args.list:
        for scene in SCENES:
            print(f"{scene['key']:22s} {scene['title']}")
        print(f"{'blending':22s} Смешение: предельные доли компонентов")
        return 0

    cfg = load_config()
    sb = StateBuilder(cfg)

    # сцена про смешение не трогает модель и агентов — незачем их поднимать
    if args.step == "blending":
        show_blending(sb, cfg)
        return 0

    scenes = [s for s in SCENES if not args.step or s["key"] == args.step]
    if not scenes:
        print(f"нет такой сцены: {args.step} (см. --list)")
        return 1
    system = build_system(sb, cfg)
    for number, scene in enumerate(scenes, 1):
        print("\n" + "=" * 80)
        print(f"СЦЕНА {number}. {scene['title']}")
        print(f"Что показываем: {scene['point']}")
        print("=" * 80)

        if "ts" in scene:
            stamps = [pd.Timestamp(scene["ts"])]
        else:
            lo, hi = cfg["demo_windows"][scene["window"]]
            stamps = pd.date_range(lo, hi, freq=scene["every"])
        for ts in stamps:
            # лимит частоты воздействий — состояние оркестратора; между сценами
            # его надо сбрасывать, иначе вторая сцена показывает последствия первой
            system._last_action_ts = None
            print()
            print(system.run(sb.build(ts)).to_operator_text())

    if not args.step:
        print("\n" + "=" * 80)
        print(f"СЦЕНА {len(scenes) + 1}. Смешение: почему рецептурой серу не вытянуть")
        print("=" * 80)
        show_blending(sb, cfg)
        print("\nПрогоны записаны в reports/runs/ — логику любого решения можно "
              "проверить постфактум.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
