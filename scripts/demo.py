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
        "point": "Оба поточных анализатора в этот момент стоят на полках — файловый "
                 "ПАК на 18.4, оперативный Q21 на 24.9 мг/кг, — но это СЛЕДСТВИЕ "
                 "останова, а не поломка приборов. Система называет "
                 "оператору причину, а не следствие, и не советует греть холодный "
                 "реактор. Остановы — самая частая причина отказов на тесте: около двух третей.",
    },
    {
        "key": "bad_data_frozen_pak",
        "title": "Недостоверные данные: лаборатория устарела, прибор заморожен",
        # Конкретный момент по журналу прогона, а не обход окна: окно
        # bad_data_frozen_pak совпадает с остановом установки, и сцена показывала
        # не отказ по данным, а повтор первой сцены.
        "ts": "2026-06-21 00:00",
        "point": "Лабораторный анализ старше трёх суток, оба поточных анализатора "
                 "заморожены (Q21 — на полке неисправности 24.9 мг/кг), в тегах "
                 "заглушки — система отказывается от "
                 "рекомендации и перечисляет причины. Отказ — это требование ТЗ, а "
                 "не отговорка.",
    },
    {
        "key": "stable",
        "title": "Устойчивый режим",
        # первый момент окна — факт вне спецификации и действие, для этой сцены он
        # не годится; моменты выбраны по прогону валидационного периода
        "ts": ["2025-11-21 00:00", "2025-12-11 00:00"],
        "point": "Риск ниже порога вмешательства — система держит режим и прямо "
                 "говорит, что менять уставки не нужно. Лишние воздействия в "
                 "спокойном режиме запрещены ТЗ.",
    },
    {
        "key": "quality_risk",
        "title": "Риск по качеству: рекомендация с объяснением",
        # конкретный момент, а не обход окна: на защите нужна сцена, где система
        # именно СОВЕТУЕТ, а не держит режим. Прежний момент (2026-02-18) после
        # перехода на журнал замен катализатора стал отказом «режим недопустим по
        # тяжести» — катализатор в конце цикла, — и сцена показывала не то, что
        # обещает. Этот же момент — пример карточки в README. Держит
        # tests/test_demo_scenes.py.
        "ts": "2026-03-05 00:00",
        "point": "Здесь видно всё, что требует п.5 ТЗ: проблема, действие, эффект "
                 "относительно бездействия, проверенные ограничения, уверенность и "
                 "объяснение. Плюс рецептура смешения на прогнозной сере.",
    },
    {
        "key": "quality_watch",
        "title": "Риск есть, но ниже порога; и Т95 за пределом при спокойной сере",
        # риск между половиной порога и порогом, и отдельно — Т95 выше предела при
        # низком риске по сере. Прежний первый момент (2026-01-29) был выбран по
        # журналу прогона, а прогон помнит запрет частых воздействий: там режим
        # держали из-за недавнего действия. Дашборд решает каждый момент с чистого
        # листа, и в нём 29.01 — действие при риске 20 %, то есть сцена противоречила
        # своему описанию. Моменты сверены так, как их видит дашборд
        # (tests/test_demo_scenes.py).
        "ts": ["2026-03-11 00:00", "2026-06-11 00:00"],
        "point": "Риск по сере есть, но ниже порога вмешательства — система держит "
                 "режим под наблюдением и говорит об этом прямо. Во втором моменте "
                 "сера спокойна, а Т95 по последнему анализу выше предела: система не "
                 "берётся его оптимизировать, но запрещает варианты, которые сделают "
                 "хуже. Порог подобран на валидации, цена его сдвига измерена "
                 "(docs/HARD_CHECKS.md §3).",
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

    # Второй рычаг смешения — присадка, и он единственный работающий. Показываем
    # его вместе с ценой: без цены рецептура выглядит бесплатной.
    recipe = agent.optimize(components)
    if recipe.cetane_improver_pct > 0:
        print(f"\n  Зато цетановое число норматив НЕ проходит: назначена "
              f"присадка {recipe.cetane_improver_pct:.3f} % массы. Она дороже "
              f"топлива в 100 раз, поэтому съедает "
              f"{recipe.improver_cost_share * 100:.1f} % цены тонны: выпуск "
              f"{recipe.throughput_tph:.1f} т/ч, чистая ценность "
              f"{recipe.net_value_tph:.1f} т/ч.")


def show_cetane(sb: StateBuilder, cfg: dict) -> None:
    """Сцена: обязательный показатель, из-за которого летнюю марку уже не получить.

    ЦЧ ГО ДТ падает четыре года подряд. У самого ГО ДТ он не нормируется — норматив
    у товарного топлива: летнее не ниже 51, зимнее не ниже 49 (ответ организаторов
    15.09). Последний анализ 50.0: летнюю марку из одного ГО ДТ без присадки или
    другого компонента не получить, для зимней остаётся одна единица.
    """
    from nefte.data.loaders import lims_series, load_lims
    from nefte.models.cetane import (
        CETANE_SPEC_MIN,
        dose_for_deficit,
        improver_cost_share,
        trend_per_year,
    )

    cetane = lims_series("Гидроочистка|2|CetaneNumber", load_lims()).sort_index()
    by_year = cetane.groupby(cetane.index.year).mean()
    print("\nЦетановое число — третий обязательный показатель по ответу организаторов.")
    print("  Среднее по годам:")
    for year, value in by_year.items():
        print(f"    {year}  {value:.1f}")
    last, when = float(cetane.iloc[-1]), cetane.index[-1]
    grades = cfg["spec"].get("grades") or {}
    floors = {g["name"]: g.get("cetane_number_min") for g in grades.values()}
    print(f"  Последний анализ {when:%d.%m.%Y}: {last:.1f}. Тренд "
          f"{trend_per_year(cetane):+.2f} единиц в год.")
    print("  Меряется у ГО ДТ, а норматив — у марки (ответ организаторов 15.09): "
          + "; ".join(f"{name} — " + ("не нормируется" if floor is None
                                      else f"не ниже {floor:g}"
                                      + (" (не проходит)" if last < floor else ""))
                      for name, floor in floors.items()))
    print(f"  Анализов всего {len(cetane)} за три с половиной года — раз в месяц. "
          "Строить по ним прогноз нечестно, и мы не строим: показатель входит "
          "ограничением по последнему анализу с поправкой на тренд.")
    print("  Цетановый индекс ASTM D976 на этих данных не работает: корреляция с "
          "лабораторией 0.03, хуже, чем просто среднее по истории.")
    dose = dose_for_deficit(max(0.0, CETANE_SPEC_MIN - last))
    if dose:
        print(f"  Цена летней марки из ГО ДТ: {dose:.3f} % присадки = "
              f"{improver_cost_share(dose) * 100:.1f} % стоимости тонны топлива.")


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
        print(f"{'cetane':22s} Цетановое число: летнюю марку из ГО ДТ уже не получить")
        return 0

    cfg = load_config()
    sb = StateBuilder(cfg)

    # сцена про смешение не трогает модель и агентов — незачем их поднимать
    if args.step == "blending":
        show_blending(sb, cfg)
        return 0
    if args.step == "cetane":
        show_cetane(sb, cfg)
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
            stamps = [pd.Timestamp(t) for t in
                      (scene["ts"] if isinstance(scene["ts"], list) else [scene["ts"]])]
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
        print("\n" + "=" * 80)
        print(f"СЦЕНА {len(scenes) + 2}. Цетановое число: летнюю марку из ГО ДТ уже не получить")
        print("=" * 80)
        show_cetane(sb, cfg)
        print("\nПрогоны записаны в reports/runs/ — логику любого решения можно "
              "проверить постфактум.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
