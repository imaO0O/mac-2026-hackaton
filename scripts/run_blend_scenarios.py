"""Смешение: несколько сценариев вместо одного жёсткого ответа.

    python scripts/run_blend_scenarios.py
    python scripts/run_blend_scenarios.py --ts "2026-07-25 00:00"

Организаторы ответили прямо: данных по смешению не будет, и от команды ждут
собственную обоснованную модель и НЕСКОЛЬКО НЕЖЁСТКИХ СЦЕНАРИЕВ. Один
«правильный» рецепт на выданных данных обосновать нечем — зато можно показать,
как рецептура и её экономика отзываются на то, что реально меняется:

* сера сырья (качество нефти плавает, и в ЛИМС это видно);
* сера гидроочищенного продукта (её мы и регулируем режимом);
* цетановое число (падает год от года, см. scripts/check_cetane.py);
* располагаемые расходы компонентов (АВТ работает не в постоянном режиме).

Сценарии — это НЕ прогноз. Это ответ на вопрос «если так, то что», и именно в
таком виде их и надо показывать: с базой из реальных данных на выбранный момент и
с явным перечнем того, что в каждом сценарии изменено.

Главное, что здесь проверяется: система не выдаёт один и тот же ответ независимо
от входа. Если рецептура и экономика не отзываются на смену сырья, значит
блок смешения декоративный, и лучше это знать заранее.

Результат: reports/blend_scenarios.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.blending import BlendingAgent, components_from_data  # noqa: E402
from nefte.agents.schemas import BlendComponent  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.models.cetane import CETANE_SPEC_MIN  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

HYDROTREATED = "ГО ДТ"


def scenarios() -> list[dict]:
    """Что меняем и зачем. Каждый сценарий — одно осмысленное «а если»."""
    return [
        {"имя": "база", "что изменено": "ничего, данные на момент", "правки": {}},
        {"имя": "сырьё чище вдвое",
         "что изменено": "сера прямогонных фракций ×0.5",
         "правки": {"straight_sulfur_factor": 0.5}},
        {"имя": "сырьё грязнее вдвое",
         "что изменено": "сера прямогонных фракций ×2",
         "правки": {"straight_sulfur_factor": 2.0}},
        {"имя": "режим мягче",
         "что изменено": "сера ГО ДТ 9.5 мг/кг (у самого предела)",
         "правки": {"godt_sulfur": 9.5}},
        {"имя": "режим жёстче",
         "что изменено": "сера ГО ДТ 5.0 мг/кг (глубокая очистка)",
         "правки": {"godt_sulfur": 5.0}},
        {"имя": "цетановое число просело",
         "что изменено": "ЦЧ ГО ДТ 49 — ниже норматива",
         "правки": {"godt_cetane": 49.0}},
        {"имя": "цетановое число с запасом",
         "что изменено": "ЦЧ ГО ДТ 55",
         "правки": {"godt_cetane": 55.0}},
        {"имя": "АВТ снизила выработку",
         "что изменено": "расходы прямогонных фракций ×0.5",
         "правки": {"straight_capacity_factor": 0.5}},
        # Сценарий, в котором смешение вообще имеет смысл. На выданных данных
        # степеней свободы у смешения почти нет (см. вывод внизу), поэтому нужен
        # хотя бы один вариант со ВТОРЫМ очищенным потоком — соседняя установка
        # или покупной компонент. Это ДОПУЩЕНИЕ, и оно помечено как допущение.
        {"имя": "второй очищенный поток",
         "что изменено": "добавлен компонент 3 мг/кг, ЦЧ 52, 80 т/ч (ДОПУЩЕНИЕ)",
         "правки": {"add_clean": True}},
        {"имя": "второй поток + мягкий режим",
         "что изменено": "то же плюс сера ГО ДТ 9.5 — разбавление даёт запас",
         "правки": {"add_clean": True, "godt_sulfur": 9.5}},
    ]


def apply_edits(components: list, edits: dict) -> list:
    """Копия набора компонентов с внесёнными правками сценария."""
    out = []
    for c in components:
        update = {}
        if c.name == HYDROTREATED:
            if "godt_sulfur" in edits:
                update["sulfur_mgkg"] = float(edits["godt_sulfur"])
            if "godt_cetane" in edits:
                update["cetane_number"] = float(edits["godt_cetane"])
        else:
            if "straight_sulfur_factor" in edits:
                update["sulfur_mgkg"] = c.sulfur_mgkg * float(edits["straight_sulfur_factor"])
            if "straight_capacity_factor" in edits:
                update["available_tph"] = c.available_tph * float(
                    edits["straight_capacity_factor"])
        out.append(c.model_copy(update=update) if update else c)
    if edits.get("add_clean"):
        out.append(BlendComponent(
            name="Очищенный со стороны", sulfur_mgkg=3.0, density_15c=834.0,
            t95_c=340.0, cfpp_c=-12.0, cetane_number=52.0, available_tph=80.0,
            is_assumption=True))
    return out


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="2026-07-25 00:00",
                    help="момент, с которого берётся база сценариев")
    args = ap.parse_args()

    cfg = load_config()
    sb = StateBuilder()
    ts = pd.Timestamp(args.ts)
    base_components = components_from_data(sb, ts)
    agent = BlendingAgent()

    print(f"База: данные на {ts:%Y-%m-%d %H:%M}\n")
    print(pd.DataFrame([{
        "компонент": c.name,
        "сера, мг/кг": round(c.sulfur_mgkg, 1),
        "ЦЧ": None if c.cetane_number is None else round(c.cetane_number, 1),
        "расход, т/ч": round(c.available_tph, 1),
    } for c in base_components]).to_string(index=False))

    rows, detail = [], []
    for scenario in scenarios():
        components = apply_edits(base_components, scenario["правки"])
        recipe = agent.optimize(components)
        godt_share = recipe.fractions.get(HYDROTREATED, 0.0)
        rows.append({
            "сценарий": scenario["имя"],
            "доля ГО ДТ": f"{godt_share:.0%}",
            "сера смеси": (None if recipe.properties.get("sulfur_mgkg") is None
                           else round(recipe.properties["sulfur_mgkg"], 2)),
            "ЦЧ смеси": (None if recipe.properties.get("cetane_number") is None
                         else round(recipe.properties["cetane_number"], 1)),
            "присадка, %": round(recipe.cetane_improver_pct, 3),
            "выпуск, т/ч": round(recipe.throughput_tph, 1),
            "чистая ценность": round(recipe.net_value_tph, 1),
            "годна": "да" if recipe.feasible else "НЕТ",
        })
        detail.append({
            "сценарий": scenario["имя"],
            "что изменено": scenario["что изменено"],
            "доли": {k: round(v, 4) for k, v in recipe.fractions.items()},
            "свойства": {k: round(v, 3) for k, v in recipe.properties.items()
                         if v == v},
            "присадка_%": round(recipe.cetane_improver_pct, 4),
            "стоимость_присадки_доля_цены": round(recipe.improver_cost_share, 4),
            "выпуск_тч": round(recipe.throughput_tph, 2),
            "чистая_ценность_тч": round(recipe.net_value_tph, 2),
            "годна": recipe.feasible,
            "нарушения": recipe.violations,
            "не_подтверждено": recipe.uncertified,
        })

    frame = pd.DataFrame(rows)
    print("\nСценарии\n")
    print(frame.to_string(index=False))

    print("\nЧто изменено в каждом:")
    for scenario in scenarios():
        print(f"  {scenario['имя']:26s} — {scenario['что изменено']}")

    # Проверка на декоративность: если ответ одинаков во всех сценариях, блок
    # смешения ни на что не реагирует, и это надо сказать вслух.
    distinct = frame.drop(columns=["сценарий"]).astype(str).drop_duplicates()
    print(f"\nРазличных ответов: {len(distinct)} из {len(frame)} сценариев.")
    if len(distinct) <= 1:
        print("ВНИМАНИЕ: ответ одинаков везде — блок смешения не реагирует на вход.")
    else:
        print("Рецептура и экономика отзываются на смену условий, а не выдают "
              "один и тот же ответ.")

    # Почему часть сценариев совпадает — это не сбой, а физика предела.
    straight = [c for c in base_components if c.name != HYDROTREATED]
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]
    if straight and min(c.sulfur_mgkg for c in straight) > limit * 50:
        ratio = min(c.sulfur_mgkg for c in straight) / limit
        print(f"\nПочему сценарии по сере сырья дают один и тот же ответ. Сера "
              f"прямогонных фракций выше предела Евро-5 в {ratio:.0f} раз. И вдвое "
              f"чище, и вдвое грязнее — всё равно на два порядка выше {limit} мг/кг, "
              f"так что в смесь они не входят ни при каком раскладе. На выданных "
              f"данных у смешения почти нет степеней свободы: товарный продукт — это "
              f"практически чистый гидроочищенный ДТ, а единственный работающий "
              f"рычаг — цетаноповышающая присадка. Сценарии со вторым очищенным "
              f"потоком показывают, как блок повёл бы себя, будь такой компонент.")

    print(f"\nНапоминание про цетановое число: норматив {CETANE_SPEC_MIN}, "
          "последний анализ на выданных данных 50.0. Сценарий «цетановое число "
          "просело» — не выдумка, а продолжение измеренного тренда.")

    out = ROOT / "reports" / "blend_scenarios.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "момент": str(ts),
        "предел_серы": cfg["spec"]["product_sulfur_mgkg"]["max"],
        "норматив_ЦЧ": CETANE_SPEC_MIN,
        "база": [{"компонент": c.name, "сера_мгкг": c.sulfur_mgkg,
                  "ЦЧ": c.cetane_number, "расход_тч": c.available_tph,
                  "допущение": c.is_assumption} for c in base_components],
        "сценарии": detail,
        "различных_ответов": int(len(distinct)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
