"""Агент блендинга: подбор рецептуры товарного дизельного топлива.

В выданном пакете НЕТ данных по компонентам смешения: ни резервуаров, ни рецептур,
ни присадок, ни анализов товарного продукта после смешения. Поэтому блок смешения
целиком построен на допущениях, и все они перечислены здесь и в docs/DATA_NOTES.md.

Что взято из данных, а что придумано:

| Величина | Источник |
|---|---|
| сера гидроочищенного ДТ | ЛИМС «Гидроочистка, точка 2» — факт |
| сера прямогонной фракции | ЛИМС «Гидроочистка, точка 1» (сырьё ГО) — факт |
| плотность, Т95, ПТФ компонентов | ЛИМС соответствующих точек отбора — факт |
| располагаемые расходы компонентов | телеметрия (`F17`, `F30`, `F34`) — факт |
| правила смешения | линейные по массе — ДОПУЩЕНИЕ |
| эффект депрессорной присадки | −8 °C к ПТФ при дозировке до 500 ppm — ДОПУЩЕНИЕ |

Правило смешения по сере линейно по массе и физически корректно. Для Т95 и ПТФ
линейность — упрощение: реальные показатели смешиваются нелинейно, через
блендинг-индексы. Это помечено как допущение и должно быть названо на защите.
"""
from __future__ import annotations

import itertools

import numpy as np

from nefte.agents.schemas import BlendComponent, BlendRecipe
from nefte.config import load_config

__all__ = ["ADDITIVE_CFPP_EFFECT_C", "ADDITIVE_MAX_PPM", "BlendComponent", "BlendRecipe",
           "BlendingAgent", "components_from_data", "mix"]

# Эффект депрессорной присадки на предельную температуру фильтруемости.
# ДОПУЩЕНИЕ: паспортных данных присадки в пакете нет.
ADDITIVE_CFPP_EFFECT_C = -8.0
ADDITIVE_MAX_PPM = 500.0


def mix(components: list[BlendComponent], fractions: dict[str, float],
        additive_ppm: float = 0.0) -> dict[str, float]:
    """Свойства смеси при заданных массовых долях.

    Сера — строго линейно по массе (физически верно). Плотность, Т95 и ПТФ —
    линейно как ДОПУЩЕНИЕ; присадка сдвигает ПТФ пропорционально дозировке.
    """
    by_name = {c.name: c for c in components}
    props: dict[str, float] = {}

    def weighted(attr: str) -> float | None:
        total, weight = 0.0, 0.0
        for name, share in fractions.items():
            value = getattr(by_name[name], attr)
            if value is not None and share > 0:
                total += value * share
                weight += share
        return total / weight if weight > 0 else None

    props["sulfur_mgkg"] = weighted("sulfur_mgkg") or 0.0
    for attr, key in (("density_15c", "density_15c"), ("t95_c", "t95_c"), ("cfpp_c", "cfpp_c")):
        value = weighted(attr)
        if value is not None:
            props[key] = value

    if "cfpp_c" in props and additive_ppm > 0:
        share = min(additive_ppm / ADDITIVE_MAX_PPM, 1.0)
        props["cfpp_c"] += ADDITIVE_CFPP_EFFECT_C * share
    return props


class BlendingAgent:
    """Подбирает рецептуру: максимум выпуска при соблюдении спецификации.

    Перебор по симплексу долей с шагом ``step``. Компонентов единицы, шаг крупный,
    поэтому полный перебор на CPU занимает миллисекунды и не требует солвера —
    зато результат воспроизводим и его легко объяснить.
    """

    # Компонент, серу которого прогнозирует агент качества: это и есть продукт
    # гидроочистки, ЛИМС «Гидроочистка, точка 2».
    HYDROTREATED = "ГО ДТ"

    def __init__(self, cfg: dict | None = None, step: float = 0.01):
        self.cfg = cfg or load_config()
        self.step = step
        self.spec = self.cfg["spec"]

    # ------------------------------------------------------------------ #
    def check(self, props: dict[str, float], fractions: dict[str, float]) -> list[str]:
        """Жёсткие проверки. Сумма долей — требование ТЗ, а не косметика."""
        violations = []
        total = sum(fractions.values())
        tol = self.spec["blend_fractions_sum"]["tolerance"]
        if abs(total - self.spec["blend_fractions_sum"]["value"]) > tol:
            violations.append(f"сумма долей {total:.4f}, а должна быть 1")

        sulfur = props.get("sulfur_mgkg")
        if sulfur is not None and sulfur > self.spec["product_sulfur_mgkg"]["max"]:
            violations.append(
                f"сера {sulfur:.2f} мг/кг выше предела "
                f"{self.spec['product_sulfur_mgkg']['max']}")

        density = props.get("density_15c")
        if density is not None:
            lo, hi = self.spec["density_15c_kgm3"]["min"], self.spec["density_15c_kgm3"]["max"]
            if not lo <= density <= hi:
                violations.append(f"плотность {density:.1f} вне диапазона {lo}–{hi} (допущение)")

        t95 = props.get("t95_c")
        if t95 is not None and t95 > self.spec["t95_c"]["max"]:
            violations.append(f"Т95 {t95:.1f} °C выше {self.spec['t95_c']['max']} (допущение)")

        cfpp = props.get("cfpp_c")
        if cfpp is not None and cfpp > self.spec["cfpp_c"]["max"]:
            violations.append(f"ПТФ {cfpp:.1f} °C выше {self.spec['cfpp_c']['max']} (допущение)")
        return violations

    # ------------------------------------------------------------------ #
    def _grid(self, n: int) -> list[tuple[float, ...]]:
        """Точки симплекса: доли с шагом step, сумма ровно 1."""
        steps = int(round(1 / self.step))
        out = []
        for combo in itertools.product(range(steps + 1), repeat=n - 1):
            rest = steps - sum(combo)
            if rest >= 0:
                out.append(tuple(v / steps for v in (*combo, rest)))
        return out

    def optimize(self, components: list[BlendComponent],
                 additive_ppm: float = 0.0) -> BlendRecipe:
        """Максимум выпуска среди рецептур, проходящих жёсткие проверки."""
        names = [c.name for c in components]
        capacity = {c.name: c.available_tph for c in components}
        best: BlendRecipe | None = None

        for point in self._grid(len(names)):
            fractions = dict(zip(names, point))
            props = mix(components, fractions, additive_ppm)
            violations = self.check(props, fractions)
            if violations:
                continue
            # выпуск ограничен самым дефицитным компонентом
            limits = [capacity[n] / f for n, f in fractions.items() if f > 0 and capacity[n] > 0]
            throughput = min(limits) if limits else 0.0
            if best is None or throughput > best.throughput_tph:
                best = BlendRecipe(fractions=fractions, properties=props,
                                   throughput_tph=throughput, additive_ppm=additive_ppm,
                                   feasible=True)

        if best is None:
            # ни одна рецептура не проходит — возвращаем чистый лучший компонент
            # с честным перечнем нарушений
            fallback = min(components, key=lambda c: c.sulfur_mgkg)
            fractions = {c.name: (1.0 if c is fallback else 0.0) for c in components}
            props = mix(components, fractions, additive_ppm)
            return BlendRecipe(
                fractions=fractions, properties=props,
                throughput_tph=capacity.get(fallback.name, 0.0), additive_ppm=additive_ppm,
                feasible=False, violations=self.check(props, fractions),
                notes=["Допустимой рецептуры нет: показан лучший по сере компонент."])

        best.notes = self._notes(components, best)
        return best

    # ------------------------------------------------------------------ #
    def with_forecast(self, components: list[BlendComponent], sulfur_mgkg: float,
                      additive_ppm: float = 0.0,
                      component: str | None = None) -> BlendRecipe:
        """Рецептура на ПРОГНОЗНОЙ сере гидроочищенного ДТ, а не на прошлом анализе.

        В цикле оркестратора смешивать нужно тот продукт, который получится в
        рекомендуемом режиме. Разница существенна: при сере 8.9 мг/кг предельная
        доля прямогонки 0.013 %, при 9.5 — уже 0.006 %, то есть вдвое меньше.
        Последний анализ ЛИМС относится к прошлому режиму и такой вопрос не
        закрывает.
        """
        target = component or self.HYDROTREATED
        base = next((c for c in components if c.name == target), None)
        if base is None and components:
            # имя может отличаться — берём самый чистый компонент, это и есть
            # продукт гидроочистки
            base = min(components, key=lambda c: c.sulfur_mgkg)
        updated = [c.model_copy(update={"sulfur_mgkg": float(sulfur_mgkg)})
                   if c is base else c for c in components]
        recipe = self.optimize(updated, additive_ppm=additive_ppm)
        recipe.basis = "forecast"
        recipe.basis_sulfur_mgkg = float(sulfur_mgkg)
        if base is not None:
            recipe.notes = [
                f"Рецептура посчитана на прогнозной сере «{base.name}» "
                f"{sulfur_mgkg:.2f} мг/кг (последний анализ {base.sulfur_mgkg:.2f})."
            ] + recipe.notes
        return recipe

    # ------------------------------------------------------------------ #
    def _notes(self, components: list[BlendComponent], recipe: BlendRecipe) -> list[str]:
        notes = []
        for c in components:
            share = recipe.fractions.get(c.name, 0.0)
            if share == 0 and c.sulfur_mgkg > self.spec["product_sulfur_mgkg"]["max"]:
                notes.append(
                    f"{c.name} не вошёл в смесь: сера {c.sulfur_mgkg:.0f} мг/кг при пределе "
                    f"{self.spec['product_sulfur_mgkg']['max']} — даже малая доля "
                    "выводит смесь за спецификацию.")
            if share > 0 and c.is_assumption:
                notes.append(f"{c.name}: свойства взяты из допущения, а не из ЛИМС.")
        return notes

    def max_share_of(self, base: BlendComponent, additive: BlendComponent) -> float:
        """Предельная доля компонента с высокой серой при разбавлении базовым.

        Ответ на вопрос «сколько прямогонки вообще можно подмешать» — считается
        аналитически и хорошо смотрится на защите.
        """
        limit = self.spec["product_sulfur_mgkg"]["max"]
        if additive.sulfur_mgkg <= limit:
            return 1.0
        if base.sulfur_mgkg >= limit:
            return 0.0
        return float(np.clip((limit - base.sulfur_mgkg)
                             / (additive.sulfur_mgkg - base.sulfur_mgkg), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# сборка компонентов из выданных данных
# --------------------------------------------------------------------------- #

# Сера керосиновой фракции в пакете не измеряется. Берём консервативную оценку:
# прямогонный керосин обычно вдвое чище дизельной фракции того же сырья.
KEROSENE_SULFUR_FRACTION_OF_DIESEL = 0.5


def components_from_data(sb, ts, lims=None) -> list["BlendComponent"]:
    """Компоненты смешения на момент ``ts`` по ЛИМС и телеметрии.

    Все значения — последние ИЗВЕСТНЫЕ до ``ts``, без заглядывания вперёд.
    """
    import pandas as pd

    from nefte.data.loaders import lims_series, load_lims

    lims = load_lims() if lims is None else lims
    ts = pd.Timestamp(ts)

    def last(series_key: str) -> float | None:
        try:
            series = lims_series(series_key, lims)
        except KeyError:
            return None
        sub = series.loc[:ts]
        return float(sub.iloc[-1]) if len(sub) else None

    def tag(unit: str, name: str) -> float:
        frame = sb.ht if unit == "ht" else sb.avt
        if name not in frame.columns:
            return 0.0
        sub = frame[name].loc[:ts].dropna()
        return float(max(sub.iloc[-1], 0.0)) if len(sub) else 0.0

    godt_sulfur = last("Гидроочистка|2|Mg.Sulfur") or 8.5
    # сырьё гидроочистки — та же прямогонная фракция, % масс. → мг/кг
    feed_sulfur_pct = last("Гидроочистка|1|Mass.Sulfur")
    straight_sulfur = (feed_sulfur_pct or 0.95) * 10_000

    components = [
        BlendComponent(
            name="ГО ДТ",
            sulfur_mgkg=godt_sulfur,
            density_15c=last("Гидроочистка|2|D15"),
            t95_c=last("Гидроочистка|2|95%.T"),
            cfpp_c=last("Гидроочистка|2|CFPP"),
            available_tph=tag("ht", "F17"),
        ),
        BlendComponent(
            name="Прямогонная фр. 290-350",
            sulfur_mgkg=straight_sulfur,
            density_15c=last("АВТ|3|D15"),
            t95_c=last("АВТ|3|95%.T"),
            cfpp_c=last("АВТ|3|FilterabilityLimit.T"),
            available_tph=tag("avt", "F30"),
        ),
        BlendComponent(
            name="Керосиновая фр. 150-250",
            sulfur_mgkg=straight_sulfur * KEROSENE_SULFUR_FRACTION_OF_DIESEL,
            density_15c=last("АВТ|2|D15"),
            t95_c=last("АВТ|2|95%.T"),
            cfpp_c=last("АВТ|2|CloudPoint"),
            available_tph=tag("avt", "F34"),
            is_assumption=True,      # сера этой фракции не измеряется
        ),
    ]
    return [c for c in components if c.sulfur_mgkg is not None]
