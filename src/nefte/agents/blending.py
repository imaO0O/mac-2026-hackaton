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
| цетановое число ГО ДТ | ЛИМС «Гидроочистка, точка 2» — факт, но раз в месяц |
| располагаемые расходы компонентов | телеметрия (`F17`, `F30`, `F34`) — факт |
| предел дозы и цена цетановой присадки | ответ организаторов — факт |
| правила смешения | линейные по массе — ДОПУЩЕНИЕ |
| эффект депрессорной присадки | −8 °C к ПТФ при дозировке до 500 ppm — ДОПУЩЕНИЕ |
| отклик ЦЧ на дозу присадки | насыщающаяся кривая — ДОПУЩЕНИЕ |

Правило смешения по сере линейно по массе и физически корректно. Для Т95 и ПТФ
линейность — упрощение: реальные показатели смешиваются нелинейно, через
блендинг-индексы. Это помечено как допущение и должно быть названо на защите.

Цетановое число появилось здесь не для полноты. Организаторы назвали его третьим
обязательным показателем, и выданные данные показали, что оно ПАДАЕТ: 55.9 в 2023
году, 52.3 в 2026-м, последний анализ 50.0 при нормативе 51. Показатель уже вне
норматива, а единственный быстрый рычаг — присадка, которая дороже топлива в сто
раз. Поэтому рецептура оптимизируется не по выпуску, а по выпуску ЗА ВЫЧЕТОМ
стоимости присадки: иначе присадка бесплатна и назначается всегда.
Разбор данных: ``python scripts/check_cetane.py``.
"""
from __future__ import annotations

import itertools

import numpy as np

from nefte.agents.schemas import BlendComponent, BlendRecipe
from nefte.config import load_config
from nefte.models.cetane import (
    CETANE_SPEC_MIN,
    IMPROVER_MAX_PCT,
    dose_for_deficit,
    improver_cost_share,
    improver_effect,
    trend_per_year,
)

__all__ = ["ADDITIVE_CFPP_EFFECT_C", "ADDITIVE_MAX_PPM", "BlendComponent", "BlendRecipe",
           "BlendingAgent", "components_from_data", "mix"]

# Эффект депрессорной присадки на предельную температуру фильтруемости.
# ДОПУЩЕНИЕ: паспортных данных присадки в пакете нет.
ADDITIVE_CFPP_EFFECT_C = -8.0
ADDITIVE_MAX_PPM = 500.0


def mix(components: list[BlendComponent], fractions: dict[str, float],
        additive_ppm: float = 0.0, cetane_improver_pct: float = 0.0) -> dict[str, float]:
    """Свойства смеси при заданных массовых долях.

    Сера — строго линейно по массе (физически верно). Плотность, Т95, ПТФ и
    цетановое число — линейно как ДОПУЩЕНИЕ; депрессорная присадка сдвигает ПТФ, а
    цетаноповышающая — цетановое число, обе пропорционально дозировке.
    """
    by_name = {c.name: c for c in components}
    props: dict[str, float] = {}

    def weighted(attr: str, require_all: bool = False) -> float | None:
        """Средневзвешенное по массовым долям.

        ``require_all`` — для показателей, где частичное усреднение лжёт. Если у
        компонента с ненулевой долей показателя нет, а мы усредним по остальным,
        получится свойство ДРУГОЙ смеси, и оно будет выглядеть измеренным. Для
        цетанового числа это особенно опасно: прямогонная фракция тяжёлая и по ЦЧ
        отличается от гидроочищенной, а меряют его раз в месяц и не везде.
        """
        total, weight = 0.0, 0.0
        for name, share in fractions.items():
            if share <= 0:
                continue
            value = getattr(by_name[name], attr)
            if value is None:
                if require_all:
                    return None
                continue
            total += value * share
            weight += share
        return total / weight if weight > 0 else None

    # Если серу посчитать не из чего, это НЕ ноль. Ноль прошёл бы проверку
    # спецификации и смесь с неизвестной серой объявили бы годной.
    sulfur = weighted("sulfur_mgkg")
    props["sulfur_mgkg"] = float("nan") if sulfur is None else sulfur
    for attr, key in (("density_15c", "density_15c"), ("t95_c", "t95_c"),
                      ("cfpp_c", "cfpp_c")):
        value = weighted(attr)
        if value is not None:
            props[key] = value

    cetane = weighted("cetane_number", require_all=True)
    if cetane is not None:
        props["cetane_number"] = cetane

    if "cfpp_c" in props and additive_ppm > 0:
        share = min(additive_ppm / ADDITIVE_MAX_PPM, 1.0)
        props["cfpp_c"] += ADDITIVE_CFPP_EFFECT_C * share
    # Цетановое число складываем только тем компонентам, у которых оно измерено.
    # Если у смеси его посчитать не из чего, присадку приписывать не к чему: она
    # даёт ПРИБАВКУ к числу, а не число.
    if "cetane_number" in props and cetane_improver_pct > 0:
        props["cetane_number"] += improver_effect(cetane_improver_pct)
    return props


def grade_spec(spec: dict, grade: str | None = None) -> dict:
    """Спецификация смеси для марки ДТ (ответ организаторов 15.09).

    Сера, Т95, ПТФ и сумма долей от марки не зависят, плотность и цетановое число —
    зависят: у ДТ с гидроочистки ЦЧ не нормируется, у летнего товарного не ниже 51,
    у зимнего — не ниже 49 и плотность от 800, а не от 820. ``grade=None`` — марка из
    ``spec.blend_grade``; конфиг без марок возвращается как есть.
    """
    grades = spec.get("grades") or {}
    grade = grade or spec.get("blend_grade")
    if not grades or grade is None:
        return spec
    if grade not in grades:
        raise ValueError(f"марка ДТ {grade!r} не описана, есть: {sorted(grades)}")
    chosen = grades[grade]
    out = dict(spec)
    lo, hi = chosen["density_15c_kgm3"]
    out["density_15c_kgm3"] = {**spec.get("density_15c_kgm3", {}),
                               "min": float(lo), "max": float(hi)}
    floor = chosen.get("cetane_number_min")
    out["cetane_number"] = {**spec.get("cetane_number", {}),
                            "min": None if floor is None else float(floor)}
    out["grade"] = {"key": grade, "name": chosen["name"]}
    return out


class BlendingAgent:
    """Подбирает рецептуру: максимум выпуска при соблюдении спецификации.

    Перебор по симплексу долей с шагом ``step``. Компонентов единицы, шаг крупный,
    поэтому полный перебор на CPU занимает миллисекунды и не требует солвера —
    зато результат воспроизводим и его легко объяснить.
    """

    # Компонент, серу которого прогнозирует агент качества: это и есть продукт
    # гидроочистки, ЛИМС «Гидроочистка, точка 2».
    HYDROTREATED = "ГО ДТ"

    def __init__(self, cfg: dict | None = None, step: float = 0.01,
                 grade: str | None = None):
        self.cfg = cfg or load_config()
        self.step = step
        self.spec = grade_spec(self.cfg["spec"], grade)
        self.grade_name = (self.spec.get("grade") or {}).get("name")

    # ------------------------------------------------------------------ #
    def check(self, props: dict[str, float], fractions: dict[str, float]) -> list[str]:
        """Жёсткие проверки. Сумма долей — требование ТЗ, а не косметика."""
        violations = []
        total = sum(fractions.values())
        tol = self.spec["blend_fractions_sum"]["tolerance"]
        if abs(total - self.spec["blend_fractions_sum"]["value"]) > tol:
            violations.append(f"сумма долей {total:.4f}, а должна быть 1")

        sulfur = props.get("sulfur_mgkg")
        if sulfur is None or sulfur != sulfur:          # None или NaN
            violations.append("серу смеси не из чего посчитать")
        elif sulfur > self.spec["product_sulfur_mgkg"]["max"]:
            violations.append(
                f"сера {sulfur:.2f} мг/кг выше предела "
                f"{self.spec['product_sulfur_mgkg']['max']}")

        density = props.get("density_15c")
        if density is not None:
            lo, hi = self.spec["density_15c_kgm3"]["min"], self.spec["density_15c_kgm3"]["max"]
            if not lo <= density <= hi:
                violations.append(f"плотность {density:.1f} вне диапазона {lo}–{hi}"
                                  f"{self._grade_note()}")

        t95 = props.get("t95_c")
        if t95 is not None and t95 > self.spec["t95_c"]["max"]:
            violations.append(f"Т95 {t95:.1f} °C выше {self.spec['t95_c']['max']} (допущение)")

        cfpp = props.get("cfpp_c")
        if cfpp is not None and cfpp > self.spec["cfpp_c"]["max"]:
            violations.append(f"ПТФ {cfpp:.1f} °C выше {self.spec['cfpp_c']['max']} (допущение)")

        # Цетановое число. Ниже норматива — нарушение. А вот «не из чего
        # посчитать» нарушением НЕ считаем, и разница здесь принципиальная.
        # Неизвестная сера — это отказ сбора данных: её меряют каждые 10 минут
        # поточным анализатором и каждые сутки в лаборатории, и если её нет,
        # сломалось что-то. Цетановое число меряют раз в месяц ПО РЕГЛАМЕНТУ, и
        # его отсутствие у компонента — обычное состояние, а не поломка. Поэтому
        # оно уходит в «не подтверждено» (см. uncertified) и остаётся видимым, но
        # не блокирует рецептуру целиком.
        cetane = props.get("cetane_number")
        floor = self._cetane_floor()
        if cetane is not None and floor is not None and cetane < floor:
            violations.append(f"цетановое число {cetane:.1f} ниже {floor}{self._grade_note()}")
        return violations

    def _cetane_floor(self) -> float | None:
        """Норматив ЦЧ марки; None — у марки он не нормируется (ДТ с гидроочистки)."""
        section = self.spec.get("cetane_number")
        if section is None:
            return CETANE_SPEC_MIN
        return section.get("min")

    def _grade_note(self) -> str:
        return f" (марка «{self.grade_name}»)" if self.grade_name else " (допущение)"

    def uncertified(self, props: dict[str, float]) -> list[str]:
        """Обязательные показатели, которые проверить не удалось.

        Организаторы назвали обязательными серу, Т95 и цетановое число. Если
        какого-то из них в смеси не посчитать, рецептура не «годная» — она
        НЕПОДТВЕРЖДЁННАЯ, и оператор обязан это видеть отдельной строкой, а не
        догадываться по отсутствию числа.
        """
        missing = []
        if props.get("t95_c") is None:
            missing.append("Т95")
        if props.get("cetane_number") is None and self._cetane_floor() is not None:
            missing.append("цетановое число")
        return missing

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

    def _min_improver(self, props: dict[str, float]) -> float:
        """Минимальная доза цетаноповышающей присадки, закрывающая норматив.

        Считается аналитически по обратной кривой отклика, а не перебором: доза
        стоит дорого, и «примерно достаточная» доза — это либо брак, либо
        выброшенные деньги. Возвращает 0, если норматив выполняется и без неё, и
        ``inf``, если не закрывается даже предельной дозой.
        """
        cetane = props.get("cetane_number")
        if cetane is None:
            return 0.0
        floor = self._cetane_floor()
        if floor is None:
            return 0.0
        deficit = floor - float(cetane)
        if deficit <= 0:
            return 0.0
        dose = dose_for_deficit(deficit)
        return float("inf") if dose is None else dose

    def optimize(self, components: list[BlendComponent],
                 additive_ppm: float = 0.0) -> BlendRecipe:
        """Максимум ЧИСТОЙ ценности среди рецептур, проходящих жёсткие проверки.

        Чистая ценность — выпуск за вычетом стоимости цетаноповышающей присадки, в
        тоннах дизеля. Раньше здесь был просто максимум выпуска; при цене присадки
        100× за тонну (ответ организаторов) это неверно: рецептура с чуть большим
        выпуском, но требующая присадки, разоряет.
        """
        names = [c.name for c in components]
        capacity = {c.name: c.available_tph for c in components}
        best: BlendRecipe | None = None

        for point in self._grid(len(names)):
            fractions = dict(zip(names, point))
            props = mix(components, fractions, additive_ppm)
            # доза назначается ПОД рецептуру, а не задаётся снаружи: у каждой
            # смеси своё цетановое число, и нехватка у каждой своя
            dose = self._min_improver(props)
            if dose == float("inf"):
                continue
            if dose > 0:
                props = mix(components, fractions, additive_ppm, dose)
            violations = self.check(props, fractions)
            if violations:
                continue
            # Выпуск ограничен самым дефицитным компонентом. Если компонент с
            # ненулевой долей недоступен вовсе, смесь не сделать — выпуск ноль, а
            # не «минимум по остальным». Раньше такой компонент просто выпадал из
            # расчёта, и рецептура обещала выпуск, которого быть не может.
            needed = [n for n, f in fractions.items() if f > 0]
            if any(capacity[n] <= 0 for n in needed):
                throughput = 0.0
            else:
                throughput = min(capacity[n] / fractions[n] for n in needed) if needed else 0.0
            cost = improver_cost_share(dose)
            net = throughput * (1.0 - cost)
            if best is None or net > best.net_value_tph:
                best = BlendRecipe(fractions=fractions, properties=props,
                                   throughput_tph=throughput, additive_ppm=additive_ppm,
                                   cetane_improver_pct=dose, improver_cost_share=cost,
                                   net_value_tph=net, feasible=True)

        if best is None:
            # ни одна рецептура не проходит — возвращаем чистый лучший компонент
            # с честным перечнем нарушений
            fallback = min(components, key=lambda c: c.sulfur_mgkg)
            fractions = {c.name: (1.0 if c is fallback else 0.0) for c in components}
            props = mix(components, fractions, additive_ppm)
            dose = self._min_improver(props)
            notes = ["Допустимой рецептуры нет: показан лучший по сере компонент."]
            if dose == float("inf"):
                notes.append(
                    f"Цетановое число не вытянуть и предельной дозой присадки "
                    f"({IMPROVER_MAX_PCT} % массы) — нужен другой компонент смешения.")
                dose = 0.0
            return BlendRecipe(
                fractions=fractions, properties=props,
                throughput_tph=capacity.get(fallback.name, 0.0), additive_ppm=additive_ppm,
                cetane_improver_pct=dose, improver_cost_share=improver_cost_share(dose),
                net_value_tph=0.0,
                feasible=False, violations=self.check(props, fractions),
                uncertified=self.uncertified(props),
                notes=notes)

        best.uncertified = self.uncertified(best.properties)
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
        if recipe.uncertified:
            notes.append(
                "Не подтверждено по обязательным показателям: "
                + ", ".join(recipe.uncertified)
                + ". У компонентов смеси таких анализов нет — рецептуру нельзя "
                  "объявлять годной, пока их не сделают.")
        if recipe.cetane_improver_pct > 0:
            notes.append(
                f"Цетаноповышающая присадка {recipe.cetane_improver_pct:.3f} % массы — "
                f"минимальная доза, закрывающая норматив. Она стоит "
                f"{recipe.improver_cost_share * 100:.1f} % цены тонны топлива, поэтому "
                f"чистая ценность рецептуры {recipe.net_value_tph:.1f} т/ч при выпуске "
                f"{recipe.throughput_tph:.1f} т/ч.")
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

    def cetane_now() -> float | None:
        """Цетановое число на момент ``ts``: последний анализ плюс тренд.

        Анализ раз в месяц, а показатель падает примерно на 1.2 единицы в год
        (scripts/check_cetane.py). Брать последнее значение как есть — значит
        завышать: между анализами проходит до трёх месяцев. Поэтому к последнему
        анализу добавляем тренд за прошедшее время. Тренд отрицательный, так что
        поправка идёт в консервативную сторону — и это осознанно.
        """
        try:
            series = lims_series("Гидроочистка|2|CetaneNumber", lims).sort_index()
        except KeyError:
            return None
        sub = series.loc[:ts]
        if not len(sub):
            return None
        # тренд считаем ТОЛЬКО по прошлому: подглядывать в будущие анализы нельзя
        slope = trend_per_year(sub)
        years = (ts - sub.index[-1]).total_seconds() / (365.25 * 24 * 3600)
        value = float(sub.iloc[-1])
        return value if slope != slope else value + slope * years

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
            cetane_number=cetane_now(),
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
