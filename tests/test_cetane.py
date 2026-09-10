"""Цетановое число: норматив, присадка и её цена.

Организаторы назвали цетановое число третьим обязательным показателем качества и
дали два числа про присадку: доза не выше 3 % и цена в 100 раз выше дизеля за
тонну. Тесты закрепляют именно те свойства, из-за которых блок легко испортить
незаметно.
"""
from __future__ import annotations

import pytest

from nefte.agents.blending import BlendComponent, BlendingAgent, mix
from nefte.models.cetane import (
    CETANE_SPEC_MIN,
    IMPROVER_MAX_PCT,
    IMPROVER_PLATEAU,
    cetane_index_d976,
    dose_for_deficit,
    improver_cost_share,
    improver_effect,
    trend_per_year,
)

GOOD = BlendComponent(name="ГО ДТ", sulfur_mgkg=8.0, density_15c=836.0,
                      t95_c=345.0, cfpp_c=-6.0, cetane_number=54.0,
                      available_tph=250.0)
LOW_CETANE = GOOD.model_copy(update={"cetane_number": 50.0})
NO_CETANE = GOOD.model_copy(update={"cetane_number": None})


# --------------------------------------------------------------------------- #
# кривая отклика присадки
# --------------------------------------------------------------------------- #

def test_improver_saturates_instead_of_growing_forever():
    """Без насыщения оптимизатор лечил бы присадкой любую проблему.

    Линейный отклик обещал бы при 3 % дозы прибавку в сотни единиц, и рецептура с
    заведомо негодным компонентом выглядела бы исправимой.
    """
    assert improver_effect(3.0) <= IMPROVER_PLATEAU + 1e-9
    assert improver_effect(30.0) == pytest.approx(improver_effect(IMPROVER_MAX_PCT))


def test_improver_effect_is_monotone():
    doses = [0.0, 0.05, 0.1, 0.5, 1.0, 3.0]
    values = [improver_effect(d) for d in doses]
    assert values == sorted(values)


def test_dose_and_effect_are_inverse():
    """Назначенная доза обязана давать ровно ту прибавку, под которую назначена."""
    for deficit in (0.5, 1.0, 3.0, 6.0):
        dose = dose_for_deficit(deficit)
        assert dose is not None
        assert improver_effect(dose) == pytest.approx(deficit, abs=1e-6)


def test_unreachable_deficit_returns_nothing_not_a_useless_dose():
    """Нехватку выше плато не закрыть. Ответ — «нельзя», а не «лейте 3 %».

    Доза, которой не хватит, хуже отказа: оператор потратит деньги и всё равно
    получит брак.
    """
    assert dose_for_deficit(IMPROVER_PLATEAU + 1.0) is None
    assert dose_for_deficit(0.0) == 0.0


def test_improver_price_is_the_organizers_number():
    """0.1 % массы присадки стоит 10 % цены тонны топлива при отношении цен 100."""
    assert improver_cost_share(0.1) == pytest.approx(0.1)
    assert improver_cost_share(3.0) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# цетановый индекс: отрицательный результат тоже надо охранять
# --------------------------------------------------------------------------- #

def test_index_refuses_nonphysical_inputs():
    """В ЛИМС встречается Т50 = 0. В логарифм такое идти не должно."""
    assert cetane_index_d976(836.0, 0.0) is None
    assert cetane_index_d976(None, 270.0) is None
    assert cetane_index_d976(836.0, 270.0) is not None


def test_index_is_not_used_as_a_forecast():
    """На выданных данных индекс с лабораторией не связан (corr 0.04).

    Тест держит границу: индекс остаётся отдельной функцией для отчёта и НЕ
    подставляется в свойства смеси. Если кто-то решит «а давайте им заполним
    пропуски», тест это заметит.
    """
    props = mix([NO_CETANE], {"ГО ДТ": 1.0})
    assert "cetane_number" not in props


# --------------------------------------------------------------------------- #
# поведение агента смешения
# --------------------------------------------------------------------------- #

def test_below_spec_cetane_is_fixed_by_minimum_dose():
    recipe = BlendingAgent().optimize([LOW_CETANE])
    assert recipe.feasible
    assert recipe.cetane_improver_pct > 0
    assert recipe.properties["cetane_number"] == pytest.approx(CETANE_SPEC_MIN, abs=1e-6)


def test_improver_is_not_dosed_when_spec_is_already_met():
    """Присадка дорогая. «С запасом» — это выброшенные деньги."""
    recipe = BlendingAgent().optimize([GOOD])
    assert recipe.cetane_improver_pct == 0.0
    assert recipe.net_value_tph == pytest.approx(recipe.throughput_tph)


def test_net_value_is_lower_than_throughput_when_improver_is_needed():
    """Ради этого вся арифметика: рецептура с присадкой стоит дешевле."""
    recipe = BlendingAgent().optimize([LOW_CETANE])
    assert recipe.net_value_tph < recipe.throughput_tph


def test_agent_dilutes_instead_of_dosing_heavily():
    """Главный экономический выбор: разбавить дешевле, чем залить присадкой.

    Компонента с провальным цетановым числом много (1000 т/ч), хорошего мало
    (250 т/ч). По голому выпуску победила бы чистая «плохая» рецептура — выпуск
    вчетверо выше. Но чтобы вытянуть её цетановое число, нужна доза у самого плато,
    а она съедает почти всю стоимость продукта. Правильный ответ — смесь: разбавить
    хорошим компонентом и обойтись малой дозой.
    """
    plenty_bad = BlendComponent(
        name="много, но провальное ЦЧ", sulfur_mgkg=8.0, density_15c=836.0,
        t95_c=345.0, cfpp_c=-6.0, cetane_number=41.1, available_tph=1000.0)
    scarce_good = BlendComponent(
        name="мало, но хорошее ЦЧ", sulfur_mgkg=8.0, density_15c=836.0,
        t95_c=345.0, cfpp_c=-6.0, cetane_number=54.0, available_tph=250.0)
    recipe = BlendingAgent(step=0.1).optimize([plenty_bad, scarce_good])

    # чистая «плохая» рецептура даёт максимальный выпуск — и всё равно не выбрана
    assert 0.0 < recipe.fractions["много, но провальное ЦЧ"] < 1.0
    assert recipe.properties["cetane_number"] >= CETANE_SPEC_MIN - 1e-6
    # доза осталась далеко от предела: разбавление сделало основную работу
    assert recipe.cetane_improver_pct < IMPROVER_MAX_PCT / 10
    assert recipe.net_value_tph > 0


def test_missing_cetane_is_reported_not_silently_passed():
    """«Не посчитали» — это не «годно», но и не «нарушение».

    Серу меряют каждые 10 минут, и её отсутствие означает поломку сбора данных.
    Цетановое число меряют раз в месяц по регламенту, и его отсутствие — обычное
    состояние. Поэтому оно уходит в «не подтверждено»: видно оператору, но не
    блокирует рецептуру целиком.
    """
    recipe = BlendingAgent().optimize([NO_CETANE])
    assert recipe.feasible
    assert "цетановое число" in recipe.uncertified
    assert any("Не подтверждено" in note for note in recipe.notes)


def test_partial_cetane_does_not_average_over_the_components_that_have_it():
    """Усреднить по половине компонентов — значит описать ДРУГУЮ смесь.

    Если у прямогонки анализа нет, а у гидроочищенного есть, «среднее» окажется
    просто числом гидроочищенного и будет выглядеть измеренным свойством смеси.
    """
    props = mix([GOOD, NO_CETANE.model_copy(update={"name": "без анализа"})],
                {"ГО ДТ": 0.5, "без анализа": 0.5})
    assert "cetane_number" not in props


def test_zero_share_component_without_cetane_does_not_block_certification():
    """Компонент с нулевой долей в смесь не входит и мешать не должен."""
    props = mix([GOOD, NO_CETANE.model_copy(update={"name": "не используем"})],
                {"ГО ДТ": 1.0, "не используем": 0.0})
    assert props["cetane_number"] == pytest.approx(54.0)


def test_trend_catches_degradation():
    """Падение показателя между редкими анализами обязано быть видно."""
    import pandas as pd

    index = pd.date_range("2023-01-01", periods=40, freq="30D")
    series = pd.Series(56.0 - 0.1 * pd.RangeIndex(40), index=index, dtype="float64")
    slope = trend_per_year(series)
    assert slope < 0
    # 0.1 единицы за 30 суток — это примерно 1.2 единицы в год
    assert slope == pytest.approx(-1.22, abs=0.05)


# --------------------------------------------------------------------------- #
# конфиг — единственный источник чисел
# --------------------------------------------------------------------------- #

def test_numbers_come_from_config_not_from_the_module():
    """Норматив и цена присадки живут в конфиге, а не рядом с ним.

    В этом проекте уже был случай, когда вердикты по тегам лежали в конфиге, а код
    о них не знал: запрещённый тег держался вне признаков исключительно по
    договорённости. Здесь та же ловушка — цену организаторов легко записать в
    конфиг «для документации» и продолжать считать по константе в модуле.
    Тест сверяет, что число ровно одно.
    """
    from nefte.config import load_config

    cfg = load_config()
    assert CETANE_SPEC_MIN == cfg["spec"]["cetane_number"]["min"]
    assert IMPROVER_MAX_PCT == cfg["economics"]["cetane_improver"]["max_pct"]
    assert improver_cost_share(1.0) == pytest.approx(
        cfg["economics"]["cetane_improver"]["price_ratio"] / 100.0)


def test_improver_price_is_declared_as_a_fact_not_an_assumption():
    """Цена и предел дозы — ответ организаторов. Пометка обязана это отражать.

    Всё остальное в блоке смешения помечено assumption: true. Если и эти два числа
    уедут в допущения, на защите пропадёт единственная опора экономики.
    """
    from nefte.config import load_config

    block = load_config()["economics"]["cetane_improver"]
    assert block["assumption"] is False
    assert "организатор" in block["source"].lower()
