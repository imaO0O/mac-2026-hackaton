# -*- coding: utf-8 -*-
"""WABT считается в трёх местах проекта и обязана быть одной величиной.

Места: ``data/features.wabt`` (общая функция), ``models/regime.instant_features``
(признак `reg_wabt` для модели качества) и ``agents/reliability`` (вход индекса
тяжести). Две последние считают её независимо, одним и тем же выражением
``ht[temps].mean(axis=1)`` — и совпадают они сейчас по совпадению, а не по
устройству: список тегов объявлен в обоих модулях отдельно.

Тот же класс расхождения уже стрелял в проекте дважды: σ Т95 держалась копией
константы в оптимизаторе, возраст лабораторного анализа считался тремя способами
в одной таблице. Оба раза копии разъезжались молча.
"""
from __future__ import annotations

import pytest


def test_the_reactor_tag_lists_agree():
    """Список температур реакторного блока объявлен дважды — значения обязаны совпасть."""
    from nefte.agents.reliability import ReliabilityAgent
    from nefte.models.regime import REACTOR_TEMPS

    assert list(REACTOR_TEMPS) == list(ReliabilityAgent.REACTOR_TEMPS), (
        f"режим считает WABT по {list(REACTOR_TEMPS)}, а индекс тяжести по "
        f"{list(ReliabilityAgent.REACTOR_TEMPS)}: одна величина по разным датчикам")


def test_missing_sensor_does_not_drag_the_average_down():
    """Отказ датчика не должен занижать WABT — он должен выпадать из среднего.

    Здесь была ошибка: веса нормировались по полному набору столбцов, а пропуск
    входил в сумму нулём. На трёх датчиках отказ одного давал 233 °C вместо 350.
    Опасна она тем, что возвращала не пропуск, а правдоподобное число.
    """
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from nefte.data.features import wabt

    frame = pd.DataFrame({"T5": [340.0, 340.0], "T6": [350.0, np.nan],
                          "T11": [360.0, 360.0]})
    result = wabt(frame)
    assert result.iloc[0] == pytest.approx(350.0)
    assert result.iloc[1] == pytest.approx(350.0), (
        f"при отказе одного датчика получилось {result.iloc[1]:.1f} °C вместо 350: "
        "пропуск снова считается нулём")


def test_all_sensors_missing_gives_a_gap_not_a_number():
    """Если не осталось ни одного датчика, ответ — пропуск, а не ноль."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from nefte.data.features import wabt

    frame = pd.DataFrame({"T5": [np.nan], "T6": [np.nan], "T11": [np.nan]})
    assert pd.isna(wabt(frame).iloc[0])


def test_the_shared_function_agrees_with_what_the_agents_actually_compute():
    """Общая функция и живой расчёт агентов обязаны давать одно и то же.

    Равные веса — это среднее, и `ht[temps].mean(axis=1)` обязано совпасть с
    `wabt(...)` во всех случаях, включая пропуски. Пока это так, три реализации
    остаются одной величиной; разойдутся — тест скажет раньше, чем разойдутся
    числа в отчётах.
    """
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from nefte.data.features import wabt

    rng = np.random.default_rng(7)
    values = rng.normal(350.0, 5.0, size=(200, 3))
    values[rng.random(values.shape) < 0.15] = np.nan
    frame = pd.DataFrame(values, columns=["T5", "T6", "T11"])

    shared, live = wabt(frame), frame.mean(axis=1)
    both = shared.notna() & live.notna()
    assert both.sum() > 100, "слишком мало сравнимых строк — проверка вырождена"
    assert np.allclose(shared[both], live[both]), (
        "общая функция WABT разошлась с тем, что считают агенты")
    assert (shared.isna() == live.isna()).all(), (
        "пропуски у общей функции и у агентов возникают в разных местах")
