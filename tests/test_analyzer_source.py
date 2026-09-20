"""Оперативный источник серы: какой из двух поточных анализаторов берём.

В телеметрии 24-2000 есть тег `Q21` — «поточный анализатор серы в г/о ДТ», и это
ВТОРОЙ прибор, а не копия ряда из файла анализаторов: между собой они связаны слабо
(corr 0.41). Пока описания в листе «КИП» стояли не на своих кодах, прибора для нас
не существовало вовсе (`docs/DATA_NOTES.md` §5б).

Выбор сделан по цене ошибки, названной заказчиком: пропущенная некондиция дороже
лишней проверки в 50–100 раз. Тесты держат то, что от этого зависит: значение из
конфига доходит до слияния источников, оба анализатора попадают в срез, а приоритет
лаборатории не нарушается ни при каком выборе.
"""
from __future__ import annotations

import pytest

from nefte.agents.quality import fuse_sulfur
from nefte.agents.schemas import Measurement, Source
from nefte.config import load_config
from tests.test_agents import make_state


def _state(lims=None, pak=None, q21=None, lims_age: float = 1.0):
    state = make_state()
    state.quality = {}
    if lims is not None:
        state.quality["lims_sulfur_mgkg"] = Measurement(
            value=lims, unit="мг/кг", source=Source.LIMS, age_hours=lims_age)
    if pak is not None:
        state.quality["pak_sulfur_ppm"] = Measurement(
            value=pak, unit="мг/кг", source=Source.PAK, age_hours=0.0)
    if q21 is not None:
        state.quality["q21_sulfur_ppm"] = Measurement(
            value=q21, unit="мг/кг", source=Source.PAK, age_hours=0.0)
    return state


def _cfg(source: str) -> dict:
    cfg = load_config()
    return {**cfg, "quality": {**cfg["quality"], "analyzer_source": source}}


def test_config_selects_which_analyzer_is_operational():
    state = _state(pak=8.0, q21=11.0)
    assert fuse_sulfur(state, _cfg("pak")).value == pytest.approx(8.0)
    assert fuse_sulfur(state, _cfg("q21")).value == pytest.approx(11.0)


def test_lab_still_wins_over_any_analyzer():
    """Приоритет ТЗ (ЛИМС → ПАК) не зависит от того, какой анализатор выбран."""
    state = _state(lims=9.0, pak=8.0, q21=11.0)
    for source in ("pak", "q21"):
        fused = fuse_sulfur(state, _cfg(source))
        assert fused.source is Source.LIMS
        assert fused.value == pytest.approx(9.0)


def test_falls_back_to_the_file_series_when_q21_is_absent():
    """На срезе без Q21 (старые данные, другой тег) остаётся ряд из файла."""
    state = _state(pak=8.5)
    assert fuse_sulfur(state, _cfg("q21")).value == pytest.approx(8.5)


def test_default_is_the_measured_choice():
    """Умолчание — результат счёта, а не вкуса: q21 дешевле по цене ошибок."""
    assert load_config()["quality"]["analyzer_source"] == "q21"
