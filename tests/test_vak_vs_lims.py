"""Сверка формул справочника с лабораторией.

Проверка правдоподобия отвечает на вопрос «похоже ли это на физическую величину»
и ловит бессмыслицу вроде 198 000 °C. Она НЕ отличает правильную величину от
чужой: плотность 869 кг/м³ выглядит нормально, даже если поток имеет 847.

Дыру нашёл участник 2 на признаке, который стоит в рабочей модели. Тесты
закрепляют механику сверки — те места, где её легко испортить незаметно.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_vak_against_lims",
    Path(__file__).resolve().parents[1] / "scripts" / "check_vak_against_lims.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

compare = _MODULE.compare
POINTS_BY_UNIT = _MODULE.POINTS_BY_UNIT
MIN_PAIRS_SHARE = _MODULE.MIN_PAIRS_SHARE


def _series(values, start="2025-01-01", freq="1h"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq))


def test_constant_offset_is_reported_and_does_not_hide_in_correlation():
    """Главный случай: формула отслеживает величину, но врёт в уровне.

    Именно так ведёт себя vak_AVT6_240_350_D15 — корреляция 0.41 при смещении
    +22 кг/м³. Если смотреть только на корреляцию, признак выглядит здоровым.
    """
    lab = _series(np.linspace(840.0, 855.0, 60))
    feature = _series(lab.to_numpy() + 22.0, freq="1h")
    stats = compare(feature, lab)
    assert stats["смещение"] == pytest.approx(22.0, abs=0.5)
    assert stats["корреляция"] > 0.95


def test_feature_is_matched_to_the_last_value_before_the_analysis():
    """Сверять надо то, что система знала на момент отбора пробы.

    Если брать ближайшее значение, а не последнее предшествующее, в сравнение
    попадёт будущее — та же утечка, что была с меткой времени ЛИМС. Каждый анализ
    здесь поставлен на полчаса позже своего отсчёта, поэтому правильный ответ —
    значение ДО него, а ближайшее по времени было бы следующим.
    """
    feature = _series(np.arange(1.0, 13.0))
    stamps = feature.index[:8] + pd.Timedelta(minutes=30)
    lab = pd.Series(np.arange(100.0, 108.0), index=stamps)
    stats = compare(feature, lab)
    # взяты 1..8, а не 2..9
    assert stats["среднее форм."] == pytest.approx(np.arange(1.0, 9.0).mean())


def test_no_pairs_gives_nothing_not_zero():
    """Пустое сравнение обязано вернуть None, а не смещение 0."""
    feature = _series([1.0, 2.0, 3.0])
    lab = pd.Series([5.0], index=[feature.index[0] - pd.Timedelta(days=1)])
    assert compare(feature, lab) is None


def test_duplicate_lab_timestamps_do_not_double_count():
    """Повторный анализ той же пробы — уточнение, а не второе наблюдение."""
    feature = _series(np.full(20, 5.0))
    stamps = list(feature.index[10:16])
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    # последний отсчёт продублирован с уточнённым значением
    lab = pd.Series(values + [99.0], index=stamps + [stamps[-1]])
    stats = compare(feature, lab)
    assert stats["пар"] == len(stamps)
    assert stats["среднее лаб."] == pytest.approx(
        np.mean(values[:-1] + [99.0]))


def test_candidate_points_are_split_by_unit():
    """Формулу АВТ нельзя сверять с продуктом гидроочистки.

    Пока кандидаты не были разделены по установкам, победителем становилась
    Гидроочистка|2 — просто потому, что у неё больше всего анализов.
    """
    assert all(p.startswith("АВТ") for p in POINTS_BY_UNIT["avt"])
    assert all(p.startswith("Гидроочистка") for p in POINTS_BY_UNIT["ht"])
    assert not set(POINTS_BY_UNIT["avt"]) & set(POINTS_BY_UNIT["ht"])


def test_pair_count_floor_exists():
    """Порог по объёму не даёт мелкой выборке выиграть отбор точки.

    У 24-2000:GODT:T90 корреляция с АВТ|2 равна 0.44 на 204 парах против 0.14 со
    своей точкой на 1251. Без порога скрипт объявил бы формулу гидроочистки
    формулой АВТ — то есть сделал бы ровно ту ошибку, которую призван ловить.
    """
    assert 0.0 < MIN_PAIRS_SHARE <= 1.0
