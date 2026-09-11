"""Монотонные ограничения и уровень цели.

Дефект нашёлся при попытке обучить вторую модель — на Т95. С монотонными
ограничениями CatBoost теряет автоматическое начальное приближение и начинает
подъём от нуля. На сере (уровень 8.5 мг/кг) это стоило смещения в 0.28 мг/кг и
было незаметно; на Т95 (уровень 347 °C) модель не доезжала до уровня вовсе —
MAE 90 против 5.3.

То есть ограничения физики, поставленные ради правильного НАПРАВЛЕНИЯ отклика,
молча портили УРОВЕНЬ прогноза. Для серы это опаснее, чем звучит: смещение вниз
означает, что продукт выглядит чище, чем он есть.

Лечится центрированием цели на медиану обучающей выборки. Тесты закрепляют и
факт лечения, и то, что физика при этом сохраняется.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.models.quality_model import SulfurModel


def _synthetic(level: float, n: int = 600, seed: int = 0):
    """Данные с известным уровнем и известным направлением отклика."""
    rng = np.random.default_rng(seed)
    temp = rng.normal(367.0, 20.0, n)
    other = rng.normal(248.0, 36.0, n)
    # выше температура — ниже сера: тот же знак, что зашит в PHYSICS_MONOTONE
    y = level - 0.1 * (temp - 367.0) + rng.normal(0.0, 0.5, n)
    X = pd.DataFrame({"ht_T5": temp, "ht_F17": other})
    return X, pd.Series(y, index=X.index)


@pytest.mark.parametrize("level", [8.5, 347.0])
def test_monotone_model_reaches_the_level_of_the_target(level: float):
    """Прогноз обязан попадать в уровень цели, а не в окрестность нуля.

    Без центрирования на уровне 347 средний прогноз получался около 268.
    """
    X, y = _synthetic(level)
    model = SulfurModel(iterations=200, monotone=True).fit(X, y)
    predicted = model.predict_frame(X)["q50"]
    assert abs(float(predicted.mean()) - float(y.mean())) < 1.0


def test_offset_is_taken_only_when_constraints_are_active():
    """Без ограничений начальное приближение работает само — сдвиг не нужен.

    Лишний сдвиг не сломал бы прогноз, но спрятал бы причину: увидев ненулевой
    offset у неограниченной модели, следующий читатель решит, что центрирование
    нужно всегда, и не найдёт настоящего дефекта.
    """
    X, y = _synthetic(347.0)
    assert SulfurModel(iterations=50, monotone=False).fit(X, y).y_offset == 0.0
    assert SulfurModel(iterations=50, monotone=True).fit(X, y).y_offset > 0.0


def test_offset_survives_save_and_load(tmp_path):
    """Сдвиг живёт в meta.json: без него загруженная модель врёт на весь уровень."""
    X, y = _synthetic(347.0)
    model = SulfurModel(iterations=100, monotone=True).fit(X, y)
    model.save(tmp_path / "m")
    restored = SulfurModel.load(tmp_path / "m")
    assert restored.y_offset == pytest.approx(model.y_offset)
    assert restored.predict_frame(X)["q50"].mean() == pytest.approx(
        model.predict_frame(X)["q50"].mean(), abs=1e-6)


def test_physics_direction_survives_centering():
    """Центрирование не должно отменять то, ради чего ограничения ставились."""
    X, y = _synthetic(347.0)
    model = SulfurModel(iterations=300, monotone=True).fit(X, y)
    cold = X.assign(ht_T5=X["ht_T5"] - 5.0)
    hot = X.assign(ht_T5=X["ht_T5"] + 5.0)
    # выше температура — ниже сера, и ограничение обязано это удержать
    assert (model.predict_frame(hot)["q50"].mean()
            < model.predict_frame(cold)["q50"].mean())
