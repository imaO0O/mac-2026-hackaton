"""Обучение и работа должны видеть одни и те же признаки.

Этот класс дефектов в проекте уже случался дважды: очистка телеметрии
существовала в двух экземплярах и разъехалась на 0.9 % значений, а метка времени
ЛИМС давала модели анализы, которых оператор ещё не видел. Здесь — третий случай
того же рода, найденный до того, как он сработал.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nefte.agents.schemas import DataQuality, ProcessState
from nefte.models.quality_model import SERVE_COMPUTED, SulfurModel


def _matrix(n: int = 200):
    index = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ht_T5": rng.normal(367.0, 5.0, n),
                         "ht_F17": rng.normal(248.0, 10.0, n)}, index=index)


def _state(ts):
    return ProcessState(ts=ts, telemetry_avt={}, telemetry_ht={}, quality={},
                        data_quality=DataQuality(missing_share=0.0, usable=True))


def _fit(features: pd.DataFrame, target):
    model = SulfurModel(iterations=40, monotone=False)
    return model.fit(features, target)


def test_serve_computed_feature_does_not_break_attach():
    """Главный случай: признак отобран, но в рабочей матрице его нет.

    `feature_age_h` предлагается отбору при обучении и в матрице отсутствовать
    обязан — он зависит от момента запроса. Пока отбор его не выбирал, всё
    работало; на следующем переобучении модель упала бы на загрузке, уже будучи
    подключённой к циклу.
    """
    matrix = _matrix()
    training = matrix.copy()
    training["feature_age_h"] = 1.0
    model = _fit(training, pd.Series(np.linspace(8.0, 9.0, len(matrix)),
                                     index=matrix.index))
    assert "feature_age_h" in model.features
    model.attach(matrix)          # не должно падать
    row = model._row_for(_state(matrix.index[100] + pd.Timedelta(minutes=30)))
    assert row is not None
    assert set(model.features) <= set(row.columns)


def test_serve_computed_feature_is_actually_computed():
    """Возраст строки признаков считается от момента запроса, а не берётся нулём.

    Ноль прошёл бы молча и означал бы «данные свежие» там, где они старые.
    """
    matrix = _matrix()
    training = matrix.copy()
    training["feature_age_h"] = 1.0
    model = _fit(training, pd.Series(np.linspace(8.0, 9.0, len(matrix)),
                                     index=matrix.index))
    model.attach(matrix)
    # момент МЕЖДУ отсчётами: ровно в узле сетки возраст и должен быть нулевым,
    # и такой тест ничего бы не проверил
    ts = matrix.index[100] + pd.Timedelta(minutes=40)
    row = model._row_for(_state(ts))
    assert row["feature_age_h"].iloc[0] == pytest.approx(40 / 60)


def test_missing_real_feature_fails_loudly():
    """А вот НАСТОЯЩЕГО пропавшего признака прощать нельзя.

    Раньше `attach` просто индексировал матрицу и падал с KeyError без объяснения;
    теперь причина названа, потому что означает она вполне конкретное: матрица и
    модель собраны по разным настройкам.
    """
    matrix = _matrix()
    model = _fit(matrix, pd.Series(np.linspace(8.0, 9.0, len(matrix)),
                                   index=matrix.index))
    with pytest.raises(RuntimeError, match="нет признаков модели"):
        model.attach(matrix.drop(columns=["ht_T5"]))


def test_every_working_model_feature_is_available_at_serve_time():
    """Сквозная проверка на рабочих моделях: ни одного признака «только для обучения».

    Тест намеренно ходит в обученные модели: он ловит расхождение, которое
    появится при следующем переобучении, а не только в синтетике.
    """
    from nefte.models.dataset import build_feature_matrix

    matrix = build_feature_matrix()
    for horizon in (0.0, 2.0):
        path = SulfurModel.default_path(horizon)
        if not path.exists():
            continue
        model = SulfurModel.load(path)
        unavailable = [f for f in model.features
                       if f not in matrix.columns and f not in SERVE_COMPUTED]
        assert not unavailable, (
            f"модель горизонта {horizon:g} ч использует признаки, которых нет "
            f"в рабочей матрице: {unavailable}")


def test_frozen_detection_is_causal():
    """Замороженный сигнал нельзя объявить раньше, чем порог набран по прошлому.

    Дефект был двойным. Как ПРИЗНАК (`pak_frozen` в матрице) полный размер прогона
    давал модели знание о будущем отказе прибора. Как СОСТОЯНИЕ в срезе оператора
    он делал систему прозорливой в бэктесте: она объявляла анализатор зависшим
    раньше, чем это стало бы известно в реальном времени.
    """
    from nefte.data.cleaning import frozen_mask

    values = pd.Series([1.0, 2.0] + [7.0] * 10 + [3.0])
    mask = frozen_mask(values, min_samples=5)
    first_true = int(np.argmax(mask.to_numpy()))
    # полка начинается с индекса 2; порог 5 набирается на индексе 6
    assert first_true == 6
    assert not mask.iloc[:6].any()


def test_frozen_detection_does_not_depend_on_what_comes_after():
    """Проверка на само заглядывание: будущее не должно менять прошлое.

    Берём один и тот же префикс и продолжаем его по-разному. Маска на префиксе
    обязана совпасть — иначе значение признака в момент t зависит от того, что
    случится позже.
    """
    from nefte.data.cleaning import frozen_mask

    prefix = [1.0, 2.0] + [7.0] * 6
    short = frozen_mask(pd.Series(prefix + [3.0]), min_samples=5)
    long = frozen_mask(pd.Series(prefix + [7.0] * 20), min_samples=5)
    assert list(short.iloc[:len(prefix)]) == list(long.iloc[:len(prefix)])
