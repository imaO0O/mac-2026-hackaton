"""Тесты агента качества: контракт модели и защита от утечки из будущего.

Ловят ровно ту ошибку, которая уже один раз случилась: почасовой ресемплинг с
меткой по левой границе тянул в строку данные из её собственного будущего и давал
«идеальную» модель с MAE 0.26 и AUC 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.models.quality_model import SulfurModel, Z90, make_model_surrogate


def _frame(values, start="2024-01-01", freq="10min"):
    idx = pd.date_range(start, periods=len(values), freq=freq, name="date")
    return pd.DataFrame({"x": values}, index=idx)


def test_resample_label_does_not_leak_future():
    """Строка с меткой t обязана содержать только данные (t-freq, t]."""
    df = _frame(np.arange(12, dtype=float))          # 10-мин сетка, 2 часа
    hourly = df.resample("1h", label="right", closed="right").last()

    # значение в 01:00 — последнее ДО 01:00 включительно, то есть 6-й отсчёт
    assert hourly.loc["2024-01-01 01:00", "x"] == 6.0
    # и оно строго меньше значений, лежащих позже метки
    assert hourly.index.max() <= df.index.max() + pd.Timedelta("1h")
    for ts, value in hourly["x"].items():
        assert value <= df.loc[:ts, "x"].max()


class _FakeModel:
    """Минимальная модель, совместимая с контрактом QualityAgent."""

    horizon_hours = 0.0
    alarm_threshold = 0.3
    features = ["ht_T5"]

    def __init__(self, matrix):
        self.feature_matrix = matrix

    def _row_for(self, state):
        idx = self.feature_matrix.index
        pos = idx.searchsorted(pd.Timestamp(state.ts), side="right") - 1
        return None if pos < 0 else self.feature_matrix.iloc[[pos]]

    def predict_frame(self, row):
        return pd.DataFrame({"q50": 20.0 - 0.05 * row["ht_T5"]}, index=row.index)


def test_model_surrogate_reacts_to_controls():
    """Суррогат обязан менять прогноз при изменении управляющей уставки."""
    from tests.test_agents import make_state

    idx = pd.date_range("2026-04-20 00:00", periods=24, freq="1h", name="date")
    matrix = pd.DataFrame({"ht_T5": np.full(len(idx), 370.0)}, index=idx)
    surrogate = make_model_surrogate(_FakeModel(matrix))

    state = make_state()
    base = surrogate(state, {"T5": 370.0})["product_sulfur_mgkg"]
    hotter = surrogate(state, {"T5": 375.0})["product_sulfur_mgkg"]
    assert hotter < base          # выше температура — ниже сера


def test_sigma_from_interval_is_positive_and_scaled():
    model = SulfurModel()
    model.features = ["a"]
    model.sigma_scale = 2.0

    class _Const:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    model.models = {"q50": _Const(8.0), "q10": _Const(6.0), "q90": _Const(10.0)}
    out = model.predict_frame(pd.DataFrame({"a": [1.0]}))
    expected = (10.0 - 6.0) / (2 * Z90) * 2.0
    assert out["sigma"].iloc[0] == pytest.approx(expected)


def test_default_paths_do_not_collide_between_horizons():
    assert SulfurModel.default_path(0) != SulfurModel.default_path(2)
