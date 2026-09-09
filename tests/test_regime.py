"""Тесты признаков режима реактора и кинетического суррогата."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.models.kinetics import arrhenius_factor, make_kinetic_surrogate
from nefte.models.regime import (
    apply_moves_to_rows,
    history_features,
    instant_features,
    regime_features,
)
from tests.test_agents import make_state


def _ht(n: int = 400, feed: float = 250.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="10min", name="date")
    return pd.DataFrame({
        "T5": np.full(n, 370.0), "T6": np.full(n, 365.0), "T11": np.full(n, 360.0),
        "F26": np.full(n, feed), "F2": np.full(n, 90000.0),
        "P24": np.full(n, 0.6), "F15": np.full(n, 3400.0), "P13": np.full(n, 3.9),
    }, index=idx)


def test_wabt_and_exotherm():
    out = instant_features(_ht())
    assert out["reg_wabt"].iloc[0] == pytest.approx(365.0)
    assert out["reg_dt_react"].iloc[0] == pytest.approx(-10.0)   # T11 - T5


def test_kinetic_index_grows_with_temperature_and_falls_with_feed():
    hot = _ht().assign(T5=372.0, T6=367.0, T11=362.0)
    assert (instant_features(hot)["reg_kinetic"].iloc[0]
            > instant_features(_ht())["reg_kinetic"].iloc[0])

    loaded = _ht(feed=300.0)
    assert (instant_features(loaded)["reg_kinetic"].iloc[0]
            < instant_features(_ht())["reg_kinetic"].iloc[0])


def test_ratios_are_not_computed_on_shutdown():
    """Делить на почти нулевой расход — значит породить выброс, а не признак."""
    ht = _ht()
    ht.iloc[100:200, ht.columns.get_loc("F26")] = 0.5
    out = instant_features(ht)
    assert out["reg_h2_oil"].iloc[150] != out["reg_h2_oil"].iloc[150]   # NaN
    assert out["reg_h2_oil"].iloc[0] == pytest.approx(90000.0 / 250.0)


def test_run_hours_reset_after_shutdown():
    ht = _ht(n=1000)
    ht.iloc[300:400, ht.columns.get_loc("F26")] = 1.0     # ~17 часов простоя
    out = history_features(ht)
    assert out["reg_run_hours"].iloc[350] < out["reg_run_hours"].iloc[299]
    assert out["reg_run_hours"].iloc[500] < out["reg_run_hours"].iloc[299]


def test_regime_features_are_finite_where_defined():
    out = regime_features(_ht())
    assert out["reg_wabt"].notna().all()
    assert np.isfinite(out["reg_kinetic"].dropna()).all()


def test_apply_moves_recomputes_derived_features():
    """Подмена одной колонки без пересчёта режима оставила бы старый WABT."""
    ht = _ht(n=50)
    matrix = pd.concat([ht.add_prefix("ht_"), regime_features(ht)], axis=1)
    matrix["ht_T5_mean6"] = matrix["ht_T5"]

    moved = apply_moves_to_rows(matrix, {"T5": 2.0}, relative=True)
    assert moved["ht_T5"].iloc[0] == pytest.approx(372.0)
    assert moved["reg_wabt"].iloc[0] == pytest.approx(365.0 + 2.0 / 3)
    assert moved["reg_kinetic"].iloc[0] > matrix["reg_kinetic"].iloc[0]
    # часовое среднее сдвигается вместе с уставкой: она удерживается
    assert moved["ht_T5_mean6"].iloc[0] == pytest.approx(372.0)


def test_apply_moves_leaves_history_features_alone():
    ht = _ht(n=50)
    matrix = pd.concat([ht.add_prefix("ht_"), regime_features(ht)], axis=1)
    moved = apply_moves_to_rows(matrix, {"T5": 5.0}, relative=True)
    assert moved["reg_run_hours"].equals(matrix["reg_run_hours"])


# --------------------------------------------------------------------------- #
# кинетический суррогат
# --------------------------------------------------------------------------- #

class _FlatModel:
    """Модель, которая всегда возвращает один уровень: удобно мерить приращение."""

    features: list[str] = []

    def __init__(self, level: float = 9.0):
        self.level = level

    def predict_with_sigma(self, state):
        return self.level, 1.0


def _state_with_feed(feed_sulfur: float = 9000.0):
    from nefte.agents.schemas import Measurement, Source

    state = make_state()
    state.telemetry_ht.update({"T5": 370.0, "T6": 365.0, "T11": 360.0,
                               "F26": 250.0, "P13": 3.9, "P24": 0.6})
    state.quality["lims_feed_sulfur_mgkg"] = Measurement(
        value=feed_sulfur, unit="мг/кг", source=Source.LIMS, age_hours=10.0)
    return state


def test_arrhenius_factor_direction():
    assert arrhenius_factor(360, 362) > 1.0
    assert arrhenius_factor(360, 358) < 1.0
    assert arrhenius_factor(360, 360) == pytest.approx(1.0)


def test_kinetic_surrogate_reacts_to_temperature():
    fn = make_kinetic_surrogate(_FlatModel(9.0))
    state = _state_with_feed()
    base = fn(state, {})["product_sulfur_mgkg"]
    hotter = fn(state, {"T5": 372.0, "T6": 367.0, "T11": 362.0})["product_sulfur_mgkg"]
    colder = fn(state, {"T5": 368.0, "T6": 363.0, "T11": 358.0})["product_sulfur_mgkg"]

    assert base == pytest.approx(9.0, rel=1e-6)
    assert hotter < base < colder                     # выше температура — ниже сера
    assert base - hotter > 0.5                        # отклик заметный, не сотые доли


def test_kinetic_surrogate_reacts_to_feed_rate():
    fn = make_kinetic_surrogate(_FlatModel(9.0))
    state = _state_with_feed()
    base = fn(state, {})["product_sulfur_mgkg"]
    faster = fn(state, {"F26": 300.0})["product_sulfur_mgkg"]
    assert faster > base            # больше нагрузка — меньше время контакта


def test_strength_zero_disables_kinetic_correction():
    """Ручка strength нужна, чтобы показать зависимость решения от допущения."""
    fn = make_kinetic_surrogate(_FlatModel(9.0), strength=0.0)
    state = _state_with_feed()
    hotter = fn(state, {"T5": 372.0, "T6": 367.0, "T11": 362.0})["product_sulfur_mgkg"]
    assert hotter == pytest.approx(9.0, rel=1e-6)


def test_falls_back_when_feed_sulfur_is_implausible():
    """Без серы сырья кинетику применять нельзя — уходим в запасной суррогат."""
    called = {}

    def base_surrogate(state, moves):
        called["yes"] = True
        return {"product_sulfur_mgkg": 7.77}

    fn = make_kinetic_surrogate(_FlatModel(9.0), base_surrogate=base_surrogate)
    state = _state_with_feed(feed_sulfur=9.5)          # сырьё чище продукта — бессмыслица
    out = fn(state, {"T5": 372.0})
    assert called.get("yes") and out["product_sulfur_mgkg"] == pytest.approx(7.77)
