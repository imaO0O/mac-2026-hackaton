"""Тесты нейросетевых моделей: контракт, отсутствие утечки, работа без GPU.

Torch не входит в requirements.txt — он ставится отдельно и только тем, кому нужен
GPU. Поэтому весь файл пропускается, если torch не установлен: демо и остальные
тесты обязаны работать на машине без него.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch ставится отдельно, см. docs/GPU_SETUP.md")

from nefte.agents.schemas import DataQuality, ProcessState  # noqa: E402
from nefte.models.anomaly_ae import LSTMAnomalyDetector  # noqa: E402
from nefte.models.sequence import (  # noqa: E402
    DEFAULT_CHANNELS,
    PRETRAIN_TARGET,
    SulfurSequenceModel,
    build_windows,
    fill_windows,
)


def _features(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Синтетическая матрица признаков с теми же колонками, что у настоящей."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", name="date")
    data = {}
    for i, name in enumerate(DEFAULT_CHANNELS):
        base = 300.0 + 10 * i
        data[name] = base + np.cumsum(rng.normal(0, 0.3, n))
    data["pak_frozen"] = np.zeros(n, dtype="float32")
    return pd.DataFrame(data, index=idx).astype("float32")


def _target(features: pd.DataFrame, count: int = 120) -> pd.Series:
    """Лабораторные анализы: разрежённые метки поверх той же сетки."""
    idx = features.index[::max(len(features) // count, 1)]
    values = 8.0 + 0.02 * (features.loc[idx, "ht_T5"] - features["ht_T5"].mean())
    return pd.Series(values.to_numpy(dtype="float64"), index=idx)


# --------------------------------------------------------------------------- #
# окна
# --------------------------------------------------------------------------- #

def test_window_ends_before_the_analysis_moment():
    """Правило «только прошлое»: окно не имеет права захватить сам момент анализа."""
    feats = _features(100)
    ts = feats.index[50]
    X, ok = build_windows(feats, pd.DatetimeIndex([ts]), ["ht_T5"], window=5,
                          horizon_hours=1.0)
    assert ok[0]
    # горизонт 1 час: последний отсчёт окна — на час раньше анализа
    assert X[0, -1, 0] == pytest.approx(feats["ht_T5"].iloc[49])


def test_incomplete_window_is_marked_not_invented():
    feats = _features(100)
    X, ok = build_windows(feats, pd.DatetimeIndex([feats.index[2]]), ["ht_T5"],
                          window=24, horizon_hours=0.0)
    assert not ok[0] and np.isnan(X[0]).all()


def test_gaps_are_filled_by_last_known_value():
    X = np.array([[[1.0], [np.nan], [3.0]]], dtype="float32")
    filled = fill_windows(X, np.array([9.0], dtype="float32"))
    assert filled[0, 1, 0] == pytest.approx(1.0)      # протянули последнее известное
    assert not np.isnan(filled).any()


# --------------------------------------------------------------------------- #
# контракт с агентом качества
# --------------------------------------------------------------------------- #

def test_model_follows_quality_agent_contract():
    """Тот же интерфейс, что у SulfurModel: агенту всё равно, что внутри."""
    feats = _features()
    y = _target(feats)
    model = SulfurSequenceModel(window=8, hidden=8, epochs=3, seeds=(42,))
    model.fit(feats, y.iloc[:80], y.iloc[80:], prefer_gpu=False).attach(feats)

    state = ProcessState(ts=feats.index[-1].to_pydatetime(),
                         data_quality=DataQuality(missing_share=0.0))
    mean, sigma = model.predict_with_sigma(state)
    assert np.isfinite(mean) and sigma > 0
    risk = model.risk_for_state(state)
    assert risk is not None and 0.0 <= risk <= 1.0


def test_quantiles_do_not_cross():
    feats = _features()
    y = _target(feats)
    model = SulfurSequenceModel(window=8, hidden=8, epochs=3, seeds=(42, 43))
    model.fit(feats, y.iloc[:80], y.iloc[80:], prefer_gpu=False).attach(feats)
    pred = model.predict_index(y.index[80:])
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()


def test_no_prediction_before_first_full_window():
    feats = _features()
    y = _target(feats)
    model = SulfurSequenceModel(window=24, hidden=8, epochs=2, seeds=(42,))
    model.fit(feats, y.iloc[:80], y.iloc[80:], prefer_gpu=False).attach(feats)

    early = ProcessState(ts=feats.index[3].to_pydatetime(),
                         data_quality=DataQuality(missing_share=0.0))
    mean, sigma = model.predict_with_sigma(early)
    assert mean != mean and sigma == float("inf")     # NaN и бесконечная σ


def test_pretraining_removes_the_target_channel_from_input():
    """Предобучаемся предсказывать ПАК — значит, самого ПАК во входе быть не должно."""
    feats = _features()
    y = _target(feats)
    model = SulfurSequenceModel(window=8, hidden=8, epochs=2, seeds=(42,))
    model.fit(feats, y.iloc[:80], y.iloc[80:], prefer_gpu=False,
              pretrain_bounds=("2024-01-01", "2024-01-20"), pretrain_epochs=2)
    assert PRETRAIN_TARGET not in model.channels
    assert model.pretrained


def test_save_and_load_round_trip(tmp_path):
    feats = _features()
    y = _target(feats)
    model = SulfurSequenceModel(window=8, hidden=8, epochs=3, seeds=(42,))
    model.fit(feats, y.iloc[:80], y.iloc[80:], prefer_gpu=False).attach(feats)
    before = model.predict_index(y.index[80:])["q50"].to_numpy()

    path = model.save(tmp_path / "seq")
    restored = SulfurSequenceModel.load(path).attach(feats)
    after = restored.predict_index(y.index[80:])["q50"].to_numpy()
    assert np.allclose(before, after, atol=1e-5)


# --------------------------------------------------------------------------- #
# автоэнкодер аномалий
# --------------------------------------------------------------------------- #

def test_autoencoder_matches_detector_interface():
    frame = _features(800)[["ht_T5", "ht_T6", "ht_P13"]]
    ae = LSTMAnomalyDetector.fit(frame, list(frame.columns),
                                 train=("2024-01-01", "2024-01-25"),
                                 window=8, hidden=8, epochs=3, prefer_gpu=False)
    assert ae.fitted
    scores = ae.scores(frame)
    assert scores.index.equals(frame.index)
    assert ae.normalized(frame).max() > 0
    shares = ae.contributions_frame(frame).dropna()
    assert not shares.empty
    assert shares.sum(axis=1).round(3).eq(1.0).all()


def test_autoencoder_threshold_comes_from_train_only():
    """Порог по обучающему периоду: иначе аномалия меряется относительно будущего."""
    frame = _features(800)[["ht_T5", "ht_T6", "ht_P13"]]
    train = ("2024-01-01", "2024-01-25")
    ae = LSTMAnomalyDetector.fit(frame, list(frame.columns), train=train,
                                 window=8, hidden=8, epochs=3, prefer_gpu=False)
    train_scores = ae.scores(frame).loc[train[0]:train[1]].dropna()
    assert ae.threshold == pytest.approx(float(np.nanquantile(train_scores, 0.99)),
                                         rel=1e-6)
