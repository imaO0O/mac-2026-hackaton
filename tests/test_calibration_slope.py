# -*- coding: utf-8 -*-
"""Наклон калибровки обязан различать три формы ошибки вероятности.

Сжатие к середине даёт ошибки разного знака в разных бинах, и ECE его гасит:
на тесте верхний бин заявлял 36 % при 50 %, соседний 16 % при 6 %, а ECE 0.055
выглядела как хорошая калибровка. Проверяем на синтетике, где форма известна
заранее, — иначе тест проверял бы модель, а не меру.
"""
import numpy as np
import pandas as pd

from nefte.models.quality_model import calibration_slope


def _sample(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(-1.5, 1.2, n)
    p_true = 1 / (1 + np.exp(-z))
    over = pd.Series((rng.random(n) < p_true).astype(int))
    return z, over


def _claimed(z, k):
    return pd.Series(1 / (1 + np.exp(-z * k)))


def test_honest_probabilities_give_slope_one():
    z, over = _sample()
    r = calibration_slope(_claimed(z, 1.0), over, n_boot=300)
    assert r["форма"] == "согласуется с верной", r
    assert 0.85 < r["b"] < 1.15


def test_compressed_probabilities_are_detected():
    """Заявленные логиты вдвое сжаты — истинный наклон 2."""
    z, over = _sample()
    r = calibration_slope(_claimed(z, 0.5), over, n_boot=300)
    assert r["форма"] == "сжата к середине", r
    assert 1.7 < r["b"] < 2.3


def test_overconfident_probabilities_are_detected():
    z, over = _sample()
    r = calibration_slope(_claimed(z, 2.0), over, n_boot=300)
    assert r["форма"] == "излишне растянута", r
    assert 0.4 < r["b"] < 0.6


def test_compression_is_invisible_to_the_level_check():
    """Ради чего мера нужна: у сжатия средняя заявленная почти не врёт."""
    z, over = _sample()
    # сжимаем вдвое и подбираем сдвиг бисекцией так, чтобы уровень совпал
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float((1 / (1 + np.exp(-(z * 0.5 + mid)))).mean()) > over.mean():
            hi = mid
        else:
            lo = mid
    claimed = pd.Series(1 / (1 + np.exp(-(z * 0.5 + (lo + hi) / 2))))
    shift = abs(float(claimed.mean()) - float(over.mean()))
    r = calibration_slope(claimed, over, n_boot=300)
    assert shift < 0.03, "синтетика не та: уровень должен почти совпадать"
    assert r["форма"] == "сжата к середине"


def test_degenerate_input_returns_empty():
    assert calibration_slope(pd.Series([0.1] * 50), pd.Series([0] * 50)) == {}
    assert calibration_slope(pd.Series([0.1, 0.9]), pd.Series([0, 1])) == {}
