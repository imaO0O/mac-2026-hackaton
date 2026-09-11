"""Бюджет тревог: обещание «не чаще, чем на 30 % моментов».

Порог по бюджету — это квантиль распределения риска, и он осмыслен ровно
настолько, насколько это распределение стабильно. У бустинга оно стабильно, у
нейросети средний риск между валидацией и тестом удваивается, и тот же приём даёт
68.7 % тревог вместо 30 %. Тесты закрепляют и сам приём, и защиту от утечки в нём.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nefte.models.quality_model import (
    pick_threshold_for_budget,
    rolling_budget_threshold,
)


def _risk(values, start="2026-01-01", freq="6h"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq))


def test_fixed_threshold_hits_the_budget_on_its_own_sample():
    """На той выборке, где подобран, порог обязан давать ровно бюджет."""
    rng = np.random.default_rng(0)
    risk = _risk(rng.uniform(0.0, 1.0, 1000))
    thr = pick_threshold_for_budget(risk, 0.3)
    assert (risk > thr).mean() == pytest.approx(0.3, abs=0.02)


def test_fixed_threshold_breaks_when_the_risk_distribution_shifts():
    """Главный случай, ради которого заведён скользящий порог.

    Риск во второй половине вдвое выше. Порог, подобранный по первой, во второй
    означает совсем другую частоту вмешательств — и именно это случилось с
    нейросетью на горизонте 2 часа.
    """
    rng = np.random.default_rng(1)
    calm = rng.uniform(0.0, 0.3, 500)
    shifted = rng.uniform(0.0, 0.6, 500)
    thr = pick_threshold_for_budget(pd.Series(calm), 0.3)
    assert (shifted > thr).mean() > 0.55


def test_rolling_threshold_holds_the_budget_through_the_shift():
    """Скользящий порог подстраивается и удерживает обещанную частоту."""
    rng = np.random.default_rng(2)
    risk = _risk(np.concatenate([rng.uniform(0.0, 0.3, 400),
                                 rng.uniform(0.0, 0.6, 400)]))
    thr = rolling_budget_threshold(risk, 0.3, window_days=30)
    valid = thr.notna()
    # берём вторую половину — ту, где распределение уже уехало
    late = valid & (risk.index >= risk.index[len(risk) // 2])
    assert (risk[late] > thr[late]).mean() == pytest.approx(0.3, abs=0.08)


def test_rolling_threshold_does_not_use_the_current_value():
    """Порог для момента t считается по риску СТРОГО до t.

    Без сдвига текущее значение участвует в собственном пороге — утечка того же
    рода, что ловилась в признаках. Проверяем прямо: одиночный выброс в конце
    не должен поднять порог, действующий на самом выбросе.
    """
    risk = _risk([0.1] * 60 + [0.99])
    thr = rolling_budget_threshold(risk, 0.3, window_days=30, min_periods=5)
    assert thr.iloc[-1] == pytest.approx(0.1, abs=1e-9)


def test_rolling_threshold_is_undefined_until_there_is_history():
    """Пока истории мало, порога нет — и это должно быть видно как NaN.

    Подставить сюда значение по умолчанию значило бы принимать решения по
    несуществующему бюджету в первые дни работы.
    """
    risk = _risk([0.2] * 10)
    thr = rolling_budget_threshold(risk, 0.3, window_days=30, min_periods=20)
    assert thr.isna().all()


# --------------------------------------------------------------------------- #
# нормированная температура: отрицательный результат, который надо охранять
# --------------------------------------------------------------------------- #

def test_normalized_wabt_rises_when_the_catalyst_gives_less():
    """Смысл величины: при том же режиме рост серы означает рост NWABT."""
    from nefte.models.regime import normalized_wabt

    wabt = pd.Series([360.0, 360.0, 360.0])
    sulfur = pd.Series([5.0, 10.0, 20.0])
    values = normalized_wabt(wabt, sulfur, reference_mgkg=10.0)
    assert values.iloc[0] < values.iloc[1] < values.iloc[2]
    # на опорной сере поправка равна нулю
    assert values.iloc[1] == pytest.approx(360.0)


def test_normalized_wabt_matches_the_kinetics_it_claims():
    """Поправка обязана совпадать с той кинетикой, на которую ссылается.

    Если коэффициент разъедется с ACTIVATION_ENERGY_KJ в kinetics.py, признак
    начнёт означать не то, что написано в его докстринге.
    """
    from nefte.models.regime import KINETIC_SENSITIVITY, normalized_wabt

    values = normalized_wabt(pd.Series([340.0]), pd.Series([20.0]),
                             reference_mgkg=10.0)
    assert values.iloc[0] - 340.0 == pytest.approx(
        np.log(2.0) / KINETIC_SENSITIVITY)


def test_normalized_wabt_is_not_in_the_feature_matrix():
    """Признак проверен и НЕ принят — тест держит это решение.

    Он отбирался моделью и даже вытеснял reg_wabt_dev30, но качества не
    прибавил: алгебраическую комбинацию двух уже имеющихся признаков бустинг
    находит сам. Если кто-то вернёт его в матрицу, это должно быть осознанным
    решением с новыми числами, а не случайностью.
    """
    from nefte.models.dataset import FEATURE_VERSION

    import nefte.models.dataset as dataset_module

    source = Path(dataset_module.__file__).read_text(encoding="utf-8")
    assert "reg_nwabt" not in source
    assert FEATURE_VERSION == 4


def test_zero_or_negative_sulfur_gives_nothing_not_minus_infinity():
    """В ЛИМС встречаются нули. Логарифм нуля — минус бесконечность, а не признак."""
    from nefte.models.regime import normalized_wabt

    values = normalized_wabt(pd.Series([360.0, 360.0]), pd.Series([0.0, -1.0]))
    assert values.isna().all()
