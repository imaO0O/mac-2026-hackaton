"""Перепад давления Р-202 в тяжести режима — за выключателем.

По выданным данным `P8` меряет не износ слоя, а гидравлику и уровень после
загрузки катализатора (`docs/DP_PROXY.md`, участник 2): внутри цикла при той же
нагрузке он растёт на 0.005 МПа за два года и после каждой загрузки прыгает вверх.
Включённый уровнем, он переводил 18 % моментов теста в тяжёлый класс, а тяжёлый
класс запрещает поднимать температуру реакторов — отказов «нет допустимых
вариантов» на тесте становилось 81 против 11.

Поэтому фактор по умолчанию выключен (`reliability.dp_factor: "off"`), и это
измеренный отказ, а не умолчание «так получилось». Тесты держат три вещи: значение
из конфига доходит до агента, выключенный фактор не влияет на тяжесть вовсе, а
включённый — влияет.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nefte.agents.reliability import DP_FACTORS, ReliabilityAgent, SeverityNorms
from nefte.config import load_config
from tests.test_agents import make_state

NORMS = SeverityNorms(bounds={"wabt": (355.0, 375.0), "P8": (0.13, 0.23),
                              "T55": (370.0, 395.0)})


def _severity(dp_factor: str, dp_value: float) -> float:
    agent = ReliabilityAgent(NORMS, dp_factor=dp_factor)
    state = make_state()
    state.telemetry_ht.update({"T5": 366.0, "T6": 365.0, "T11": 365.0, "P8": dp_value})
    state.telemetry_avt.update({"T55": 380.0})
    return agent.assess(state).severity_index


def test_config_default_is_off_and_says_why():
    """Умолчание — часть постановки: выключено по измеренному отказу."""
    section = load_config()["reliability"]
    assert section["dp_factor"] == "off"
    assert section["dp_factor"] in DP_FACTORS


def _history(n: int = 400):
    idx = pd.date_range("2024-01-01", periods=n, freq="10min", name="date")
    ht = pd.DataFrame({"T5": 370.0, "T6": 365.0, "T11": 366.0, "F26": 250.0,
                       "F2": 90000.0, "P13": 3.9, "P8": 0.19}, index=idx)
    avt = pd.DataFrame({"T55": 380.0}, index=idx)
    return avt, ht


@pytest.mark.parametrize("mode", ["off", "level", "growth"])
def test_switch_reaches_the_agent_from_config(mode):
    cfg = load_config()
    local = {**cfg, "reliability": {**cfg["reliability"], "dp_factor": mode},
             "split": {**cfg["split"], "train": ["2024-01-01", "2024-01-03"]}}
    avt, ht = _history()
    assert ReliabilityAgent.from_history(avt, ht, local).dp_factor == mode


def test_unknown_switch_value_is_an_error_not_a_silent_default():
    cfg = load_config()
    local = {**cfg, "reliability": {**cfg["reliability"], "dp_factor": "нет такого"},
             "split": {**cfg["split"], "train": ["2024-01-01", "2024-01-03"]}}
    avt, ht = _history()
    with pytest.raises(ValueError, match="dp_factor"):
        ReliabilityAgent.from_history(avt, ht, local)


def test_disabled_factor_does_not_move_severity():
    """Выключенный — значит не входит: перепад у верха шкалы и у низа дают одно."""
    assert _severity("off", 0.14) == pytest.approx(_severity("off", 0.30))


def test_enabled_factor_moves_severity():
    """Обратная сторона: при level фактор действительно работает."""
    low, high = _severity("level", 0.14), _severity("level", 0.30)
    assert high > low + 0.05


def test_weights_are_renormalized_not_diluted():
    """Выключение перераспределяет вес, а не занижает тяжесть.

    Если бы свёртка делила на постоянную сумму весов, выключение фактора занижало
    бы тяжесть на его вклад — и «жёсткий» режим молча стал бы «умеренным».
    """
    same = _severity("level", 0.18)   # перепад ровно посередине шкалы
    assert _severity("off", 0.18) == pytest.approx(same, abs=0.02)


# --------------------------------------------------------------------------- #
# вариант «прирост» (dp_factor: growth) — участник 2
# --------------------------------------------------------------------------- #

import numpy as np  # noqa: E402

from nefte.models.catalyst import dp_growth_series  # noqa: E402

GROWTH_TRAIN = ("2024-01-01", "2024-04-30")
LOADING = "2024-05-30"          # 150-е сутки синтетической истории


def _dp_history(days: int = 330, rate: float = 0.0002, step: float = 0.03,
                seed: int = 0) -> pd.DataFrame:
    """P8 = гидравлика + рост внутри цикла; на 150-е сутки загрузка со скачком вверх.

    Так ведёт себя P8 на выданных данных (docs/DP_PROXY.md): после загрузки свежего
    катализатора уровень прыгает вверх, а закоксовывание — это рост ОТ этого уровня.
    """
    idx = pd.date_range("2024-01-01", periods=days * 144, freq="10min", name="date")
    rng = np.random.default_rng(seed)
    day = np.arange(len(idx)) / 144
    feed = 250.0 + 20.0 * np.sin(np.arange(len(idx)) / 500.0) + rng.normal(0, 3.0, len(idx))
    cycle_day = np.where(day >= 150, day - 150, day)
    p8 = (0.09 + 1.2e-6 * feed ** 2 + rate * cycle_day + np.where(day >= 150, step, 0.0)
          + rng.normal(0, 0.002, len(idx)))
    return pd.DataFrame({"P8": p8, "F26": feed}, index=idx)


def _growth(ht: pd.DataFrame) -> pd.Series:
    series = dp_growth_series(ht, ht["F26"], GROWTH_TRAIN, [LOADING])
    assert series is not None
    return series


def test_growth_ignores_the_jump_after_a_fresh_loading():
    """Скачок уровня после загрузки — не износ: опора цикла пересчитывается."""
    series = _growth(_dp_history())
    after_loading = series.loc["2024-06-04":"2024-06-24"].median()
    assert after_loading < 0.3, "скачок загрузки прошёл бы в фактор как тяжесть"


def test_growth_follows_coking_within_a_cycle():
    series = _growth(_dp_history())
    assert series.loc["2024-11-01":"2024-11-20"].median() > (
        series.loc["2024-07-10":"2024-07-20"].median() + 0.3)


def test_growth_uses_only_the_past():
    """Значение на момент после обучающего периода не зависит от будущего."""
    ht = _dp_history()
    cut = pd.Timestamp("2024-09-15 12:00")
    full = _growth(ht)
    past = dp_growth_series(ht.loc[:cut], ht["F26"].loc[:cut], GROWTH_TRAIN, [LOADING])
    assert past is not None
    assert full.loc[cut] == pytest.approx(past.loc[cut])


def test_growth_mode_puts_the_series_into_severity():
    cfg = load_config()
    ht = _dp_history()
    ht = ht.assign(T5=370.0, T6=365.0, T11=366.0, F2=90000.0, P13=3.9)
    avt = pd.DataFrame({"T55": 380.0}, index=ht.index)
    local = {**cfg, "reliability": {**cfg["reliability"], "dp_factor": "growth",
                                    "catalyst_changes": [LOADING]},
             "split": {**cfg["split"], "train": list(GROWTH_TRAIN)}}
    agent = ReliabilityAgent.from_history(avt, ht, local)
    assert agent.dp_series is not None
    frame = pd.DataFrame({"wabt": ht[["T5", "T6", "T11"]].mean(axis=1), "P8": ht["P8"],
                          "T55": avt["T55"]})
    _, factors = agent.severity_series(frame, with_factors=True)
    assert "dp_r202" in factors.columns
    assert factors["dp_r202"].notna().sum() > 1000
