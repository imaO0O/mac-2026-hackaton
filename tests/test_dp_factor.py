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


@pytest.mark.parametrize("mode", ["off", "level"])
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
