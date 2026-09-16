"""Тесты детекторов достоверности и нормировки тяжести режима (участник 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.agents.reliability import ReliabilityAgent, SeverityNorms
from nefte.data.validity import SignalValidity


def _telemetry(n: int = 200, start: str = "2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="10min", name="date")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "T6": 350 + rng.normal(0, 1, n),
        "F30": 100 + rng.normal(0, 5, n),
        "D10": np.full(n, 307.0),          # мёртвый датчик
    }, index=idx)


def test_sentinel_and_dead_tag_detected():
    df = _telemetry()
    df.loc[df.index[10], "T6"] = 307.0
    v = SignalValidity.build(df, unit="avt")

    assert v.sentinel.loc[df.index[10], "T6"]
    assert np.isnan(v.clean.loc[df.index[10], "T6"])
    assert "D10" in v.dead_tags


def test_frozen_plateau_is_masked_not_kept():
    """Полка маскируется — но только с того отсчёта, где порог набран по прошлому.

    Раньше тест требовал маску на ВСЕЙ полке с первого отсчёта. Это выглядело
    естественно и было заглядыванием вперёд: система узнавала об отказе прибора
    раньше, чем это в принципе возможно. Цена исправления — первые 17 отсчётов
    полки остаются в данных как достоверные; это осознанный размен, потому что
    единственная альтернатива — снова развести очистку для обучения и для работы,
    а они у нас уже однажды разъехались.
    """
    df = _telemetry()
    df.iloc[50:80, df.columns.get_loc("F30")] = 42.0     # 30 отсчётов = 5 часов
    v = SignalValidity.build(df, unit="avt")

    threshold = 18                                       # frozen_min_samples
    assert v.frozen.iloc[50 + threshold - 1:80]["F30"].all()
    assert v.clean.iloc[50 + threshold - 1:80]["F30"].isna().all()
    assert not v.frozen.iloc[:50]["F30"].any()
    assert not v.frozen.iloc[50:50 + threshold - 1]["F30"].any()


def test_flags_at_explains_reasons():
    df = _telemetry()
    df.loc[df.index[10], "T6"] = 307.0
    v = SignalValidity.build(df, unit="avt")

    flags = v.flags_at(df.index[10])
    assert "заглушка" in flags["T6"]
    assert "датчик не даёт сигнала" in flags["D10"]


def test_summary_counts_every_reason():
    df = _telemetry()
    df.loc[df.index[5], "T6"] = 307.0
    summary = SignalValidity.build(df, unit="avt").summary()

    assert set(summary.columns) >= {"заглушка_%", "полка_%", "отриц_%", "итого_%"}
    assert summary.loc["T6", "заглушка_%"] > 0
    assert summary.loc["D10", "мёртвый"]


def test_severity_norms_use_train_period_only():
    """Нормировка по всей истории — это заглядывание в будущее."""
    idx = pd.date_range("2023-01-01", "2026-08-01", freq="D", name="date")
    values = pd.Series(np.linspace(0, 100, len(idx)), index=idx)   # монотонный рост
    df = pd.DataFrame({"wabt": values})

    full = SeverityNorms.fit(df, ["wabt"])
    train = SeverityNorms.fit(df, ["wabt"], train=("2023-01-01", "2025-06-30"))

    assert train.bounds["wabt"][1] < full.bounds["wabt"][1]
    # значение из 2026 года выходит за обучающий диапазон — так и должно быть
    assert train.normalize("wabt", 100.0) > 1.0


def test_severity_grows_with_regime_stress():
    from tests.test_agents import make_state

    norms = SeverityNorms(bounds={"wabt": (350.0, 370.0), "P8": (0.13, 0.23),
                                  "T55": (370.0, 395.0)})
    agent = ReliabilityAgent(norms)

    mild = make_state()
    mild.telemetry_ht.update({"T5": 352.0, "T6": 351.0, "T11": 350.0, "P8": 0.14})
    harsh = make_state()
    harsh.telemetry_ht.update({"T5": 369.0, "T6": 369.0, "T11": 370.0, "P8": 0.22})

    assert agent.assess(mild).severity_index < agent.assess(harsh).severity_index
    assert agent.assess(harsh).risk_class in {"medium", "high"}


def test_high_severity_forbids_raising_temperature():
    norms = SeverityNorms(bounds={"wabt": (350.0, 360.0), "P8": (0.13, 0.18),
                                  "T55": (370.0, 380.0)})
    agent = ReliabilityAgent(norms)

    from tests.test_agents import make_state
    state = make_state()
    state.telemetry_ht.update({"T5": 380.0, "T6": 380.0, "T11": 380.0, "P8": 0.25})
    state.telemetry_avt.update({"T55": 400.0})

    out = agent.assess(state)
    assert out.risk_class == "high" and not out.admissible
    for _, (lo, hi) in out.constraints.items():
        assert hi <= 380.0          # только вниз
    assert out.notes


def test_ramp_factor_reacts_to_fast_change():
    idx = pd.date_range("2024-01-01", periods=50, freq="10min", name="date")
    calm = pd.Series(np.zeros(len(idx)), index=idx)
    agent_calm = ReliabilityAgent(SeverityNorms(bounds={"wabt": (0.0, 1.0)}), ramp_series=calm)
    agent_fast = ReliabilityAgent(SeverityNorms(bounds={"wabt": (0.0, 1.0)}),
                                  ramp_series=pd.Series(np.ones(len(idx)), index=idx))

    from tests.test_agents import make_state
    state = make_state()
    state.ts = idx[-1].to_pydatetime()
    state.telemetry_ht.update({"T5": 0.5, "T6": 0.5, "T11": 0.5})

    assert agent_calm.assess(state).severity_index < agent_fast.assess(state).severity_index


def test_agent_without_data_returns_medium_and_says_so():
    from tests.test_agents import make_state
    state = make_state()
    state.telemetry_ht = {}
    state.telemetry_avt = {}

    out = ReliabilityAgent(SeverityNorms(bounds={})).assess(state)
    assert out.severity_index == pytest.approx(0.5)
    assert out.notes and "Нет данных" in out.notes[0]
