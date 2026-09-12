"""Тесты новых механизмов агента надёжности.

Проверяем то, что стоило крови при отладке: останов виден только по сырому
сигналу, короткий простой не омолаживает катализатор, а аномалия — это про
сочетание параметров, а не про каждый по отдельности.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.agents.reliability import ReliabilityAgent, SeverityNorms
from nefte.models.anomaly import RegimeAnomalyDetector
from nefte.models.regime import hours_since_outage, outage_mask
from tests.test_agents import make_state


def _feed(n: int = 2000, outages: list[tuple[int, int]] | None = None) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="10min", name="date")
    values = np.full(n, 250.0)
    for start, length in (outages or []):
        values[start:start + length] = 0.5
    return pd.Series(values, index=idx)


# --------------------------------------------------------------------------- #
# остановы и наработка
# --------------------------------------------------------------------------- #

def test_short_dip_is_not_an_outage():
    """Провал на час — это не останов, счётчик наработки сбрасывать нельзя."""
    feed = _feed(outages=[(500, 6)])          # 1 час
    assert not outage_mask(feed, min_outage_hours=6).any()


def test_long_dip_is_an_outage():
    feed = _feed(outages=[(500, 60)])         # 10 часов
    # Маска причинная: останов объявляется, когда длительность набрана по прошлому.
    # Раньше тест требовал `mask.iloc[500:560].all()`, то есть признания останова с
    # первого же низкого отсчёта — а это заглядывание вперёд.
    threshold = 6 * 6                          # 6 часов по 10-минутным отсчётам
    mask = outage_mask(feed, min_outage_hours=6)
    assert not mask.iloc[500:500 + threshold - 1].any()
    assert mask.iloc[500 + threshold - 1:560].all()
    # Ретроспективный режим по-прежнему видит эпизод целиком — он для отсева истории
    assert outage_mask(feed, min_outage_hours=6, retrospective=True).iloc[500:560].all()


def test_catalyst_age_ignores_short_outages():
    """Шестичасовой простой катализатор не омолаживает: сброс только на длинном."""
    feed = _feed(n=4000, outages=[(500, 60), (2000, 300)])   # 10 ч и 50 ч
    short_rule = hours_since_outage(feed, min_outage_hours=6)
    long_rule = hours_since_outage(feed, min_outage_hours=48)

    assert (short_rule.diff() < 0).sum() == 2      # оба простоя считаются
    assert (long_rule.diff() < 0).sum() == 1       # только длинный


def test_hours_since_outage_grows_after_restart():
    feed = _feed(n=2000, outages=[(500, 60)])
    run = hours_since_outage(feed, min_outage_hours=6)
    # Внутри останова счётчик обнуляется НЕ сразу, а когда длительность набрана:
    # в первые шесть часов система ещё не знает, что это останов, а не провал
    # расхода. Раньше тест проверял индекс 520 — то есть требовал знания заранее.
    assert run.iloc[520] > 1.0                              # ещё не распознан
    assert run.iloc[555] == pytest.approx(0.0, abs=0.2)     # уже распознан
    assert run.iloc[560] < run.iloc[900]                    # после пуска растёт


# --------------------------------------------------------------------------- #
# распознавание остановленной установки
# --------------------------------------------------------------------------- #

def test_unit_down_needs_both_signs():
    """Нулевой расход бывает при отказе датчика, низкая температура — при пуске."""
    agent = ReliabilityAgent(feed_median=250.0)

    cold_and_empty = make_state()
    cold_and_empty.telemetry_ht.update({"F26": 1.0, "T5": 20.0, "T6": 18.0, "T11": 15.0})
    assert agent.is_unit_down(cold_and_empty)

    only_cold = make_state()
    only_cold.telemetry_ht.update({"F26": 250.0, "T5": 20.0, "T6": 18.0, "T11": 15.0})
    assert not agent.is_unit_down(only_cold)

    only_empty = make_state()
    only_empty.telemetry_ht.update({"F26": 1.0, "T5": 370.0, "T6": 365.0, "T11": 360.0})
    assert not agent.is_unit_down(only_empty)


def test_down_mask_from_history_wins_over_cleaned_state():
    """На останове очищенный срез пуст — решает маска, посчитанная по сырым данным."""
    idx = pd.date_range("2026-04-15", periods=100, freq="10min", name="date")
    down = pd.Series(True, index=idx)
    agent = ReliabilityAgent(feed_median=250.0, down_series=down)

    state = make_state()
    state.ts = idx[50].to_pydatetime()
    state.telemetry_ht.update({"F26": None, "T5": None})   # очистка всё убрала
    assert agent.is_unit_down(state)


def test_stopped_unit_gets_no_control_recommendation():
    idx = pd.date_range("2026-04-15", periods=100, freq="10min", name="date")
    agent = ReliabilityAgent(feed_median=250.0, down_series=pd.Series(True, index=idx))
    state = make_state()
    state.ts = idx[50].to_pydatetime()

    result = agent.assess(state)
    assert not result.admissible
    assert any("остановлена" in note for note in result.notes)
    assert result.severity_index == 0.0      # холодный реактор не «мягкий режим»


# --------------------------------------------------------------------------- #
# многомерный детектор
# --------------------------------------------------------------------------- #

def _regime_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="10min", name="date")
    wabt = rng.normal(365.0, 2.0, n)
    # кратность газ/сырьё связана с температурой: вместе они и задают режим
    h2_oil = 350.0 + (wabt - 365.0) * 12.0 + rng.normal(0, 3.0, n)
    return pd.DataFrame({"wabt": wabt, "h2_oil": h2_oil}, index=idx)


def test_detector_catches_impossible_combination():
    """Каждое значение в норме по отдельности, но вместе они невозможны."""
    frame = _regime_frame()
    detector = RegimeAnomalyDetector.fit(frame, ["wabt", "h2_oil"])
    assert detector.fitted

    normal = {"wabt": 369.0, "h2_oil": 350.0 + 4 * 12.0}
    broken = {"wabt": 369.0, "h2_oil": 350.0 - 4 * 12.0}   # оба в диапазоне истории

    assert detector.score_row(normal) < 1.0
    assert detector.score_row(broken) > detector.score_row(normal) * 3


def test_detector_threshold_matches_declared_quantile():
    frame = _regime_frame()
    detector = RegimeAnomalyDetector.fit(frame, ["wabt", "h2_oil"], quantile=0.99)
    share = float(detector.flags(frame).mean())
    assert 0.005 < share < 0.02          # около 1 % по построению


def test_detector_explains_contribution():
    frame = _regime_frame()
    detector = RegimeAnomalyDetector.fit(frame, ["wabt", "h2_oil"])
    parts = detector.contributions({"wabt": 369.0, "h2_oil": 300.0})
    assert set(parts) == {"wabt", "h2_oil"}
    assert sum(parts.values()) == pytest.approx(1.0, abs=0.01)


def test_detector_without_fit_is_silent():
    detector = RegimeAnomalyDetector()
    assert detector.score_row({"wabt": 365.0}) is None
    assert detector.contributions({"wabt": 365.0}) == {}


# --------------------------------------------------------------------------- #
# пороги risk_class
# --------------------------------------------------------------------------- #

def test_risk_thresholds_are_used_from_agent():
    norms = SeverityNorms(bounds={"wabt": (355.0, 375.0)})
    strict = ReliabilityAgent(norms=norms, thresholds=(0.2, 0.4))
    lenient = ReliabilityAgent(norms=norms, thresholds=(0.8, 0.95))

    state = make_state()
    state.telemetry_ht.update({"T5": 370.0, "T6": 368.0, "T11": 366.0})

    assert strict.assess(state).risk_class == "high"
    assert lenient.assess(state).risk_class == "low"


def test_boundary_case_is_announced():
    """У границы класса решение зависит от весов — оператор должен это видеть."""
    norms = SeverityNorms(bounds={"wabt": (355.0, 375.0)})
    agent = ReliabilityAgent(norms=norms, thresholds=(0.30, 0.70))

    state = make_state()
    # severity ≈ 0.68 — почти порог «тяжёлого» режима
    state.telemetry_ht.update({"T5": 369.0, "T6": 369.0, "T11": 368.5})
    notes = " ".join(agent.assess(state).notes)
    assert "вплотную к порогу" in notes

    far = make_state()
    far.telemetry_ht.update({"T5": 358.0, "T6": 357.0, "T11": 356.0})
    assert "вплотную к порогу" not in " ".join(agent.assess(far).notes)


# --------------------------------------------------------------------------- #
# вердикты по подозрительным тегам
# --------------------------------------------------------------------------- #

def test_unusable_tags_are_not_in_features():
    """Тег со статусом do_not_use не должен молча вернуться в признаки."""
    from nefte.config import load_config
    from nefte.models.dataset import AVT_TAGS

    verdicts = load_config()["telemetry"].get("tag_verdicts", {})
    banned = {key.split(":")[1] for key, v in verdicts.items()
              if key.startswith("avt:") and v["status"] == "do_not_use"}

    assert banned, "вердикты должны быть заданы в configs/config.yaml"
    assert not (banned & set(AVT_TAGS)), (
        f"в признаках АВТ есть запрещённые теги: {banned & set(AVT_TAGS)}. "
        "Если решение изменилось — сначала обновите вердикт в конфиге."
    )


def test_every_verdict_has_a_reason():
    from nefte.config import load_config

    for tag, verdict in load_config()["telemetry"]["tag_verdicts"].items():
        assert verdict["status"] in {"usable", "do_not_use"}, tag
        assert len(verdict["reason"]) > 40, f"вердикт по {tag} без обоснования"


# --------------------------------------------------------------------------- #
# переходный режим
# --------------------------------------------------------------------------- #

def test_ramp_sees_a_feed_step_not_only_temperatures():
    """Скачок расхода сырья — тоже быстрое изменение режима.

    Пока фактор считался только по реакторным температурам, ступенька по сырью
    была для него невидимой, и система спокойно добавляла +2 °C поверх скачка.
    """
    n = 2000
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="10min", name="date")
    # обычный шум расхода нужен, иначе нормировать нечем: p95 обучающего периода
    # окажется нулём и фактор просто не соберётся
    feed = 250.0 + rng.normal(0, 1.5, n)
    feed[1500:] += 80.0                       # ступенька +32 %
    temps = {name: 360.0 + rng.normal(0, 0.2, n) for name in ("T5", "T6", "T11")}
    ht = pd.DataFrame({**temps, "F26": feed}, index=idx)
    avt = pd.DataFrame({"T55": 380.0 + rng.normal(0, 0.2, n)}, index=idx)
    cfg = {"split": {"train": ["2024-01-01", "2024-01-20"]}}

    with_feed = ReliabilityAgent.from_history(avt, ht, cfg).ramp_series
    without_feed = ReliabilityAgent.from_history(avt, ht.drop(columns=["F26"]),
                                                 cfg).ramp_series
    assert with_feed is not None and without_feed is not None

    # температуры только шумят: всплеск в момент ступеньки даёт именно расход
    assert with_feed.iloc[1500] > ReliabilityAgent.RAMP_LIMIT
    assert with_feed.iloc[1500] > with_feed.iloc[1400]
    # уберём расход из набора — и тот же момент перестанет выделяться:
    # ровно так фактор и вёл себя до правки
    assert with_feed.iloc[1500] > without_feed.iloc[1500]


def test_fast_ramp_forbids_raising_reactor_temperature():
    """В переходном режиме снижать можно, ужесточать нельзя."""
    idx = pd.date_range("2024-01-01", periods=50, freq="10min", name="date")
    agent = ReliabilityAgent(SeverityNorms(bounds={"wabt": (355.0, 375.0)}),
                             ramp_series=pd.Series(np.ones(len(idx)), index=idx))
    state = make_state()
    state.ts = idx[-1].to_pydatetime()

    out = agent.assess(state)
    assert out.factors["ramp"] > agent.RAMP_LIMIT
    for tag in ("T5", "T6", "T11"):
        current = state.telemetry_ht[tag]
        lo, hi = out.constraints[tag]
        assert hi <= current and lo < current
    assert any("меняется быстро" in note for note in out.notes)
