"""Имитация обязана показывать системе последствия её собственных действий.

Дефект, ради которого файл заведён. Обученная модель качества берёт признаки из
матрицы по метке времени, то есть из истории. Имитатор подменял в срезе
измеренную серу смоделированной и считал контур замкнутым, но прогноз и риск от
подмены не зависели: подставленные 2 и 15 мг/кг давали одинаковое решение. Прогон
с тремя разными «физиками» процесса дал серу 2.6, 5.9 и 7.0 мг/кг при ОДНИХ И ТЕХ
ЖЕ 25 вмешательствах — система не видела, что уже сделала.

Тесты держат три вещи: обёртка без сдвига ничего не меняет; сдвиг двигает прогноз
и риск в нужную сторону; имитатор подключает обёртку к системе, а процесс
оставляет на исходном суррогате — иначе сдвиг посчитался бы дважды.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source
from nefte.models.kinetics import make_kinetic_surrogate
from nefte.sim import ClosedLoopSimulator, FeedbackModel


class _Model:
    horizon_hours = 0.0
    alarm_threshold = 0.18

    def __init__(self, level=9.0, sigma=1.2, risk=0.30):
        self.level, self.sigma, self.risk = level, sigma, risk

    def predict_with_sigma(self, state):
        return self.level, self.sigma

    def risk_for_state(self, state):
        return self.risk


def _state() -> ProcessState:
    return ProcessState(
        ts=datetime(2026, 6, 1, 12, 0), telemetry_avt={},
        telemetry_ht={"T5": 370.0, "T6": 365.0, "T11": 366.0,
                      "F26": 250.0, "P13": 3.9, "F25": 13600.0},
        quality={"lims_feed_sulfur_mgkg": Measurement(
            value=9000.0, unit="мг/кг", source=Source.LIMS, age_hours=5.0)},
        data_quality=DataQuality(missing_share=0.0, usable=True),
    )


def test_zero_shift_changes_nothing():
    model, state = _Model(), _state()
    wrapped = FeedbackModel(model)
    assert wrapped.predict_with_sigma(state) == model.predict_with_sigma(state)
    assert wrapped.risk_for_state(state) == model.risk_for_state(state)
    assert wrapped.alarm_threshold == model.alarm_threshold


def test_shift_moves_forecast_and_risk_the_right_way():
    state = _state()
    wrapped = FeedbackModel(_Model())
    wrapped.shift = -3.0
    mean, _ = wrapped.predict_with_sigma(state)
    lower = wrapped.risk_for_state(state)
    wrapped.shift = +3.0
    higher = wrapped.risk_for_state(state)
    assert mean == pytest.approx(6.0)
    assert lower < 0.30 < higher
    assert 0.0 < lower and higher < 1.0


def test_simulator_wires_feedback_to_the_system_but_not_to_the_process():
    model = _Model()
    original = make_kinetic_surrogate(model)
    system = SimpleNamespace(quality=SimpleNamespace(model=model),
                             optimizer=SimpleNamespace(surrogate=original))
    sim = ClosedLoopSimulator(state_builder=None, system=system, surrogate=original)

    assert isinstance(system.quality.model, FeedbackModel)
    assert system.optimizer.surrogate is not original
    assert sim.surrogate is original, "процесс обязан отвечать по исходному суррогату"

    state = _state()
    sim.feedback.shift = -2.0
    seen = system.optimizer.surrogate(state, {})["product_sulfur_mgkg"]
    truth = sim.surrogate(state, {})["product_sulfur_mgkg"]
    assert seen == pytest.approx(7.0)
    assert truth == pytest.approx(9.0)
    # тот же момент, другой сдвиг: кэш уровня не должен вернуть старое значение
    sim.feedback.shift = -4.0
    assert system.optimizer.surrogate(state, {})["product_sulfur_mgkg"] == pytest.approx(5.0)


def test_persistence_system_is_left_alone():
    """Без модели риск считается по измерению из среза, и подмены серы достаточно."""
    surrogate = lambda state, moves: {"product_sulfur_mgkg": 8.0}  # noqa: E731
    system = SimpleNamespace(quality=SimpleNamespace(model=None),
                             optimizer=SimpleNamespace(surrogate=surrogate))
    sim = ClosedLoopSimulator(state_builder=None, system=system, surrogate=surrogate)
    assert sim.feedback is None
    assert system.optimizer.surrogate is surrogate


def test_outage_resets_accumulated_setpoint_shifts():
    """Сдвиги уставок не переживают останов.

    Без сброса имитация применяла сдвиг, сделанный до останова, к холодному
    реактору, и Аррениус при 40 °C раздувал два градуса в четырёхкратный рост серы.
    """
    stamps = [datetime(2026, 4, 14, h, 0) for h in (0, 4, 8)]
    down_from = stamps[2]

    builder = SimpleNamespace(build=lambda ts: _state_at(ts))
    applied = []
    calls = {"n": 0}

    def run(state):
        calls["n"] += 1
        acts = calls["n"] == 1
        action = SimpleNamespace(id="act", deltas={"T5": 1.0}) if acts else None
        return SimpleNamespace(abstained=not acts and state.ts >= down_from,
                               action=action, confidence=0.8,
                               outcome=lambda: "меняем уставки" if acts else "держим режим")

    system = SimpleNamespace(
        quality=SimpleNamespace(model=None),
        optimizer=SimpleNamespace(surrogate=None),
        reliability=SimpleNamespace(is_unit_down=lambda state: state.ts >= down_from),
        run=run, record_applied=applied.append)
    surrogate = lambda state, moves: {"product_sulfur_mgkg": 8.0}  # noqa: E731
    sim = ClosedLoopSimulator(builder, system, surrogate, t95_fn=lambda s, m: None)
    steps = sim.run(stamps)

    assert steps[1].offsets == {"T5": 1.0} and not steps[1].reset
    assert steps[2].reset and steps[2].offsets == {}
    assert applied[-1] == {"T5": -1.0}, "оркестратор обязан узнать, что сдвиг снят"


def _state_at(ts) -> ProcessState:
    state = _state()
    return state.model_copy(update={"ts": ts})
