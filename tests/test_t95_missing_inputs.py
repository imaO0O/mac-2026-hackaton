# -*- coding: utf-8 -*-
"""Пустой вход формулы Т95, который вариант не трогает, не отключает проверку.

В 11 % моментов теста расходы газа 24-2000 F9 и F2 были отбракованы очисткой, и оценка Т95
при любом ходе T5 возвращала «не знаем». С ней молча не проверялось жёсткое
ограничение оптимизатора по Т95. Оптимизатор эти теги не двигает, и в приращении
их значения сокращаются — поэтому приращение считается и без них.

Но T6 «тронут», когда двигают T5 или T11, даже если самого T6 в срезе нет:
подставлять за него нельзя, иначе ход T5 молча дал бы нулевое приращение.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_t95 import make_state  # noqa: E402

from nefte.agents.optimizer import default_t95_estimator  # noqa: E402


def _without(state, *tags):
    ht = {k: (None if k in tags else v) for k, v in state.telemetry_ht.items()}
    return state.model_copy(update={"telemetry_ht": ht})


def test_missing_untouched_inputs_give_the_same_increment():
    fn = default_t95_estimator()
    full = make_state()
    gap = _without(make_state(), "F9", "F2")
    move = {"T5": full.telemetry_ht["T5"] + 2.0}
    with_all = fn(full, move)
    with_gap = fn(gap, move)
    assert with_all is not None
    assert with_gap is not None, "пустые F9/F2 снова отключают оценку Т95"
    assert abs(with_all - with_gap) < 1e-9


def test_missing_t6_under_a_t5_move_stays_unknown():
    fn = default_t95_estimator()
    state = _without(make_state(), "T6")
    assert fn(state, {"T5": state.telemetry_ht["T5"] + 2.0}) is None


def test_hold_is_still_the_lab_value():
    fn = default_t95_estimator()
    state = _without(make_state(), "F9", "F2", "T6")
    assert fn(state, {}) == 355.0
