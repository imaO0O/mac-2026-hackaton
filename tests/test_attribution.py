"""Разбор прогноза по группам каналов: что в нём обязано быть правдой.

Разбор ничего не решает — он объясняет. Именно поэтому его легко испортить
незаметно: неверно сложенные вклады или переехавшая группа не уронят ни один
прогон, а оператору будут показывать неправду. Тесты держат три вещи: сумма
вкладов сходится к тому самому прогнозу, что напечатан в карточке; строка
появляется ровно тогда, когда причина необычна; отсутствие модели не ломает
карточку, а просто снимает разбор.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.models.attribution import (
    CONTROLLABLE,
    GROUPS,
    MEASURED,
    OTHER,
    USUAL,
    cause_sentence,
    group_of,
    grouped_contributions,
    unusual_cause,
)
from nefte.models.quality_model import SulfurModel

SCALE = (0.3, 1.0)


def _frame(n: int = 300, seed: int = 0):
    """Синтетика с каналами из РАЗНЫХ групп: разбору есть что делить."""
    rng = np.random.default_rng(seed)
    temp = rng.normal(367.0, 20.0, n)              # режим реактора
    feed = rng.normal(1200.0, 200.0, n)            # сырьё с АВТ
    meter = rng.normal(8.5, 1.5, n)                # показания по сере
    load = rng.normal(248.0, 36.0, n)              # нагрузка
    y = (8.5 - 0.08 * (temp - 367.0) + 0.004 * (feed - 1200.0)
         + 0.3 * (meter - 8.5) + rng.normal(0.0, 1.0, n))
    X = pd.DataFrame({"ht_T5": temp, "avt_T66": feed,
                      "ht_Q21": meter, "ht_F26": load})
    return X, pd.Series(y, index=X.index)


@pytest.fixture(scope="module")
def fitted():
    X, y = _frame()
    return SulfurModel(iterations=80, monotone=False).fit(X, y), X


def test_groups_know_the_channels_they_are_named_after():
    assert group_of("ht_Q21_mean6") == MEASURED
    assert group_of("avt_T66") == USUAL
    assert group_of("vak_AVT6_350_I350") == USUAL
    assert group_of("reg_wabt_dev30") == "режим реактора"
    assert group_of("reg_h2_partial") == "водород"
    assert group_of("ht_F26") == "нагрузка и квенч"
    # незнакомый канал не теряется и не приписывается чужой группе
    assert group_of("ht_ZZ99") == OTHER


def test_group_names_do_not_repeat():
    names = [name for name, _ in GROUPS]
    assert len(names) == len(set(names))
    assert set(CONTROLLABLE) <= set(names)


def test_contributions_add_up_to_the_very_forecast_on_the_card(fitted):
    """База плюс вклады = прогноз. Иначе разбор объясняет не то число."""
    model, X = fitted
    for position in (0, 7, 42, len(X) - 1):
        row = X.iloc[[position]]
        grouped, base = grouped_contributions(model, row)
        predicted = float(model.predict_frame(row)["q50"].iloc[0])
        assert base + float(grouped.sum()) == pytest.approx(predicted, abs=1e-6)


def test_line_appears_exactly_when_the_cause_is_unusual(fitted):
    """Никаких «почти»: строка есть тогда и только тогда, когда ведёт не сырьё."""
    model, X = fitted
    seen = {True: 0, False: 0}
    for position in range(0, len(X), 5):
        row = X.iloc[[position]]
        grouped, _ = grouped_contributions(model, row)
        rest = grouped.drop(index=[MEASURED], errors="ignore")
        leader_is_usual = rest.abs().idxmax() == USUAL
        cause = unusual_cause(model, row)
        assert (cause is None) is leader_is_usual
        seen[leader_is_usual] += 1
        if cause is not None:
            assert cause["группа"] != USUAL
            assert cause["вклад, мг/кг"] == pytest.approx(
                float(rest[cause["группа"]]), abs=0.005)
    # обе ветки должны встретиться, иначе тест проверяет одну половину правила
    assert seen[True] > 0 and seen[False] > 0


def test_small_contribution_is_called_small(fitted):
    """Вклад меньше эффекта одного градуса — это объяснение, а не повод двигать."""
    cause = {"группа": "режим реактора", "вклад, мг/кг": 0.07,
             "обычная причина": USUAL, "управляемая": True}
    text = cause_sentence(cause, SCALE)
    assert "меньше, чем даёт один градус" in text
    assert "+0.07" in text

    big = {**cause, "вклад, мг/кг": 0.9}
    assert "уставками" in cause_sentence(big, SCALE)
    assert "меньше, чем даёт один градус" not in cause_sentence(big, SCALE)


def test_unmanageable_cause_says_so():
    cause = {"группа": OTHER,
             "вклад, мг/кг": 0.5, "обычная причина": USUAL, "управляемая": False}
    assert "не правится" in cause_sentence(cause, SCALE)


def test_no_model_means_no_line_instead_of_a_crash(fitted):
    """Персистенция вместо модели — обычный режим работы, а не сбой."""
    _, X = fitted
    assert unusual_cause(None, X.iloc[[0]]) is None
    assert grouped_contributions(object(), X.iloc[[0]]) is None
    assert cause_sentence(None, SCALE) == ""


def test_row_without_the_needed_columns_is_refused(fitted):
    model, X = fitted
    assert grouped_contributions(model, X.iloc[[0]].drop(columns=["ht_T5"])) is None
    assert grouped_contributions(model, X.iloc[0:0]) is None


def test_groups_cover_the_working_model():
    """Переименовали тег — группа опустеет, и разбор тихо станет бессмысленным."""
    path = SulfurModel.default_path(0.0, "sulfur")
    if not path.exists():
        pytest.skip(f"нет обученной модели {path}")
    model = SulfurModel.load(path)
    counts = pd.Series([group_of(f) for f in model.features]).value_counts()
    for name in (MEASURED, USUAL, "режим реактора", "водород", "нагрузка и квенч"):
        assert counts.get(name, 0) > 0, f"группа «{name}» пуста"
    # незнакомых каналов должно быть немного: иначе разбор объясняет «прочим»
    assert counts.get(OTHER, 0) / len(model.features) <= 0.20
