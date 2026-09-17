"""Вероятность, по которой принимаются решения.

Оркестратор сравнивает вероятность превышения с порогом, бюджет тревог задан в
тех же единицах. Значит, решающее правило стоит на том, что 0.2 действительно
означает «примерно один случай из пяти». Тесты закрепляют свойства, потеря
которых это предположение ломает молча.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.models.quality_model import SulfurModel, probability_metrics


def _frame(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    temp = rng.normal(367.0, 20.0, n)
    y = 8.5 - 0.1 * (temp - 367.0) + rng.normal(0.0, 1.5, n)
    X = pd.DataFrame({"ht_T5": temp, "ht_F17": rng.normal(248.0, 36.0, n)})
    return X, pd.Series(y, index=X.index)


def test_platt_applies_to_the_interval_source_too():
    """Раньше поправка доходила только до классификатора.

    Рабочая модель выбирает источником интервал, поэтому подобранная поправка не
    применялась НИКОГДА — при том что поле risk_calibration лежало в отчёте и
    утверждало обратное.
    """
    X, y = _frame()
    model = SulfurModel(iterations=80, monotone=False).fit(X, y)
    model.risk_source = "interval"
    model.risk_calibration = (0.5, -1.0)
    assert not model.predict_risk(X).equals(model.predict_risk(X, raw=True))


def test_calibration_is_refused_when_validation_base_rate_is_atypical():
    """Платт сдвигает УРОВЕНЬ к частоте того окна, где подобран.

    Если на валидации событий заметно больше, чем на обучении, поправка перенесёт
    в модель свойство периода. На наших данных ровно так и было: 19.5 % против
    15.1 %, и применённая поправка ухудшила калибровку на тесте.
    """
    X, y = _frame()
    model = SulfurModel(iterations=80, monotone=False).fit(X, y)
    model.risk_source = "interval"
    a, b = model.calibrate_risk(X, y, train_base_rate=float((y > model.limit).mean()) / 2)
    assert (a, b) == (1.0, 0.0)
    assert model.risk_calibration is None
    assert "не применена" in model.risk_calibration_note


def test_calibration_is_applied_when_base_rates_agree():
    """Обратная сторона: правило не должно запрещать поправку всегда."""
    X, y = _frame()
    model = SulfurModel(iterations=80, monotone=False).fit(X, y)
    model.risk_source = "interval"
    model.calibrate_risk(X, y, train_base_rate=float((y > model.limit).mean()))
    assert model.risk_calibration is not None
    assert model.risk_calibration_note == ""


def test_calibration_decision_survives_save_and_load(tmp_path):
    """Причина отказа должна пережить сохранение: иначе она исчезнет из отчёта."""
    X, y = _frame()
    model = SulfurModel(iterations=80, monotone=False).fit(X, y)
    model.risk_source = "interval"
    model.calibrate_risk(X, y, train_base_rate=float((y > model.limit).mean()) / 2)
    model.save(tmp_path / "m")
    restored = SulfurModel.load(tmp_path / "m")
    assert restored.risk_calibration is None
    assert "не применена" in restored.risk_calibration_note


def test_balanced_classifier_is_brought_back_to_the_training_rate():
    """Классификатор учится с весами Balanced — будто превышений половина.

    Без пересчёта его «вероятность» завышена по построению. На матрице версии 8 он
    выиграл выбор источника на валидации, поправка Платта не применилась, и средний
    риск на тесте стал 0.44 при частоте превышений 0.145. После пересчёта средняя
    вероятность на обучении обязана быть близка к частоте превышений там же.
    """
    X, y = _frame(n=800)
    model = SulfurModel(iterations=150, monotone=False).fit(X, y)
    rate = float((y > model.limit).mean())
    assert model.clf_train_rate == pytest.approx(rate)
    model.risk_source = "classifier"
    corrected = model.predict_risk(X, raw=True)
    balanced = pd.Series(model.clf.predict_proba(X[model.features])[:, 1], index=X.index)
    assert balanced.mean() > rate + 0.1
    assert abs(corrected.mean() - rate) < abs(balanced.mean() - rate) / 3


def test_classifier_takes_the_source_only_with_a_margin():
    """Источник риска не должен переключаться по шуму сида.

    PR-AUC классификатора на валидации скачет между сидами на 0.13, у интервала —
    на 0.04. На матрице версии 8 классификатор выиграл 0.0088 и забрал источник,
    а с ним и все решения системы. Правило: забирает, только если выигрывает с
    запасом MIN_SOURCE_MARGIN.
    """
    X, y = _frame(n=600)
    model = SulfurModel(iterations=120, monotone=False).fit(X, y)

    class Stub:
        """Классификатор с управляемым скором: проверяем правило, а не обучение."""

        def __init__(self, score):
            self.score = np.asarray(score)

        def predict_proba(self, X):
            return np.column_stack([1 - self.score, self.score])

    over = (y > model.limit).to_numpy()
    rng = np.random.default_rng(0)
    model.clf_train_rate = None          # скор стоящий, пересчёт тут не при чём
    interval = model.predict_risk(X, raw=True).to_numpy()

    # чуть лучше интервала — источником остаётся интервал
    model.clf = Stub(np.clip(interval + 0.02 * over, 1e-6, 1 - 1e-6))
    assert model.select_risk_source(X, y) == "interval"
    scores = model.risk_source_scores
    assert scores["classifier"] > scores["interval"], "проверять нечего: скор не лучше"
    assert scores["classifier"] - scores["interval"] < 0.05

    # заметно лучше — забирает
    strong = np.where(over, rng.uniform(0.7, 0.95, len(over)), rng.uniform(0.01, 0.2, len(over)))
    model.clf = Stub(strong)
    assert model.select_risk_source(X, y) == "classifier"
    assert (model.risk_source_scores["classifier"]
            >= model.risk_source_scores["interval"] + 0.05)


def test_prior_correction_keeps_the_ranking_and_the_extremes():
    model = SulfurModel()
    model.clf_train_rate = 0.15
    proba = np.array([0.01, 0.3, 0.5, 0.7, 0.99])
    corrected = model._undo_class_balance(proba)
    assert np.all(np.diff(corrected) > 0)
    # при p = 0.5 классификатор «не знает» — это и есть частота на обучении
    assert corrected[2] == pytest.approx(0.15)
    model.clf_train_rate = None
    assert np.array_equal(model._undo_class_balance(proba), proba)


def test_prior_correction_survives_save_and_load(tmp_path):
    X, y = _frame()
    model = SulfurModel(iterations=80, monotone=False).fit(X, y)
    model.save(tmp_path / "m")
    restored = SulfurModel.load(tmp_path / "m")
    assert restored.clf_train_rate == pytest.approx(model.clf_train_rate)
    model.risk_source = restored.risk_source = "classifier"
    assert np.allclose(model.predict_risk(X), restored.predict_risk(X))


# --------------------------------------------------------------------------- #
# сами метрики калибровки
# --------------------------------------------------------------------------- #

def test_perfect_probability_beats_the_constant():
    """Идеально калиброванная вероятность обязана бить «всегда базовая частота»."""
    rng = np.random.default_rng(1)
    p = pd.Series(rng.uniform(0.0, 1.0, 2000))
    y = pd.Series((rng.uniform(size=2000) < p).astype(int))
    m = probability_metrics(p, y)
    assert m["brier"] < m["brier_base"]
    assert abs(m["calibration_shift"]) < 0.05
    assert m["ece"] < 0.05


def test_systematically_inflated_probability_is_caught():
    """Главный случай, ради которого метрика заведена.

    Вероятность, завышенная вдвое, сохраняет ИДЕАЛЬНОЕ различение — ROC-AUC её не
    заметит вовсе. Перекос и ECE обязаны заметить.
    """
    rng = np.random.default_rng(2)
    true_p = pd.Series(rng.uniform(0.0, 0.4, 2000))
    y = pd.Series((rng.uniform(size=2000) < true_p).astype(int))
    inflated = (true_p * 2).clip(upper=1.0)
    honest = probability_metrics(true_p, y)
    bad = probability_metrics(inflated, y)
    assert bad["calibration_shift"] > 0.1
    assert bad["ece"] > honest["ece"] * 3


def test_metrics_are_empty_when_there_is_nothing_to_measure():
    """Без обоих классов калибровка не определена — возвращаем пусто, а не ноль."""
    p = pd.Series([0.1, 0.2, 0.3])
    assert probability_metrics(p, pd.Series([0, 0, 0])) == {}
