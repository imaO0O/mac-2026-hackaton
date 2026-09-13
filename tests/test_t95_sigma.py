# -*- coding: utf-8 -*-
"""Неопределённость Т95 зависит от возраста опорного анализа, а не от одного числа.

Уровень Т95 берётся из последнего лабораторного анализа, поэтому σ — это разброс
ухода показателя за время с момента отбора пробы. Пока σ была константой,
измеренной на медианном шаге в сутки, она занижалась всякий раз, когда анализ
оказывался старше, — а старше суток он в трети моментов решения.

Тесты закрепляют три разных утверждения, и разделены именно поэтому: два первых
про форму кривой (её можно пересчитать, и выводы не должны молча измениться),
третий — про то, что обе карточки одного цикла берут σ из одного места.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nefte.agents.quality import T95_SIGMA_C, t95_sigma

REPORT = pathlib.Path(__file__).resolve().parents[1] / "reports" / "t95_sigma.json"


@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip("отчёт не собран: scripts/check_t95_sigma.py")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_sigma_grows_with_age_and_never_shrinks(report):
    """Разброс не может уменьшаться с возрастом анализа."""
    sigmas = [b["sigma_c"] for b in report["buckets"]]
    assert sigmas == sorted(sigmas), (
        f"σ по вёдрам идёт не по возрастанию: {sigmas} — либо мало пар в ведре, "
        "либо в ряд попал брак")


def test_the_series_reverts_rather_than_wanders(report):
    """Полка, а не корень из времени — именно это оправдывает ступеньку.

    Если ряд начнёт вести себя как случайное блуждание, ступенчатая таблица с
    полкой на последнем ведре станет неверной: σ будет расти и дальше, а мы
    перестанем её догонять.
    """
    grown = report["sigma_growth_ratio"]
    walk = report["random_walk_would_give"] / report["buckets"][0]["sigma_c"]
    assert grown < 0.5 * walk, (
        f"σ выросла в {grown:.2f} раза при {walk:.2f} у блуждания — ряд перестал "
        "возвращаться к среднему, полку на последнем ведре надо пересмотреть")


def test_unknown_age_takes_the_worst_measured_not_the_best():
    """Возраст неизвестен — берётся худшее измеренное, а не среднее.

    Обратный порядок был бы самообманом: незнание возраста не делает анализ свежим.
    """
    assert t95_sigma(None) >= t95_sigma(1.0)
    assert t95_sigma(10_000.0) == t95_sigma(None), (
        "за последним ведром σ обязана выходить на полку, а не падать")


def test_fallback_constant_matches_the_freshest_bucket(report):
    """Запасная константа — это σ самого свежего ведра, а не старое число."""
    assert abs(T95_SIGMA_C - report["buckets"][0]["sigma_c"]) < 0.05


def test_quality_and_optimizer_use_the_same_sigma():
    """Одна величина не может иметь двух неопределённостей в одном цикле."""
    import inspect

    from nefte.agents import optimizer

    src = inspect.getsource(optimizer)
    # Сначала общей была только σ, теперь общая вся вероятность нарушения: к σ
    # добавилась поправка калибровки, и две карточки одного цикла обязаны
    # получать одно и то же число, а не одну и ту же σ.
    assert "t95_violation_risk(" in src, (
        "оптимизатор обязан брать вероятность нарушения Т95 той же функцией, "
        "что и агент качества")
    assert "T95_SIGMA_C" not in src, (
        "оптимизатор снова взял константу напрямую — карточки разъедутся")
