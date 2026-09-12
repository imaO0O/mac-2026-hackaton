# -*- coding: utf-8 -*-
"""Проверка «заденет ли правка прогон» обязана считать импорты транзитивно.

Смысл инструмента целиком в транзитивности: прямые импорты видно и глазами, а
подводят как раз дальние. `train_quality.py` нигде не упоминает
`nefte.data.validity`, но зависит от него через две ступени, и правка там
поехала бы в числа молча.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_run_safety.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("check_run_safety", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_run_safety"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_closure_reaches_dependencies_no_one_imports_directly(module):
    """Транзитивная зависимость обязана попадать в список."""
    deps = {p.relative_to(ROOT).as_posix()
            for p in module.closure(ROOT / "scripts" / "train_quality.py")}
    assert "src/nefte/models/dataset.py" in deps, "прямая зависимость потерялась"
    assert "src/nefte/data/validity.py" in deps, (
        "дальняя зависимость не найдена — обход перестал быть транзитивным, "
        "и проверка снова видит только то, что видно глазами")


def test_training_does_not_depend_on_the_agents(module):
    """Обучение не должно зависеть от агентов — иначе правка карточки двигает модель.

    Это не только про удобство прогонов. Агенты — слой решений, модель — слой
    измерения; если обучение начнёт зависеть от агента, любая правка текста в
    карточке оператора будет требовать переобучения, а расхождение между ними
    станет невидимым.
    """
    deps = {p.relative_to(ROOT).as_posix()
            for p in module.closure(ROOT / "scripts" / "train_quality.py")}
    leaked = sorted(d for d in deps
                    if d.startswith("src/nefte/agents/") and not d.endswith("schemas.py"))
    assert not leaked, f"обучение потянуло агентов: {leaked}"


def test_the_cycle_does_depend_on_them(module):
    """Обратное утверждение: у цикла решений зависимость от агентов обязана быть.

    Без этой половины первый тест проходил бы и на сломанном обходе, который
    просто ничего не находит.
    """
    deps = {p.relative_to(ROOT).as_posix()
            for p in module.closure(ROOT / "scripts" / "run_cycle.py")}
    assert "src/nefte/agents/quality.py" in deps
    assert "src/nefte/pipeline.py" in deps
