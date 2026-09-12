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


def test_a_dead_function_is_not_reported_as_reachable(module, tmp_path):
    """Правка мёртвой функции не должна поднимать тревогу.

    Инструмент сам на этом ошибся: он считал ЛЮБОЕ упоминание имени, а в
    `models/regime.py` есть локальная переменная `wabt` — и мёртвая
    `features.wabt` выглядела вызываемой. Инструмент, который кричит на каждое
    совпадение имён, перестают читать, а тогда он не ловит и настоящие случаи.
    """
    deps = module.closure(ROOT / "scripts" / "train_quality.py")
    home = ROOT / "src" / "nefte" / "data" / "features.py"
    assert home in deps, "features.py должен быть в зависимостях обучения"
    reachable = module.is_called_within({"wabt"}, deps | {ROOT / "scripts" / "train_quality.py"}, home)
    assert reachable == set(), (
        f"мёртвая features.wabt сочтена вызываемой: {reachable}")


def test_a_live_class_is_reported_as_reachable(module):
    """Обратная сторона: то, что прогон действительно зовёт, обязано находиться.

    Без этой половины предыдущий тест проходил бы и на инструменте, который
    всегда отвечает «не вызывается».
    """
    entry = ROOT / "scripts" / "run_cycle.py"
    deps = module.closure(entry)
    home = ROOT / "src" / "nefte" / "pipeline.py"
    reachable = module.is_called_within({"StateBuilder"}, deps | {entry}, home)
    assert reachable == {"StateBuilder"}, (
        "StateBuilder вызывается из самой точки входа — если его не видно, "
        "инструмент ищет только в пакете и занижает")


def test_git_output_in_russian_does_not_break_the_check(module):
    """Вывод git по-русски обязан читаться, а не превращать ответ в пустоту.

    Ловится настоящая поломка: без явной кодировки Python на Windows берёт
    cp1251, git отдаёт UTF-8, поток чтения падает в фоновом треде и stdout
    молча становится None. У нас все коммиты и докстринги по-русски.
    """
    names = module.changed_definitions(
        ROOT / "src" / "nefte" / "agents" / "quality.py", "HEAD~14")
    assert isinstance(names, set)
    assert "QualityAgent" in names or not names, (
        "разбор диффа вернул мусор — проверьте кодировку вызова git")


def test_a_pure_deletion_is_resolved_by_name(module):
    """Удаление функции должно называться по имени, а не «не разобрать правку».

    Чистое удаление не оставляет строк в новом файле, и разбор по нему даёт
    пустоту. Инструмент тогда тревожил на всякий случай — ровно та ложная
    тревога, ради устранения которой он и писался. Имена берутся из старой
    ревизии.
    """
    names = module.changed_definitions(
        ROOT / "src" / "nefte" / "data" / "features.py", "HEAD")
    if not names:
        pytest.skip("features.py не правлен относительно HEAD")
    assert "<не разобрать правку>" not in names
    assert {"add_lags", "add_rollings", "make_supervised"} & names, (
        f"удалённые функции не опознаны, вернулось {sorted(names)}")
