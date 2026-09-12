# -*- coding: utf-8 -*-
"""Счёт, зависящий от матрицы, обязан записывать, на какой сборке он посчитан.

Проверка структурная, а не по списку имён, и вот почему. Список имён уже
подводил дважды: отчёты нейросетей не записывали версию, и контракт свежести
молча пропускал все девять; отчёты имитации и прогона по тестовому периоду
оказались в том же положении, и нашлось это случайно — когда числа из них попали
в README.

Оба раза дело было не в том, что кто-то поленился, а в том, что ничто не
связывало «скрипт зависит от матрицы» с «отчёт называет версию». Здесь эта связь
и проверяется: если скрипт транзитивно тянет `models/dataset` и пишет в
`reports/`, он обязан пользоваться `nefte.provenance.report_provenance` — либо
стоять в списке исключений с причиной, которую видно.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from nefte.config import ROOT

SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"

# Скрипты, которые тянут dataset транзитивно, но НЕ зависят от матрицы по сути.
# Каждое исключение — с причиной: молчаливых здесь быть не должно.
EXEMPT = {
    "train_anomaly_ae.py": (
        "автоэнкодер читает телеметрию и ПАК напрямую (load_telemetry, load_pak); "
        "с dataset его связывает только константа единиц PCT_TO_MGKG через pipeline"),
    "prepare_data.py": "готовит сырые данные до того, как матрица вообще существует",
    "demo.py": "показывает готовые отчёты, своих не пишет",
    "run_cycle.py": "печатает карточку, отчётов в reports/ не пишет",
    "run_blending.py": "рецептура считается от лабораторных анализов, не от матрицы",
    "check_tag_meaning.py": "сверяет смысл тегов по сырой телеметрии",
    "find_delays.py": "ищет запаздывание по сырым рядам, матрицу не строит",
    "check_cetane.py": "цетановое число берётся из ЛИМС, матрица не участвует",
    "check_t95_sigma.py": "разброс Т95 считается по парам лабораторных анализов",
    "train_quality.py": "сам обучает модель и пишет версию напрямую",
    "train_sequence.py": "сам обучает модель и пишет версию напрямую",
    "backtest_reliability.py": "бэктест агента надёжности идёт по телеметрии",
    "run_blend_scenarios.py": "сценарии смешения считаются от лабораторных анализов",
    "check_avt_formulas.py": "сверка формул АВТ с лабораторией, матрица не участвует",
    "check_catalyst_life.py": "ресурс катализатора считается по телеметрии и ЛИМС",
}


def _module_file(name: str) -> pathlib.Path | None:
    rel = pathlib.Path(*name.split("."))
    for candidate in (SRC / rel.with_suffix(".py"), SRC / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _direct_imports(path: pathlib.Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith("nefte"))
        elif isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("nefte"):
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def _closure(entry: pathlib.Path) -> set[str]:
    seen: set[str] = set()
    queue = list(_direct_imports(entry))
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_file(name)
        if path is not None:
            queue.extend(_direct_imports(path))
    return seen


def _writes_reports(path: pathlib.Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return 'write_text(json.dumps' in text and '"reports"' in text or "reports" in text \
        and "write_text(json.dumps" in text


def _candidates() -> list[pathlib.Path]:
    return sorted(p for p in SCRIPTS.glob("*.py")
                  if p.name != "__init__.py" and _writes_reports(p)
                  and "nefte.models.dataset" in _closure(p))


@pytest.mark.parametrize("path", _candidates(), ids=lambda p: p.name)
def test_script_declares_the_build_it_ran_on(path: pathlib.Path):
    """Либо пользуется общим источником происхождения, либо объявлен исключением."""
    text = path.read_text(encoding="utf-8")
    if "report_provenance" in text or "FEATURE_VERSION" in text:
        return
    reason = EXEMPT.get(path.name)
    assert reason, (
        f"{path.name} зависит от матрицы признаков и пишет отчёт, но не записывает "
        "версию сборки. Добавьте report_provenance() в отчёт или внесите скрипт в "
        "EXEMPT с причиной — молчаливых исключений здесь быть не должно")


def test_the_exemption_list_has_no_ghosts():
    """В списке исключений не должно быть скриптов, которых уже нет."""
    missing = sorted(name for name in EXEMPT if not (SCRIPTS / name).exists())
    assert not missing, f"в списке исключений скрипты, которых нет: {missing}"


def test_there_is_something_to_check():
    """Сама выборка не должна опустеть — иначе проверка выше ничего не проверяет."""
    assert len(_candidates()) >= 8
