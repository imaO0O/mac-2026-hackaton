# -*- coding: utf-8 -*-
"""Ссылки вида «docs/HARD_CHECKS.md §8.7» обязаны указывать на существующий раздел.

Ссылка по номеру ломается молча: достаточно вставить раздел в середину, и все
последующие номера означают не то, что означали. Это не гипотеза — так и вышло,
когда в документе обнаружились ДВА раздела с номером 8.6, и перенумерация хвоста
развела пять ссылок из разных файлов.

Проверяются и документы, и код: половина ссылок живёт в докстрингах скриптов, и
там расхождение заметить ещё труднее.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from nefte.config import ROOT

REFERENCE = re.compile(r"([A-Z_]+\.md)`?\s*§\s*([0-9]+(?:\.[0-9]+)*)")
# Номер раздела целиком, с точкой после него или без: «## 3.» и «### 3.1 » —
# оба законны. Без взгляда вперёд на пробел «3.1» разбиралось как «3» с точкой,
# и каждый подраздел выглядел дублем своего раздела.
HEADING = re.compile(r"^#+\s*([0-9]+(?:\.[0-9]+)*)\.?(?=\s)", re.M)


def _sections(doc: Path) -> set[str]:
    return set(HEADING.findall(doc.read_text(encoding="utf-8")))


def _sources() -> list[Path]:
    out: list[Path] = []
    for pattern in ("docs/*.md", "scripts/*.py", "src/nefte/**/*.py", "tests/*.py",
                    "README.md"):
        out.extend(ROOT.glob(pattern))
    return [p for p in out if p.is_file()]


def _references() -> list[tuple[Path, str, str]]:
    found = []
    for path in _sources():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for doc, section in REFERENCE.findall(text):
            found.append((path, doc, section))
    return found


def test_every_section_reference_resolves():
    """Каждая ссылка §N.M указывает на раздел, который есть в документе."""
    cache: dict[str, set[str]] = {}
    broken = []
    for path, doc, section in _references():
        target = ROOT / "docs" / doc
        if not target.exists():
            broken.append(f"{path.name}: нет документа {doc}")
            continue
        if doc not in cache:
            cache[doc] = _sections(target)
        # ссылка на §5–7 или §5.6 — проверяем ровно то, что написано
        if section not in cache[doc]:
            broken.append(f"{path.relative_to(ROOT).as_posix()}: {doc} §{section} "
                          f"не существует")
    assert not broken, "битые ссылки на разделы:\n  " + "\n  ".join(broken)


def test_no_document_has_two_sections_with_the_same_number():
    """Двух разделов с одним номером быть не должно — из-за этого всё и поехало."""
    for doc in (ROOT / "docs").glob("*.md"):
        numbers = HEADING.findall(doc.read_text(encoding="utf-8"))
        duplicates = {n for n in numbers if numbers.count(n) > 1}
        assert not duplicates, f"{doc.name}: номера разделов повторяются: {sorted(duplicates)}"


def test_there_are_references_to_check():
    """Сама выборка не должна опустеть."""
    assert len(_references()) >= 10
