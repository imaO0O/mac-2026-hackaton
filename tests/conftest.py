"""Режим «без выданных данных» для проверки в CI.

Данные хакатона в репозитории не хранятся, поэтому на чистой машине часть тестов
читать нечего. Молча пропускать их нельзя — пропуск неотличим от успеха, — и
падать тоже неверно: упал бы не код, а отсутствие файлов.

Поэтому режим явный. С переменной окружения ``NEFTE_NO_DATA=1`` тест, упавший ровно
на том, что нет файла из каталога выданных данных или их кэша, отмечается
пропущенным с именем файла. Любая другая ошибка остаётся ошибкой. Без переменной
поведение прежнее: локально отсутствие данных — это падение.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _data_roots() -> list[str]:
    try:
        from nefte.config import load_config

        paths = load_config()["paths"]
        return [str(Path(paths["source_dir"])), str(Path(paths["data_dir"]))]
    except Exception:                                   # noqa: BLE001
        return []


def _missing_data_file(exc: BaseException | None, roots: list[str]) -> str | None:
    """Имя отсутствующего файла выданных данных, если ошибка — именно это."""
    while exc is not None:
        if isinstance(exc, FileNotFoundError):
            name = str(exc.filename or exc)
            if any(root and root.lower() in name.lower() for root in roots):
                return name
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not os.getenv("NEFTE_NO_DATA") or not report.failed or call.excinfo is None:
        return
    missing = _missing_data_file(call.excinfo.value, _data_roots())
    if missing:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0,
                           f"нет выданных данных: {Path(missing).name}")
