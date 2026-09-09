"""Мелкие утилиты, не относящиеся ни к одному агенту."""
from __future__ import annotations

import sys

__all__ = ["use_utf8_console"]


def use_utf8_console() -> None:
    """Переводит вывод скриптов в UTF-8.

    Консоль Windows по умолчанию отдаёт cp866/cp1251, и любой вывод со стрелкой
    «→» или знаком «σ» роняет скрипт с UnicodeEncodeError — падает не расчёт, а
    печать результата. Демо и отчёты на русском, поэтому вызываем это первым
    делом в каждом скрипте. Ошибки кодирования не глушим совсем, а заменяем
    символом: лучше «?» в одном знаке, чем упавший прогон.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
