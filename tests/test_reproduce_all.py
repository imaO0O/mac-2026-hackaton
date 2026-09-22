"""Каждый шаг пересборки ссылается на существующий скрипт.

Пересборка идёт часами, а шаг с опечаткой в имени скрипта падает только когда до
него дойдёт очередь, — и отчёт молча остаётся от прошлой сборки. Проверяется до
запуска: все `scripts/*.py` и `scripts/*.sh`, которые вызывает reproduce_all.sh,
лежат на месте.
"""
from __future__ import annotations

import re

from nefte.config import ROOT


def test_every_step_calls_an_existing_script():
    text = (ROOT / "scripts" / "reproduce_all.sh").read_text(encoding="utf-8")
    called = set(re.findall(r"\bscripts/[\w./-]+\.(?:py|sh)\b", text))
    assert called, "в reproduce_all.sh не нашлось ни одного вызова"
    missing = sorted(name for name in called if not (ROOT / name).exists())
    assert not missing, "нет скриптов: " + ", ".join(missing)
