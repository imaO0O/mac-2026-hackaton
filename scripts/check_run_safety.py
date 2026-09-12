# -*- coding: utf-8 -*-
"""Заденет ли правка уже запущенный (или ещё не запущенный) длинный прогон.

    python scripts/check_run_safety.py train_quality.py
    python scripts/check_run_safety.py train_sequence.py --since HEAD~3

Скрипт отвечает на один вопрос: **пересекаются ли файлы, которые я правлю, с
тем, что этот прогон импортирует.** Если нет — правки безопасны, прогон можно
не трогать. Если да — числа поедут, и прогон придётся повторять.

Почему это отдельная команда, а не «посмотреть глазами». За один день ошибка
порядка работ случилась дважды:

* матрицу пересобрали четыре раза, и последняя пересборка застала запущенное
  обучение сетей — прогон пришлось повторять целиком;
* в `train_sequence.py` добавили запись версии матрицы, пока прогон сетей уже
  шёл; отчёты вышли со свежими числами, но без поля версии, и проверка свежести
  стала молча их пропускать.

Оба раза вопрос был тот же самый и решался за секунды — если его задать. Глазами
он решается плохо, потому что импорты транзитивны: `train_quality.py` не
упоминает `nefte.data.validity`, но зависит от него через две ступени.

ВАЖНО про уже идущий прогон. Python читает исходник при импорте, поэтому
правка не влияет на УЖЕ загруженный процесс. Но длинные прогоны у нас — это
оболочечные циклы из нескольких запусков python, и каждый следующий запуск
берёт уже новый код. Именно так внутри одного задания ранние шаги оказались на
старом коде, а поздние на новом.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = "nefte"


def module_path(name: str) -> pathlib.Path | None:
    """Файл модуля пакета по его имени."""
    rel = pathlib.Path(*name.split("."))
    for candidate in (SRC / rel.with_suffix(".py"), SRC / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def imports_of(path: pathlib.Path) -> set[str]:
    """Имена модулей пакета, импортируемые файлом напрямую."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith(PACKAGE))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(PACKAGE):
                out.add(node.module)
                # from nefte.models import dataset — подмодули тоже считаются
                out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def closure(entry: pathlib.Path) -> set[pathlib.Path]:
    """Все файлы пакета, от которых зависит точка входа. Транзитивно."""
    seen_modules: set[str] = set()
    files: set[pathlib.Path] = set()
    queue = list(imports_of(entry))
    while queue:
        name = queue.pop()
        if name in seen_modules:
            continue
        seen_modules.add(name)
        path = module_path(name)
        if path is None:
            continue
        files.add(path)
        queue.extend(imports_of(path))
    return files


def changed_files(since: str | None) -> list[pathlib.Path]:
    """Что изменено: незакоммиченное, либо всё начиная с ревизии."""
    cmd = (["git", "diff", "--name-only", since] if since
           else ["git", "status", "--porcelain"])
    try:
        raw = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"не удалось спросить git: {exc}")
        return []
    out = []
    for line in raw.splitlines():
        name = line[3:] if not since else line
        name = name.strip().strip('"')
        if name:
            out.append(ROOT / name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="имя скрипта в scripts/, например train_quality.py")
    ap.add_argument("--since", default=None,
                    help="сравнивать с этой ревизией вместо незакоммиченного")
    args = ap.parse_args()

    entry = ROOT / "scripts" / args.script
    if not entry.exists():
        print(f"нет такого скрипта: {entry}")
        return 2

    deps = closure(entry)
    changed = [p for p in changed_files(args.since) if p.suffix == ".py"]
    config_changed = [p for p in changed_files(args.since)
                      if p.name.endswith((".yaml", ".yml"))]

    print(f"{args.script} зависит от {len(deps)} файлов пакета:")
    for path in sorted(deps):
        print("   ", path.relative_to(ROOT).as_posix())

    clash = sorted(p for p in changed if p.resolve() in {d.resolve() for d in deps})
    print()
    if not changed and not config_changed:
        print("изменённых файлов нет — вопрос не стоит")
        return 0

    print("изменено:", ", ".join(p.relative_to(ROOT).as_posix()
                                 for p in changed + config_changed) or "ничего")
    print()
    if clash:
        print("ПЕРЕСЕЧЕНИЕ — прогон затронут:")
        for path in clash:
            print("   ", path.relative_to(ROOT).as_posix())
        print("Числа поедут. Идущий прогон надо повторить, будущий — запускать после правок.")
    else:
        print("пересечения нет: правки не входят в зависимости прогона")
    if config_changed:
        print()
        print("конфиг тронут — проверьте отдельно, читает ли прогон изменённые ключи:")
        for path in config_changed:
            print("   ", path.relative_to(ROOT).as_posix())
        print("Отпечаток матрицы (dataset.cache_key) включает не весь конфиг, "
              "поэтому добавление ключа обычно безопасно, а правка существующего — нет.")
    return 1 if clash else 0


if __name__ == "__main__":
    raise SystemExit(main())
