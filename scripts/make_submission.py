"""Собрать архив решения для отправки и проверить его на вменяемость.

    python scripts/make_submission.py                 # архив кода, ~3 МБ — письмом
    python scripts/make_submission.py --data          # данные и модели — ссылкой
    python scripts/make_submission.py --out D:/oil/neftekod.zip --no-reports

Архивов два, потому что организаторы просили приложить данные (сессия 18.09,
docs/DATA_NOTES.md §5в), а в письмо сто мегабайт не влезают.

**Архив кода.** Код (`src`, `scripts`, `app`, `tests`), конфигурация, документация с
презентацией, README, образец `.env` и отчёты `reports/*.json` — всё, чем
подтверждён каждый вывод. Отчёты входят: без них проверяющему пришлось бы верить на
слово или ждать многочасовой пересборки.

**Архив данных (`--data`).** Выданные файлы организаторов без изменений — ровно те,
что читает код (`configs/config.yaml → paths.files`), — и обученные модели
`models/`: с ними карточка и дашборд запускаются сразу, без обучения. Модели можно и
пересобрать: `python scripts/train_quality.py --horizon 0` или
`bash scripts/reproduce_all.sh`. Кэши (распакованные csv, parquet) не входят — их
делает `python scripts/prepare_data.py`.

Проверка после сборки — не формальность. Архив открывается заново и проверяется:
на месте ли точки входа, не утекли ли данные или модели, читается ли README, и
лежит ли внутри раздел «если на проверку есть пятнадцать минут». Если что-то не
так, скрипт падает и архив не остаётся: лучше не отправить ничего, чем отправить
архив, который у проверяющего не запустится.
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

# Что кладём в архив: каталоги целиком и отдельные файлы из корня.
TREES = ("src", "scripts", "app", "tests", "configs", "docs")
FILES = ("README.md", "requirements.txt", "pyproject.toml", "pytest.ini",
         "setup.cfg", ".gitattributes", ".env.example")
REPORTS = "reports"

# Что не кладём никогда: кэши, окружение, выданные данные и обученные модели.
SKIP_DIRS = {"__pycache__", ".venv", ".git", ".pytest_cache", ".reproduce",
             ".ipynb_checkpoints", "node_modules"}
SKIP_GLOBS = ("*.pyc", "*.pyo", "*.log", "*.cbm", "*.pt", "*.pth", "*.joblib",
              "*.pkl", "*.parquet", "*.csv", "*.xlsx", "*.xls", "*.zip")

# Без этих файлов архив бессмыслен: проверяющий не поймёт, что запускать.
REQUIRED = ("README.md", "configs/config.yaml", "scripts/reproduce_all.sh",
            "scripts/run_cycle.py", "src/nefte/agents/orchestrator.py",
            "app/dashboard.py")
# Раздел README, ради которого он вообще читается первым.
README_MARK = "За пять минут"


def wanted(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    return not any(fnmatch.fnmatch(path.name, pattern) for pattern in SKIP_GLOBS)


def collect(with_reports: bool) -> list[Path]:
    picked: list[Path] = []
    for name in TREES:
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and wanted(path.relative_to(ROOT)):
                picked.append(path)
    for name in FILES:
        path = ROOT / name
        if path.is_file():
            picked.append(path)
    if with_reports:
        for path in sorted((ROOT / REPORTS).rglob("*.json")):
            if wanted(path.relative_to(ROOT)):
                picked.append(path)
    return picked


def verify(archive: Path, with_reports: bool) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        broken = zf.testzip()
        assert broken is None, f"битый файл в архиве: {broken}"
        for name in REQUIRED:
            assert name in names, f"в архиве нет {name}"
        leaked = [n for n in names
                  if n.startswith(("data/", "models/", ".venv/", ".reproduce/"))
                  or n.endswith((".cbm", ".pt", ".csv", ".parquet"))]
        assert not leaked, f"в архив попало лишнее: {leaked[:5]}"
        readme = zf.read("README.md").decode("utf-8")
        assert README_MARK in readme, "в README нет раздела для быстрой проверки"
        if with_reports:
            assert any(n.startswith("reports/") for n in names), "нет отчётов"
        # запускаемость кода проверяем по синтаксису: архив не должен содержать
        # файл, который не компилируется, — это признак обрезанной записи
        import ast

        for name in sorted(n for n in names if n.endswith(".py")):
            ast.parse(zf.read(name).decode("utf-8"), filename=name)
    print(f"проверка архива пройдена: {len(names)} файлов")


# Как пользоваться архивом данных — кладётся в сам архив.
DATA_README = """Данные и модели к решению «Нефтекод»

data/    выданные файлы организаторов без изменений
models/  обученные модели (можно пересобрать, см. README решения)

1. Распакуйте рядом с кодом решения; папку models/ положите в корень решения.
2. В .env решения (образец — .env.example) укажите:
     NEFTE_SOURCE_DIR=<путь к распакованной папке data>
     NEFTE_DATA_DIR=<любая папка для кэша, вне облачной синхронизации>
3. python scripts/prepare_data.py   — распаковка и кэш, один раз
4. python scripts/run_cycle.py --ts "2026-03-05 00:00"
"""


def collect_data() -> list[tuple[Path, str]]:
    """Выданные файлы, которые читает код, и модели: (путь, имя в архиве)."""
    from nefte.config import load_config, source_dir

    picked = []
    for key, name in load_config()["paths"]["files"].items():
        path = source_dir() / name
        if path.is_file():               # csv в files — это кэш в data_dir, не выдача
            picked.append((path, f"data/{name}"))
    assert picked, f"в {source_dir()} нет выданных файлов — проверьте NEFTE_SOURCE_DIR"
    for path in sorted((ROOT / "models").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            picked.append((path, path.relative_to(ROOT).as_posix()))
    return picked


def build_data(out: Path) -> int:
    files = collect_data()
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("ДАННЫЕ.txt", DATA_README)
        for path, name in files:
            # rar и xlsx уже сжаты — повторное сжатие только тратит время
            kind = zipfile.ZIP_STORED if path.suffix in (".rar", ".xlsx") else zipfile.ZIP_DEFLATED
            zf.write(path, name, compress_type=kind)
    with zipfile.ZipFile(tmp) as zf:
        assert zf.testzip() is None, "битый файл в архиве данных"
        names = set(zf.namelist())
        assert any(n.startswith("data/") for n in names), "нет данных"
        assert any(n.startswith("models/sulfur_h0/") for n in names), "нет рабочей модели"
    tmp.replace(out)
    data = sum(1 for _, n in files if n.startswith("data/"))
    print(f"архив данных: {out} — {out.stat().st_size / 1e6:.1f} МБ; "
          f"выданных файлов {data}, файлов моделей {len(files) - data}")
    return 0


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data", action="store_true",
                        help="архив выданных данных и обученных моделей вместо кода")
    parser.add_argument("--no-reports", action="store_true",
                        help="без reports/*.json (архив меньше, доказательств нет)")
    args = parser.parse_args()
    if args.data:
        return build_data(args.out or ROOT.parent / "neftekod_data.zip")
    args.out = args.out or ROOT.parent / "neftekod_submission.zip"

    with_reports = not args.no_reports
    files = collect(with_reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())
    verify(tmp, with_reports)
    tmp.replace(args.out)

    size = args.out.stat().st_size / 1e6
    by_tree: dict[str, int] = {}
    for path in files:
        by_tree[path.relative_to(ROOT).parts[0]] = \
            by_tree.get(path.relative_to(ROOT).parts[0], 0) + 1
    print(f"\nархив: {args.out} — {size:.1f} МБ, файлов {len(files)}")
    for name, count in sorted(by_tree.items(), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {count}")
    print("\nданные и модели — отдельным архивом: "
          "python scripts/make_submission.py --data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
