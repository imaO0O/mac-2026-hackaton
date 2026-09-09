"""Однократная подготовка данных: распаковка архива + сборка parquet-кэша.

    python scripts/prepare_data.py            # распаковать и построить кэш
    python scripts/prepare_data.py --check    # только проверить, что всё на месте

Пути берутся из configs/config.yaml и переменных окружения NEFTE_SOURCE_DIR /
NEFTE_DATA_DIR. Сами данные в git не попадают (см. .gitignore).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import data_dir, load_config, source_dir  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

UNRAR_CANDIDATES = [
    r"C:\Program Files\WinRAR\UnRAR.exe",
    r"C:\Program Files\7-Zip\7z.exe",
    "unrar",
    "7z",
]


def find_extractor() -> tuple[str, list[str]]:
    for exe in UNRAR_CANDIDATES:
        path = exe if Path(exe).exists() else shutil.which(exe)
        if path:
            args = ["x", "-o+", "-idq"] if "rar" in Path(path).name.lower() else ["x", "-y"]
            return path, args
    raise RuntimeError(
        "Не найден распаковщик rar. Установите WinRAR или 7-Zip, либо распакуйте "
        "data.rar вручную в <NEFTE_DATA_DIR>/raw/"
    )


def unpack(force: bool = False) -> None:
    cfg = load_config()
    archive = source_dir() / cfg["paths"]["files"]["archive"]
    raw = data_dir() / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    targets = [raw / "avt_tags.csv", raw / "242000_tags.csv"]
    if all(t.exists() for t in targets) and not force:
        print(f"[skip] csv уже распакованы в {raw}")
        return
    if not archive.exists():
        raise FileNotFoundError(f"{archive} не найден")

    exe, args = find_extractor()
    print(f"[unrar] {archive} → {raw} ({Path(exe).name})")
    subprocess.run([exe, *args, str(archive), str(raw) + "\\"], check=True)

    # архив содержит папку data/ — поднимаем файлы на уровень raw/
    nested = raw / "data"
    if nested.exists():
        for f in nested.iterdir():
            f.replace(raw / f.name)
        nested.rmdir()


def build_cache() -> None:
    from nefte.data.loaders import (
        load_lims,
        load_pak,
        load_tag_dictionary,
        load_telemetry,
    )

    for name, fn in [
        ("telemetry avt", lambda: load_telemetry("avt")),
        ("telemetry ht", lambda: load_telemetry("ht")),
        ("lims", load_lims),
        ("pak", load_pak),
        ("tags", load_tag_dictionary),
    ]:
        obj = fn()
        size = len(obj) if not isinstance(obj, dict) else sum(len(v) for v in obj.values())
        print(f"[cache] {name}: {size} записей")


def check() -> int:
    ok = True
    for path in [source_dir(), data_dir() / "raw" / "avt_tags.csv",
                 data_dir() / "raw" / "242000_tags.csv"]:
        exists = Path(path).exists()
        ok &= exists
        print(f"{'OK ' if exists else 'НЕТ'} {path}")
    return 0 if ok else 1


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="только проверка путей")
    ap.add_argument("--force", action="store_true", help="перераспаковать архив")
    a = ap.parse_args()

    if a.check:
        return check()
    unpack(force=a.force)
    build_cache()
    print(f"\nГотово. Кэш: {data_dir() / 'cache'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
