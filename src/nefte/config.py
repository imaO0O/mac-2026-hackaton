"""Загрузка конфигурации и разрешение путей к данным.

Приоритет источников пути: переменная окружения > configs/config.yaml.
Так у каждого участника свой локальный путь к данным, а конфиг в git один.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Читает config.yaml и применяет переопределения из окружения."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if env := os.getenv("NEFTE_SOURCE_DIR"):
        cfg["paths"]["source_dir"] = env
    if env := os.getenv("NEFTE_DATA_DIR"):
        cfg["paths"]["data_dir"] = env
    return cfg


def source_dir() -> Path:
    """Папка с исходной выдачей организаторов (xlsx / pdf / data.rar)."""
    return Path(load_config()["paths"]["source_dir"])


def data_dir() -> Path:
    """Локальный кэш: raw/ (распакованные csv) и cache/ (parquet)."""
    return Path(load_config()["paths"]["data_dir"])


def source_file(key: str) -> Path:
    return source_dir() / load_config()["paths"]["files"][key]


def data_file(key: str) -> Path:
    return data_dir() / load_config()["paths"]["files"][key]


def cache_dir() -> Path:
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d
