"""Пути к данным по умолчанию — от корня решения, а не от текущей папки.

Архив решения распаковывают на чужой машине; без `.env` данные ищутся в `data/`
рядом с кодом. Относительный путь от текущей папки сломался бы при первом запуске
скрипта не из корня.
"""
from __future__ import annotations

from pathlib import Path

import nefte.config as config


def test_relative_paths_resolve_against_the_solution_root(monkeypatch):
    monkeypatch.delenv("NEFTE_SOURCE_DIR", raising=False)
    monkeypatch.delenv("NEFTE_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    monkeypatch.chdir(Path(config.ROOT).parent)
    config.load_config.cache_clear()
    try:
        paths = config.load_config()["paths"]
    finally:
        config.load_config.cache_clear()
    assert Path(paths["source_dir"]) == config.ROOT / "data"
    assert Path(paths["data_dir"]) == config.ROOT / "data" / "cache"


def test_env_override_still_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("NEFTE_SOURCE_DIR", str(tmp_path))
    config.load_config.cache_clear()
    try:
        assert Path(config.load_config()["paths"]["source_dir"]) == tmp_path
    finally:
        config.load_config.cache_clear()
