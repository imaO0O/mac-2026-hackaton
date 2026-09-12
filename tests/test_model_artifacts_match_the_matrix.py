# -*- coding: utf-8 -*-
"""Сохранённая модель обязана уметь питаться ТЕКУЩЕЙ матрицей признаков.

Устаревший артефакт модели — не то же, что устаревший отчёт, и опаснее его.
Отчёт даёт неверные числа, и это ловит контракт свежести. Модель ПАДАЕТ, и
падает так, что причину надо раскапывать.

Так и вышло. После перевода серы сырья в мг/кг колонка сменила имя
(`lims_feed_sulfur` → `lims_feed_sulfur_mgkg`). Модель сети на горизонте 2 ч,
обученная до этого, хранила старое имя в списке каналов — и прогон по тестовому
периоду упал с `KeyError: ['lims_feed_sulfur'] not in index` из середины
подготовки окон, где не видно ни модели, ни причины.

Проверяется здесь не «все модели свежие» — переобучать всё на каждый чих незачем,
— а то, что **несовпадение обнаруживается сразу и называется по имени**.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nefte.config import ROOT

MODELS = ROOT / "models"


def _sequence_models() -> list[pathlib.Path]:
    return sorted(p for p in MODELS.glob("sulfur_seq_*") if (p / "meta.json").exists())


@pytest.fixture(scope="module")
def matrix():
    pytest.importorskip("pandas")
    from nefte.models.dataset import build_feature_matrix
    return build_feature_matrix()


@pytest.mark.parametrize("path", _sequence_models(), ids=lambda p: p.name)
def test_saved_channels_exist_in_the_matrix(path: pathlib.Path, matrix):
    """Каналы сохранённой модели должны быть в матрице — иначе она непригодна."""
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    missing = [c for c in meta.get("channels", []) if c not in matrix.columns]
    assert not missing, (
        f"{path.name} обучена на каналах, которых в матрице больше нет: "
        f"{missing}. Переобучите или удалите артефакт — иначе он падает при "
        "первой же попытке им воспользоваться")


def test_the_failure_is_named_not_raw(matrix):
    """Несовпадение обязано быть понятным отказом, а не KeyError изнутри.

    Проверяется на подделанном списке каналов, а не на реальной устаревшей
    модели: реальные мы как раз чиним, и тест не должен зависеть от того,
    осталась ли хоть одна сломанная.
    """
    torch = pytest.importorskip("torch")  # noqa: F841
    from nefte.models.sequence import SulfurSequenceModel

    paths = _sequence_models()
    if not paths:
        pytest.skip("обученных сетей нет")
    model = SulfurSequenceModel.load(paths[0])
    model.channels = list(model.channels) + ["канала_такого_нет"]

    with pytest.raises(RuntimeError) as err:
        model.attach(matrix)
    text = str(err.value)
    assert "канала_такого_нет" in text, "отказ не называет недостающий канал"
    assert "train_sequence.py" in text, "отказ не говорит, что делать"
