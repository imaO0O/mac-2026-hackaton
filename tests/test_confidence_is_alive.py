# -*- coding: utf-8 -*-
"""Уверенность в карточке обязана шевелиться, иначе она украшение.

Это проверка ровно того упрёка, ради которого множители и вводились: раньше
уверенность считалась по одной σ и стояла на 0.95 почти всегда. Упрёк был
записан в докстринге `confidence_parts`, но никогда не проверялся числом — а
заявление «теперь шевелится» ничем не отличается от прежнего, пока его не
измерили.

Меряется на тестовом периоде, потому что именно там модель стареет: конец
обучения 2025-06-30, тест идёт весь 2026 год, и возраст модели проходит от
6.6 до 13.1 месяца при сроке годности 6.
"""
from __future__ import annotations

import pytest

from nefte.agents.quality import BASELINE_SIGMA_MGKG, confidence_parts, fuse_sulfur
from nefte.agents.schemas import Source
from nefte.config import load_config

CEILING = 0.95
FLOOR = 0.05


@pytest.fixture(scope="module")
def confidences():
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from nefte.pipeline import StateBuilder

    cfg = load_config()
    builder = StateBuilder(cfg)
    stale = cfg["quality"]["staleness_hours"]
    shelf = cfg["quality"]["model_shelf_life_months"]
    train_end = pd.Timestamp(cfg["split"]["train"][1])
    lo, hi = cfg["split"]["test"]

    out = []
    for ts in pd.date_range(lo, hi, freq="12h"):
        state = builder.build(ts)
        m = fuse_sulfur(state, cfg)
        threshold = stale["lims"] if m.source == Source.LIMS else stale["pak"]
        age_months = max(0.0, (ts - train_end).total_seconds() / 86400.0 / 30.44)
        parts = confidence_parts(
            BASELINE_SIGMA_MGKG, m.source, m.age_hours, threshold,
            state.data_quality.usable,
            model_age_months=age_months, shelf_life_months=shelf)
        value = max(FLOOR, min(CEILING, float(np.prod(list(parts.values())))))
        out.append((ts, value))
    return out


def test_confidence_does_not_sit_on_the_ceiling(confidences):
    """Меньше четверти срезов на потолке — иначе цифра снова украшение."""
    share = sum(1 for _, c in confidences if c >= CEILING - 0.001) / len(confidences)
    assert share < 0.25, (
        f"уверенность на потолке в {share:.1%} срезов — она перестала нести "
        "информацию, проверьте множители")


def test_confidence_falls_as_the_model_ages(confidences):
    """Первая половина тестового периода увереннее второй.

    Это не свойство данных, а обязательное следствие измеренного дрейфа: модель
    обучена один раз, и чем дальше от конца обучения, тем меньше оснований ей
    верить. Если порядок когда-нибудь перевернётся — значит множитель возраста
    отключился или перестал доходить до карточки.
    """
    half = len(confidences) // 2
    first = sum(c for _, c in confidences[:half]) / half
    second = sum(c for _, c in confidences[half:]) / (len(confidences) - half)
    assert first > second + 0.1, (
        f"первая половина теста {first:.3f}, вторая {second:.3f}: уверенность "
        "перестала падать со старением модели")
