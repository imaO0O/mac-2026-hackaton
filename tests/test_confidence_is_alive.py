# -*- coding: utf-8 -*-
"""Уверенность в карточке обязана отличать хороший срез от плохого.

История этой проверки стоит того, чтобы её записать, потому что она дважды
поймала меня самого.

Сначала уверенность считалась по одной σ и стояла на 0.95 почти всегда — это
записано упрёком в докстринге `confidence_parts`, и ради него добавлены
множители источника, свежести и возраста модели. Потом я проверил, что «теперь
шевелится», и получил красивую картину: медиана падает 0.898 в январе до 0.455 в
августе.

Картина держалась на арифметической ошибке. Срок годности модели был выведен как
«допустимые 0.5 мг/кг поделить на темп смещения» — время прохождения допуска ОТ
НУЛЯ, — а смещение начинается с −0.84 и по модулю сначала УБЫВАЕТ. Срок вышел
втрое короче реального, и множитель возраста резал уверенность сильнее всего там,
где измеренное смещение наименьшее. С исправленным сроком уверенность снова на
полке в 63 % срезов.

Поэтому здесь проверяется не «шевелится ли», а то, ради чего цифра нужна:
**различает ли она срезы, которым верить нельзя**. Это свойство не зависит от
того, какое число стоит в сроке годности, и переживёт следующий пересчёт дрейфа.
"""
from __future__ import annotations

import pytest

from nefte.agents.quality import BASELINE_SIGMA_MGKG, confidence_parts, fuse_sulfur
from nefte.agents.schemas import Source
from nefte.config import load_config

CEILING = 0.95
FLOOR = 0.05


@pytest.fixture(scope="module")
def slices():
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
        out.append({"уверенность": value, "пригоден": state.data_quality.usable,
                    "устарел": bool(m.is_stale)})
    return out


def _median(values):
    values = sorted(values)
    if not values:
        return float("nan")
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def test_unusable_slices_are_sharply_less_trusted(slices):
    """Непригодный срез обязан отличаться от пригодного в разы, а не на проценты."""
    good = [s["уверенность"] for s in slices if s["пригоден"]]
    bad = [s["уверенность"] for s in slices if not s["пригоден"]]
    assert good and bad, "на тестовом периоде должны быть срезы обоих видов"
    assert _median(good) > 5 * _median(bad), (
        f"пригодные {_median(good):.3f} против непригодных {_median(bad):.3f}: "
        "уверенность перестала отличать одно от другого")


def test_stale_source_is_sharply_less_trusted(slices):
    """То же для устаревшего источника: это вторая причина не верить цифре."""
    fresh = [s["уверенность"] for s in slices if not s["устарел"]]
    stale = [s["уверенность"] for s in slices if s["устарел"]]
    assert fresh and stale
    assert _median(fresh) > 5 * _median(stale), (
        f"свежий источник {_median(fresh):.3f} против устаревшего "
        f"{_median(stale):.3f}: множитель свежести перестал работать")


def test_no_age_penalty_inside_the_checked_range():
    """Внутри проверенного возраста уверенность за возраст НЕ режется.

    Прямое следствие разбора дрейфа: до края измеренного качество решений не
    ухудшается (MAE +0.0033/мес при ранговой корреляции +0.10, p=0.87), значит и
    резать не за что. Резать надо за незнание, которое начинается дальше.
    """
    shelf = load_config()["quality"]["model_shelf_life_months"]
    parts = confidence_parts(BASELINE_SIGMA_MGKG, Source.LIMS, 1.0, 24.0, True,
                             model_age_months=shelf, shelf_life_months=shelf)
    assert parts["возраст модели"] == pytest.approx(1.0), (
        "на краю срока годности множитель возраста обязан быть единицей")


def test_penalty_appears_beyond_the_checked_range():
    """А за краем — обязан появиться, иначе множитель мёртвый."""
    shelf = load_config()["quality"]["model_shelf_life_months"]
    parts = confidence_parts(BASELINE_SIGMA_MGKG, Source.LIMS, 1.0, 24.0, True,
                             model_age_months=2 * shelf, shelf_life_months=shelf)
    assert parts["возраст модели"] == pytest.approx(0.5), (
        "вдвое старше срока годности — уверенность обязана падать вдвое")


def test_shelf_life_does_not_exceed_what_drift_measured():
    """Срок годности не может уходить за край проверенного дрейфа.

    Ловит обратную ошибку к исправленной: если однажды срок поставят по
    экстраполяции (17.6 мес по смещению, 28 по выигрышу у персистенции), карточка
    начнёт молчать там, где мы ничего не измеряли.
    """
    import json
    import pathlib

    report = pathlib.Path(__file__).resolve().parents[1] / "reports" / "drift_h0.json"
    if not report.exists():
        pytest.skip("отчёт о дрейфе не собран: scripts/check_drift.py")
    data = json.loads(report.read_text(encoding="utf-8"))
    measured = (data.get("срок_годности") or {}).get("измерено_до_мес")
    if measured is None:
        pytest.skip("отчёт снят до того, как край измеренного стал записываться")
    shelf = load_config()["quality"]["model_shelf_life_months"]
    assert shelf <= measured, (
        f"срок годности {shelf} мес выходит за проверенные {measured:.1f} мес — "
        "это экстраполяция, а не измерение")
