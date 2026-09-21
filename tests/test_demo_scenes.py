"""Сцены защиты показывают то, что обещают их описания.

Найдено 21.09: две сцены из пяти показывали не то. «Риск по качеству:
рекомендация с объяснением» после перехода на журнал замен катализатора стала
отказом «режим недопустим по тяжести», а «риск ниже порога» — действием при риске
20 %. Вторая сломалась тоньше: момент выбирали по журналу прогона, а прогон помнит
запрет частых воздействий, тогда как дашборд решает каждый момент с чистого листа.

Поэтому проверка идёт так же, как решает дашборд: запрет сбрасывается перед каждым
моментом (``app/dashboard.py``). Сломается настройка — сцена упадёт здесь, а не на
защите.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nefte.config import load_config


@pytest.fixture(scope="module")
def world():
    from nefte.pipeline import StateBuilder
    from scripts.demo import SCENES
    from scripts.run_cycle import build_system

    cfg = load_config()
    sb = StateBuilder(cfg)
    system = build_system(sb, cfg)
    system.log_runs = False
    return cfg, sb, system, {scene["key"]: scene for scene in SCENES}


def decide(world, ts: str):
    """Решение так, как его показывает дашборд: без памяти о прошлых действиях."""
    _, sb, system, _ = world
    state = sb.build(pd.Timestamp(ts))
    system._last_action_ts = None
    risk = system.quality.assess(state).spec_risk.get("product_sulfur_mgkg")
    return state, float(risk), system.run(state)


def moments(world, key: str) -> list[str]:
    ts = world[3][key]["ts"]
    return list(ts) if isinstance(ts, list) else [ts]


def test_unit_down_scene_refuses_because_the_unit_is_down(world):
    for ts in moments(world, "unit_down"):
        _, _, rec = decide(world, ts)
        assert rec.abstained and "остановлена" in rec.abstain_reason, ts


def test_bad_data_scene_refuses_because_of_the_data(world):
    for ts in moments(world, "bad_data_frozen_pak"):
        state, _, rec = decide(world, ts)
        assert rec.abstained, ts
        assert "остановлена" not in rec.abstain_reason, ts
        assert not state.data_quality.usable or rec.confidence < 0.35, ts


def test_stable_scene_holds_with_risk_below_the_threshold(world):
    threshold = world[2].act_risk_threshold
    for ts in moments(world, "stable"):
        _, risk, rec = decide(world, ts)
        assert rec.outcome() == "держим режим", ts
        assert risk < threshold, ts


def test_quality_risk_scene_actually_recommends(world):
    for ts in moments(world, "quality_risk"):
        _, _, rec = decide(world, ts)
        assert rec.outcome() == "меняем уставки", (ts, rec.abstain_reason)
        assert rec.blend is not None and rec.blend.feasible, ts


def test_watch_scene_risk_between_half_threshold_and_threshold(world):
    threshold = world[2].act_risk_threshold
    first, second = moments(world, "quality_watch")
    _, risk, rec = decide(world, first)
    assert rec.outcome() == "держим режим", first
    assert 0.5 * threshold <= risk < threshold, (first, risk)

    state, risk, rec = decide(world, second)
    assert rec.outcome() == "держим режим", second
    t95 = state.quality.get("lims_t95_c")
    limit = float(world[0]["spec"]["t95_c"]["max"])
    assert t95 is not None and t95.value > limit, second
