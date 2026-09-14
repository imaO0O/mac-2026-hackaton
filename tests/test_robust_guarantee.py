"""Робастная гарантия: запас по сере должен держаться и при слабом отклике.

Отклик серы на уставки принят (кинетика первого порядка), а не измерен. С
пессимистичным суррогатом вариант получает «гарантированный запас» только если
проходит предел при обеих кинетиках. Бездействие не затрагивается, и без
пессимистичного суррогата поведение прежнее.
"""
from __future__ import annotations

from nefte.agents.optimizer import OptimizerAgent
from nefte.agents.quality import QualityAgent
from nefte.agents.reliability import ReliabilityAgent
from tests.test_agents import make_state
from tests.test_decision_defects import NORMS

BOUNDS = {"T5": (365.0, 375.0), "T11": (360.0, 370.0), "F26": (200.0, 300.0),
          "P13": (3.5, 4.2)}


def _surrogate(gain: float):
    """Линейный отклик: сера 9.0 при текущем режиме, gain мг/кг на градус T5."""
    def fn(state, moves):
        now = state.telemetry_ht["T5"]
        return {"product_sulfur_mgkg": 9.0 + gain * (moves.get("T5", now) - now)}
    return fn


def _evaluate(robust):
    state = make_state(lims=(9.0, 1.0), pak=(9.0, 0.1))
    reliability = ReliabilityAgent(NORMS)
    r = reliability.assess(state)
    q = QualityAgent().assess(state)
    optimizer = OptimizerAgent(bounds=BOUNDS, surrogate=_surrogate(-1.5),
                               robust_surrogate=robust, t95_fn=lambda s, m: None)
    cands = optimizer.evaluate(state, optimizer.generate(state, r), q, r)
    return {c.id: c for c in cands}


def test_without_robust_surrogate_nothing_changes():
    plain = _evaluate(None)
    assert any(c.guaranteed for c in plain.values() if c.id != "hold")
    assert all("product_sulfur_mgkg_weak_kinetics" not in c.predicted_quality
               for c in plain.values())


def test_weak_response_withdraws_guarantees_it_cannot_back():
    plain = _evaluate(None)
    robust = _evaluate(_surrogate(-0.3))          # отклик впятеро слабее
    lost = [cid for cid, c in plain.items()
            if cid != "hold" and c.guaranteed and not robust[cid].guaranteed]
    assert lost, "слабый отклик не снял ни одной гарантии — проверка не работает"
    for cid in lost:
        weak = robust[cid].predicted_quality["product_sulfur_mgkg_weak_kinetics"]
        assert weak > robust[cid].predicted_quality["product_sulfur_mgkg"]


def test_hold_is_not_touched():
    plain, robust = _evaluate(None), _evaluate(_surrogate(-0.3))
    assert plain["hold"].guaranteed == robust["hold"].guaranteed
    assert "product_sulfur_mgkg_weak_kinetics" not in robust["hold"].predicted_quality


def test_nominal_guarantee_ranks_above_plain_improvement():
    """Когда при слабом отклике запаса нет ни у кого, выше стоит вариант с запасом
    хотя бы при принятой кинетике — а не просто лучший по свёртке."""
    from nefte.agents.optimizer import OptimizerAgent

    optimizer = OptimizerAgent(bounds=BOUNDS, surrogate=_surrogate(-1.5),
                               robust_surrogate=_surrogate(-0.01), t95_fn=lambda s, m: None)
    state = make_state(lims=(9.0, 1.0), pak=(9.0, 0.1))
    reliability = ReliabilityAgent(NORMS)
    r = reliability.assess(state)
    q = QualityAgent().assess(state)
    ranked = optimizer.rank(optimizer.evaluate(state, optimizer.generate(state, r), q, r))
    assert not any(c.guaranteed for c in ranked if c.id != "hold")
    nominal = [c for c in ranked if c.guaranteed_nominal]
    assert nominal, "при принятой кинетике запас должен быть хоть у кого-то"
    first_plain = next((i for i, c in enumerate(ranked) if not c.guaranteed_nominal), len(ranked))
    assert all(ranked.index(c) < first_plain for c in nominal)
