"""Один код — две разные величины: значение гидроочистки не берётся у тега АВТ.

Восемь кодов есть на обеих установках и означают разное (`docs/AVT_SCHEMES.md` §4):
`T11` АВТ — 3-е ЦО при 65 °C, `T11` 24-2000 — температура реакторного блока;
`F26` АВТ — пар в К-6, `F26` 24-2000 — расход сырья. В коде значение ищется
сначала в словаре гидроочистки, потом в словаре АВТ. Это безопасно ровно до тех
пор, пока пустое значение гидроочистки хранится в срезе как ключ с None: тогда
поиск останавливается на нём и не уходит к одноимённому тегу АВТ. `StateBuilder`
так и делает — кладёт в срез все колонны установки, пустые как None.

Значения АВТ в тестах нарочно положены ВНУТРЬ границ гидроочистки. С настоящими
(65 °C, −5.6) тест прошёл бы и при подстановке: такие значения в границы не
попадают, и тег выпадает сам. Парная проверка показывает, что тест чувствителен:
если ключа гидроочистки в срезе нет вовсе, значение АВТ подставляется.
"""
from __future__ import annotations

from nefte.agents.optimizer import OptimizerAgent
from nefte.agents.reliability import ReliabilityAssessment
from tests.test_agents import make_state

BOUNDS = {"T11": (360.0, 370.0), "F26": (200.0, 300.0)}
# одноимённые теги АВТ со значениями, которые ПРОШЛИ бы в границы гидроочистки
AVT_IN_RANGE = {"T11": 365.0, "F26": 250.0}


def _optimizer() -> OptimizerAgent:
    return OptimizerAgent(bounds=BOUNDS, surrogate=lambda s, m: {"product_sulfur_mgkg": 8.0})


def _reliability(state) -> ReliabilityAssessment:
    return ReliabilityAssessment(ts=state.ts, severity_index=0.3, risk_class="low",
                                 admissible=True, constraints={}, factors={}, notes=[])


def test_empty_ht_value_is_not_taken_from_avt_namesake():
    for tag in BOUNDS:
        state = make_state()
        state.telemetry_ht[tag] = None
        state.telemetry_avt.update(AVT_IN_RANGE)
        bounds = _optimizer()._effective_bounds(state, _reliability(state))
        assert tag not in bounds, f"{tag}: границы построены по одноимённому тегу АВТ"


def test_the_check_is_sensitive_when_the_ht_key_is_absent():
    """Обратная сторона: без ключа гидроочистки поиск уходит в АВТ.

    Поэтому срез обязан содержать все колонны установки. Если этот тест начнёт
    падать, значит поиск стал строже, и первый тест можно усилить.
    """
    state = make_state()
    del state.telemetry_ht["T11"]
    state.telemetry_avt.update(AVT_IN_RANGE)
    bounds = _optimizer()._effective_bounds(state, _reliability(state))
    assert "T11" in bounds


def test_present_ht_value_wins_over_avt_namesake():
    state = make_state()
    state.telemetry_avt.update({"T11": 64.7})
    lo, hi = _optimizer()._effective_bounds(state, _reliability(state))["T11"]
    # 365 гидроочистки ± шаг, а не 64.7 АВТ
    assert 360.0 <= lo <= hi <= 370.0
