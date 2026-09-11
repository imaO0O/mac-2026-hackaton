"""Тесты оценки дезактивации катализатора.

Проверяем не «код запускается», а три свойства, без которых оценка остаточного
ресурса превращается в гадание: нормировка обязана двигаться в физически верную
сторону, смена катализатора обязана отличаться от обычного ремонта по ШАГУ
активности, а не по длительности останова, и непараметрическая оценка обязана
отвечать «сколько прожил цикл ПОСЛЕ этого уровня», а не «сколько всего».
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nefte.models.catalyst import (
    RESET_STEP_C,
    Cycle,
    normalized_wabt,
    outage_steps,
    remaining_by_analogue,
    sulfur_drift_per_month,
)


def _normalized(wabt: float, feed: float, sulfur_out: float,
                sulfur_in: float = 9300.0, feed_reference: float = 250.0,
                target: float = 8.0) -> float:
    index = pd.date_range("2024-01-01", periods=1, freq="D")
    series = [pd.Series([v], index=index) for v in (wabt, feed, sulfur_out, sulfur_in)]
    return float(normalized_wabt(*series, feed_reference=feed_reference,
                                 target_sulfur=target).iloc[0])


def test_normalized_wabt_equals_measured_at_reference_conditions():
    """Сера ровно эталонная, нагрузка ровно эталонная — нормировать нечего."""
    assert _normalized(365.0, 250.0, 8.0) == pytest.approx(365.0, abs=1e-6)


def test_worse_sulfur_means_higher_required_temperature():
    """Сера выше эталона — значит, для эталона понадобилась бы температура выше.

    Это и есть смысл нормировки: она переводит «сегодня получилось хуже» в
    «сегодня катализатор слабее на столько-то градусов».
    """
    assert _normalized(365.0, 250.0, 12.0) > 365.0 > _normalized(365.0, 250.0, 5.0)


def test_same_sulfur_at_higher_throughput_means_stronger_catalyst():
    """Та же сера при большей нагрузке — катализатор сильнее, не слабее.

    Нагрузка выше — время контакта меньше, и удержать серу на том же уровне
    труднее. Значит, на ЭТАЛОННОЙ нагрузке этому катализатору хватило бы
    температуры пониже. Без этой поправки рост производительности установки
    читался бы как дезактивация.
    """
    assert _normalized(365.0, 300.0, 8.0) < _normalized(365.0, 250.0, 8.0)


def test_deactivation_shows_up_as_drift_of_normalized_temperature():
    """Модельная проверка: катализатор слабеет, оператор поднимает температуру.

    Сера при этом держится постоянной, то есть по СЫРОЙ сере не видно ничего —
    видно только по нормированной температуре. Ради этого она и считается.
    """
    days = np.arange(0.0, 360.0, 10.0)
    index = pd.date_range("2024-01-01", periods=len(days), freq="10D")
    wabt = pd.Series(360.0 + 0.03 * days, index=index)      # ровно +0.913 °C/мес
    constant = lambda value: pd.Series(value, index=index)  # noqa: E731
    nwabt = normalized_wabt(wabt, constant(250.0), constant(8.0), constant(9300.0),
                            feed_reference=250.0)
    slope = float(np.polyfit(days, nwabt.to_numpy(), 1)[0]) * 30.44
    assert slope == pytest.approx(0.03 * 30.44, abs=0.05)


def _observations(levels: dict[str, float]) -> pd.DataFrame:
    """Наблюдения NWABT с заданным уровнем на каждом участке."""
    frames = [pd.DataFrame({"nwabt": level},
                           index=pd.date_range(start, periods=40, freq="D"))
              for start, level in levels.items()]
    return pd.concat(frames)


def test_long_outage_counts_as_catalyst_change_only_if_activity_returns():
    """Длительность останова сама по себе ничего не говорит.

    Два останова: после первого нормированная температура падает на 20 °C
    (катализатор сменили), после второго растёт (просто ремонт). Раньше сменой
    засчитывались оба, и «циклов» получалось больше, чем было на самом деле.
    """
    observations = _observations({"2024-01-01": 380.0, "2024-03-01": 360.0,
                                  "2024-06-01": 366.0})
    outages = [(pd.Timestamp("2024-02-11"), pd.Timestamp("2024-02-29")),
               (pd.Timestamp("2024-04-10"), pd.Timestamp("2024-05-31"))]
    steps = outage_steps(observations, outages)
    assert [s["смена катализатора"] for s in steps] == [True, False]
    assert steps[0]["шаг, °C"] <= RESET_STEP_C
    assert steps[1]["шаг, °C"] > 0


def _completed_cycle(days: np.ndarray, frame: pd.DataFrame) -> Cycle:
    return Cycle(index=0, start=frame.index[0], end=frame.index[-1],
                 n_points=len(frame), days=float(days.max()), nwabt_start=360.0,
                 nwabt_end=380.0, rate_c_per_month=1.5, rate_ci=(1.0, 2.0),
                 completed=True)


def test_analogue_answers_how_long_the_cycle_lived_after_this_level():
    """Аналогия обязана мерить ОСТАТОК, а не всю длину цикла."""
    days = np.arange(0.0, 400.0, 2.0)
    frame = pd.DataFrame({"cycle": 0, "run_days": days, "nwabt": 360.0 + 0.05 * days},
                         index=pd.date_range("2024-01-01", periods=len(days), freq="2D"))
    [row] = remaining_by_analogue(frame, [_completed_cycle(days, frame)], level_c=370.0)
    # уровень 370 достигается около 200-х суток из 398
    assert 180 <= row["достиг уровня на сутки"] <= 220
    assert row["оставалось, сут"] == row["вывод на сутки"] - row["достиг уровня на сутки"]


def test_unfinished_cycle_is_not_used_as_an_analogue():
    """У незавершённого цикла нет «остатка» — его нельзя приводить в пример."""
    frame = pd.DataFrame({"cycle": 0, "run_days": [0.0, 10.0], "nwabt": [360.0, 380.0]},
                         index=pd.date_range("2026-05-01", periods=2, freq="10D"))
    running = Cycle(index=0, start=frame.index[0], end=None, n_points=2, days=10.0,
                    nwabt_start=360.0, nwabt_end=380.0, rate_c_per_month=4.0,
                    rate_ci=(3.0, 5.0), completed=False)
    assert remaining_by_analogue(frame, [running], level_c=370.0) == []


def test_lost_activity_translates_into_rising_sulfur():
    """Градусы потерянной активности переводятся в миллиграммы серы."""
    drift = sulfur_drift_per_month(0.85, sulfur_out=8.0, sulfur_in=9300.0, wabt_c=365.0)
    assert 0.5 < drift < 5.0
    # вдвое быстрее теряем активность — вдвое быстрее растёт сера
    assert sulfur_drift_per_month(1.70, 8.0, 9300.0, 365.0) == pytest.approx(2 * drift)
