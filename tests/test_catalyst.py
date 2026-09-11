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
    lag_against_reference,
    level_at_runday,
    local_rate,
    normalized_wabt,
    outage_steps,
    remaining_by_analogue,
    remaining_by_lag,
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


def _two_cycles() -> pd.DataFrame:
    """Два цикла: эталонный выходит на полку, текущий продолжает греться."""
    days = np.arange(0.0, 200.0, 2.0)
    reference = pd.DataFrame({
        "cycle": 1, "run_days": days,
        # рост до 60-х суток, дальше полка — так вёл себя настоящий цикл 1
        "nwabt": 355.0 + np.minimum(days, 60.0) * 0.08,
    }, index=pd.date_range("2024-01-01", periods=len(days), freq="2D"))
    current = pd.DataFrame({
        "cycle": 2, "run_days": days,
        "nwabt": 355.0 + days * 0.08,          # полки нет
    }, index=pd.date_range("2026-01-01", periods=len(days), freq="2D"))
    return pd.concat([reference, current])


def test_level_at_runday_reports_an_interval_not_just_a_number():
    frame = _two_cycles()
    level = level_at_runday(frame, cycle=1, day=100.0, half_window=20)
    assert level["уровень, °C"] == pytest.approx(359.8, abs=0.2)
    low, high = level["95% ДИ"]
    assert low <= level["уровень, °C"] <= high


def test_level_at_runday_refuses_to_answer_on_thin_data():
    """Меньше восьми анализов в окне — ответа нет, а не ответ наугад."""
    frame = _two_cycles()
    assert level_at_runday(frame, cycle=1, day=100.0, half_window=1) is None


def test_lag_is_called_significant_only_when_the_interval_excludes_zero():
    """До расхождения траекторий отставание обязано быть незначимым."""
    frame = _two_cycles()
    rows = lag_against_reference(frame, cycle=2, reference=1, days=[40.0, 150.0],
                                 half_window=20)
    early, late = rows[0], rows[1]
    assert early["отставание, °C"] == pytest.approx(0.0, abs=0.2)
    assert not early["значимо"]
    # на 150-х сутках эталон давно на полке, а текущий всё греется
    assert late["отставание, °C"] > 5.0
    assert late["значимо"]


def test_remaining_by_lag_subtracts_the_lag_from_what_the_reference_lived():
    """Способ (г): факт по эталону минус наше отставание, переведённое в месяцы."""
    frame = _two_cycles()
    reference = Cycle(index=1, start=pd.Timestamp("2024-01-01"),
                      end=pd.Timestamp("2025-06-01"), n_points=100, days=500.0,
                      nwabt_start=355.0, nwabt_end=380.0, rate_c_per_month=1.0,
                      rate_ci=(0.8, 1.2), completed=True)
    current = Cycle(index=2, start=pd.Timestamp("2026-01-01"), end=None, n_points=100,
                    days=150.0, nwabt_start=355.0, nwabt_end=367.0,
                    rate_c_per_month=2.0, rate_ci=(1.5, 2.5), completed=False)
    out = remaining_by_lag(frame, [reference, current], current, reference,
                           rate_c_per_month=1.0)
    # эталон с 150-х суток прожил ещё 350 суток = 11.5 мес, отставание ~7 °C
    assert out["эталон прожил ещё, мес"] == pytest.approx(11.5, abs=0.1)
    assert out["отставание, °C"] > 5.0
    assert out["остаток, мес"] == pytest.approx(
        out["эталон прожил ещё, мес"] - out["отставание, мес наработки"], abs=0.1)


def test_left_censored_cycle_cannot_be_the_reference():
    """У левообрезанного цикла нет «тех же суток»: его наработка отсчитана не от пуска."""
    frame = _two_cycles()
    censored = Cycle(index=1, start=pd.Timestamp("2024-01-01"),
                     end=pd.Timestamp("2025-06-01"), n_points=100, days=500.0,
                     nwabt_start=355.0, nwabt_end=380.0, rate_c_per_month=1.0,
                     rate_ci=(0.8, 1.2), completed=True, left_censored=True)
    current = Cycle(index=2, start=pd.Timestamp("2026-01-01"), end=None, n_points=100,
                    days=150.0, nwabt_start=355.0, nwabt_end=367.0,
                    rate_c_per_month=2.0, rate_ci=(1.5, 2.5), completed=False)
    assert remaining_by_lag(frame, [censored, current], current, censored, 1.0) is None


def test_local_rate_sees_the_plateau_the_whole_cycle_slope_hides():
    """Средний наклон по циклу и наклон на полке — разные числа, и это главное."""
    frame = _two_cycles()
    whole = local_rate(frame, cycle=1, since_day=0.0, until_day=200.0)
    plateau = local_rate(frame, cycle=1, since_day=60.0, until_day=200.0)
    assert whole > 0.5
    assert plateau == pytest.approx(0.0, abs=0.05)
    assert local_rate(frame, cycle=1, since_day=195.0, until_day=200.0) is None
