"""Кинетический суррогат «режим → сера»: гибрид модели и физики.

Зачем понадобился. Обученная модель — хороший виртуальный анализатор (MAE 1.26
против 1.48 у поточного), но на изменение уставок она почти не реагирует: сдвиг
реакторной температуры на 2 °C меняет прогноз на сотые доли мг/кг, тогда как в
реальной гидроочистке это десятые. Причина не в модели, а в данных: связь
«температура → сера» в истории слабая (|r| ≈ 0.17), потому что установка и так
работает в узком коридоре режимов. Модель не может выучить отклик, которого в
данных нет, а оптимизатору сравнивать сценарии без отклика бессмысленно.

Решение — стандартная промышленная схема «модель уровня + физика приращения»:

* **уровень** берём у обученной модели: где мы находимся сейчас, она знает лучше;
* **приращение** считаем по кинетике гидрообессеривания первого порядка,
  откалиброванной на текущем измерении.

Как это работает. Для реакции первого порядка ``S_out = S_in · exp(−k/LHSV)``.
Текущие значения серы сырья и продукта известны, значит текущий комплекс
``τ = k/LHSV = ln(S_in / S_out)`` вычисляется без подгонки. При изменении режима
он пересчитывается по Аррениусу и по времени контакта, а сера продукта берётся
обратно из того же уравнения.

ДОПУЩЕНИЯ, которые надо назвать на защите:
* энергия активации 100 кДж/моль — типовая для ГДС, в пакете её нет;
* реакция считается псевдопервого порядка по сере;
* объём катализатора неизвестен, поэтому время контакта входит обратно
  пропорционально расходу сырья;
* влияние водорода учитывается степенной поправкой с показателем 0.5.

Чем этот подход честен: физика применяется ТОЛЬКО к приращению относительно
текущей точки, а сам уровень остаётся измеренным. Если сегодняшнее состояние
модель оценила верно, то и направление, и порядок величины отклика будут
разумными, даже когда в истории такого манёвра не было.
"""
from __future__ import annotations

import math

from nefte.agents.schemas import ProcessState
from nefte.models.regime import (
    ACTIVATION_ENERGY_KJ,
    FEED,
    GAS_CONSTANT_KJ,
    MAKEUP_H2,
    PRESSURE,
    REACTOR_TEMPS,
)

# Показатель степени при парциальном давлении водорода в кинетике ГДС.
H2_ORDER = 0.5
# Насколько сера сырья может быть выше продукта, чтобы задача имела смысл.
MIN_CONVERSION_RATIO = 1.5
DEFAULT_FEED_SULFUR_MGKG = 9000.0


def _wabt(values: dict[str, float | None]) -> float | None:
    temps = [values.get(t) for t in REACTOR_TEMPS if values.get(t) is not None]
    return sum(temps) / len(temps) if temps else None


def arrhenius_factor(t_from_c: float, t_to_c: float,
                     activation_kj: float = ACTIVATION_ENERGY_KJ) -> float:
    """Во сколько раз меняется константа скорости при изменении температуры."""
    t1, t2 = t_from_c + 273.15, t_to_c + 273.15
    if t1 <= 0 or t2 <= 0:
        return 1.0
    return math.exp(activation_kj / GAS_CONSTANT_KJ * (1.0 / t1 - 1.0 / t2))


def make_kinetic_surrogate(model, base_surrogate=None, strength: float = 1.0,
                           tag_prefix: str = "ht_"):
    """Суррогат «режим → качество»: уровень от модели, приращение от кинетики.

    Parameters
    ----------
    model : обученная ``SulfurModel`` — даёт текущий уровень серы.
    base_surrogate : суррогат для случая, когда кинетику применить нельзя
        (нет серы сырья, нет температур). По умолчанию — прогноз модели как есть.
    strength : 0…1, доля кинетического приращения. 1.0 — чистая физика,
        0.0 — поведение прежнего суррогата. Позволяет показать на защите, как
        решение зависит от силы допущения.
    """

    def _model_level(state: ProcessState) -> float:
        mean, _ = model.predict_with_sigma(state)
        return float(mean)

    def _fn(state: ProcessState, moves: dict[str, float]) -> dict[str, float]:
        level = _model_level(state)
        if level != level:                       # NaN
            return {"product_sulfur_mgkg": float("nan")}

        feed_meas = state.quality.get("lims_feed_sulfur_mgkg")
        feed_sulfur = (feed_meas.value if feed_meas and feed_meas.value
                       else DEFAULT_FEED_SULFUR_MGKG)

        current = dict(state.telemetry_ht)
        new = {**current, **moves}

        t_now, t_new = _wabt(current), _wabt(new)
        if (t_now is None or t_new is None or feed_sulfur is None
                or feed_sulfur < level * MIN_CONVERSION_RATIO or level <= 0):
            if base_surrogate is not None:
                return base_surrogate(state, moves)
            return {"product_sulfur_mgkg": level}

        # текущий кинетический комплекс, вычисленный без подгонки
        tau = math.log(feed_sulfur / level)

        factor = arrhenius_factor(t_now, t_new)

        # время контакта обратно пропорционально расходу сырья
        f_now, f_new = current.get(FEED), new.get(FEED)
        if f_now and f_new and f_new > 0:
            factor *= f_now / f_new

        # водород: давление в реакторе и подпитка ВСГ
        for tag in (PRESSURE, MAKEUP_H2):
            a, b = current.get(tag), new.get(tag)
            if a and b and a > 0 and b > 0:
                factor *= (b / a) ** H2_ORDER

        tau_new = tau * (1.0 + strength * (factor - 1.0))
        tau_new = max(tau_new, 0.0)
        value = feed_sulfur * math.exp(-tau_new)
        return {"product_sulfur_mgkg": float(value)}

    # Метка для оркестратора: приращение эффекта посчитано по физике, а не
    # измерено. Оператору это надо сказать, иначе «−3.4 мг/кг» читается как факт.
    _fn.kind = "kinetic"
    return _fn
