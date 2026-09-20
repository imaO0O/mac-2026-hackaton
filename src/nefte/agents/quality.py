"""Агент качества.

Отвечает на три вопроса: какое качество сейчас, каким оно станет через горизонт H,
и какова вероятность выйти за спецификацию. Здесь лежит БАЗОВАЯ версия (персистенция
+ нормальная неопределённость), чтобы сквозной цикл работал с первого дня.
Участник 1 заменяет ``predict`` на обученную модель, не трогая контракт.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from nefte.agents.schemas import (
    DataQuality,
    Measurement,
    ProcessState,
    QualityAssessment,
    Source,
)
from nefte.config import load_config

# Разброс между лабораторией и поточным анализатором на исторических парах:
# MAE 1.70 мг/кг, смещение -0.26. Используется как априорная σ базовой модели.
# Неопределённость нашего знания о ТЕКУЩЕМ Т95, °C. Это не точность формулы ВАК, а
# разброс того, насколько показатель успел уехать с момента отбора пробы: уровень
# мы берём из последнего лабораторного анализа.
#
# Число важное: запас до предела 360 °C обычно около 12 °C, то есть меньше двух
# сигм. Показывать Т95 как точно известную величину нельзя.
#
# Раньше здесь стояла ОДНА константа 6.64, измеренная на медианном шаге в сутки и
# применявшаяся при любом возрасте опорного анализа. Но анализ Т95 старше суток в
# трети моментов решения, и разброс за это время успевает подрасти
# (scripts/check_t95_sigma.py):
#
#     0-30 ч  6.79      80-130 ч  7.59
#    30-54 ч  7.16     130-200 ч  7.81
#    54-80 ч  7.53     200-400 ч  8.14
#
# Ряд возвращается к среднему, а не блуждает: случайное блуждание дало бы на том
# же плече рост в 4.47 раза, фактический рост — 1.20. Поэтому кривая пологая, и
# усложнение оправдано ровно этими двадцатью процентами, а не разами.
T95_SIGMA_C = 6.79            # запасное значение: возраст анализа неизвестен
T95_SIGMA_TABLE_PATH = "reports/t95_sigma.json"
# Поправка к вероятности нарушения Т95 (scripts/check_t95_calibration.py) и порог
# заметки в карточке при СЫРОЙ вероятности — до поправки он был 0.2.
T95_RISK_CALIBRATION_PATH = "reports/t95_risk_calibration.json"
T95_NOTE_RAW_RISK = 0.2

BASELINE_SIGMA_MGKG = 1.7

# Множители уверенности по источнику значения. ДОПУЩЕНИЕ: прямого сравнения в
# данных нет, числа выбраны консервативно и по смыслу. Лаборатория — контрольный
# факт (ТЗ), поточный прибор шумнее её (MAE 1.7 на исторических парах), а
# виртуальный анализатор работает вообще без измерения продукта.
SOURCE_CONFIDENCE = {Source.LIMS: 1.0, Source.PAK: 0.85, Source.VAK: 0.5,
                     Source.NONE: 0.0}
VAK_CONFIDENCE_FACTOR = SOURCE_CONFIDENCE[Source.VAK]


def confidence_parts(sigma: float, source: Source, age_hours: float | None,
                     stale_after_hours: float, usable: bool,
                     model_age_months: float | None = None,
                     shelf_life_months: float | None = None) -> dict[str, float]:
    """Из чего складывается уверенность. Возвращает множители, а не одно число.

    Раньше уверенность считалась только по σ прогноза — и оказывалась 0.95 почти
    всегда, потому что σ обученной модели стабильна. В карточке оператора это было
    украшением: цифра, которая не меняется, решению не помогает.

    Теперь учитывается ещё и то, ОТКУДА взято значение и насколько оно свежее.
    Для модели это принципиально: σ ничего не знает ни про устаревший анализ, ни
    про то, что прибор молчит, — а решение зависит от этого сильнее, чем от
    ширины интервала.
    """
    parts = {
        # разброс прогноза: пока он не хуже исторического разброса «лаборатория
        # против поточного прибора», уверенность не режем
        "σ прогноза": 1.0 / (1.0 + max(sigma - BASELINE_SIGMA_MGKG, 0.0)
                             / BASELINE_SIGMA_MGKG),
        "источник": SOURCE_CONFIDENCE.get(source, 0.0),
    }
    if age_hours is not None and stale_after_hours > 0:
        overdue = max(age_hours - stale_after_hours, 0.0)
        parts["свежесть"] = 1.0 / (1.0 + overdue / stale_after_hours)
    if not usable:
        # Непригодный срез — это не «чуть хуже», а «числу верить нельзя»: часть
        # тегов забракована детекторами, и прогноз считается на том, что осталось.
        #
        # Здесь стояло 0.5, и разрыв между пригодными и непригодными срезами был
        # пятикратным — но держался он не на этом множителе. Пока оперативным
        # источником был ряд из файла анализаторов, он в тех же окнах оказывался
        # заморожен, и уверенность резалась ещё и за источник. Стоило перевести
        # оперативный источник на Q21, который в этих окнах жив, как разрыв упал
        # до двукратного (tests/test_confidence_is_alive.py поймал это сразу).
        # Множитель должен работать сам по себе, а не за компанию с другим.
        parts["достоверность данных"] = 0.2
    # Возраст САМОЙ МОДЕЛИ. Дрейф измерен: смещение прогноза уезжает на
    # 0.083 мг/кг в месяц, потому что уровень серы падает быстрее, чем модель это
    # отслеживает (scripts/check_drift.py). Пока этот множитель не появился,
    # измеренный дрейф оставался бумажным: в отчёте он был, а на решение не влиял.
    #
    # Уверенность не режется, пока модель моложе срока годности, и дальше падает
    # обратно пропорционально просрочке — тем же законом, что и для устаревшего
    # анализа, чтобы карточка оператора читалась единообразно.
    if model_age_months is not None and shelf_life_months:
        overdue = max(model_age_months - shelf_life_months, 0.0)
        parts["возраст модели"] = 1.0 / (1.0 + overdue / shelf_life_months)
    return parts


def fuse_sulfur(state: ProcessState, cfg: dict | None = None) -> Measurement:
    """Выбирает значение серы по приоритету ЛИМС → ПАК → ВАК с учётом свежести.

    Лабораторный результат — контрольный факт (ТЗ). Но если он устарел, а ПАК
    исправен, оперативное значение полезнее устаревшего лабораторного: возвращаем
    ПАК и явно помечаем источник, чтобы это попало в объяснение оператору.
    """
    cfg = cfg or load_config()
    stale = cfg["quality"]["staleness_hours"]

    lims = state.quality.get("lims_sulfur_mgkg")
    # Оперативный анализатор: ряд из файла ПАК или тег Q21 телеметрии. Это разные
    # приборы (corr между ними 0.41), и какой из них ближе к лаборатории —
    # измерено, а не выбрано: scripts/check_analyzer_source.py, правило в PLAN.
    source = str(cfg["quality"].get("analyzer_source", "pak"))
    pak = state.quality.get(
        "q21_sulfur_ppm" if source == "q21" else "pak_sulfur_ppm")
    if pak is None and source == "q21":
        pak = state.quality.get("pak_sulfur_ppm")

    def fresh(m, hours) -> bool:
        return m is not None and m.value is not None and (m.age_hours or 0) <= hours

    if fresh(lims, stale["lims"]):
        return lims.model_copy(update={"source": Source.LIMS})
    # Свежесть ПАК проверяется СВОИМ порогом (час), а не порогом лаборатории.
    # Раньше проверялось только «не залип»: поточный анализатор, замолчавший
    # десять часов назад, не залипший — он просто молчит, и его последнее
    # значение подставлялось как оперативное.
    if fresh(pak, stale["pak"]) and not pak.is_frozen:
        comment = "ЛИМС устарел, взят поточный анализатор" if lims else "нет ЛИМС"
        return pak.model_copy(update={"source": Source.PAK, "comment": comment})
    if lims is not None and lims.value is not None:
        return lims.model_copy(update={"source": Source.LIMS, "is_stale": True,
                                       "comment": "ПАК недостоверен, взят устаревший ЛИМС"})
    # Лаборатории нет вовсе — устаревший ПАК всё же лучше пустоты, но только с
    # честной пометкой: приоритет ТЗ (ЛИМС → ПАК) уже соблюдён выше.
    if pak is not None and pak.value is not None and not pak.is_frozen:
        return pak.model_copy(update={"source": Source.PAK, "is_stale": True,
                                      "comment": "нет ЛИМС, показание ПАК устарело"})
    return Measurement(value=None, unit="мг/кг", source=Source.NONE,
                       comment="нет достоверного источника качества")


def spec_risk_normal(pred: float, sigma: float, limit: float) -> float:
    """P(значение > limit) при нормальной ошибке прогноза."""
    if sigma <= 0:
        return float(pred > limit)
    z = (limit - pred) / sigma
    return float(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


@lru_cache(maxsize=1)
def _t95_sigma_table() -> tuple[tuple[float, float], ...]:
    """Измеренная таблица «возраст анализа → разброс», из отчёта.

    Читается из файла, а не зашивается в код: число получено измерением и
    обязано пересчитываться вместе с остальными. Если отчёта нет — пустая
    таблица, и в дело идёт запасная константа.
    """
    path = Path(__file__).resolve().parents[3] / T95_SIGMA_TABLE_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    return tuple((float(b["age_to_h"]), float(b["sigma_c"]))
                 for b in raw.get("buckets", []))


def t95_sigma(age_hours: float | None) -> float:
    """Разброс знания о текущем Т95 при опорном анализе такого возраста.

    Ступенька, а не сглаживание: вёдра измерены, промежуточные значения — нет,
    и придумывать между ними кривую значило бы выдавать интерполяцию за
    измерение. За последним ведром σ не растёт — это и есть полка, ради которой
    мерялась форма.
    """
    table = _t95_sigma_table()
    if not table:
        return T95_SIGMA_C
    if age_hours is None:
        return table[-1][1]          # возраст неизвестен — берём худшее из измеренного
    for upper, sigma in table:
        if age_hours < upper:
            return sigma
    return table[-1][1]



@lru_cache(maxsize=1)
def _t95_risk_calibration() -> tuple[float, float, float] | None:
    """Поправка Платта к вероятности нарушения Т95: ``(a, b, частота на обучении)``.

    Зачем. Нормальное приближение по оценке Т95 и σ по возрасту анализа ЗАВЫШАЕТ
    вероятность нарушения в 2–4 раза на всех трёх периодах (заявлено 14.5 % при
    наблюдаемых 4.9 % на тесте) и проигрывает по Brier константе «всегда базовая
    частота». Упорядочивает оно при этом хорошо — ROC-AUC 0.78, — так что лечится
    монотонной поправкой, подобранной на обучающем периоде.

    Читается из отчёта и только если поправка там ПРИНЯТА по записанному заранее
    правилу (валидация: Brier лучше и наклон калибровки ближе к единице). Нет
    отчёта или поправка не принята — вероятность остаётся сырой, как раньше.
    """
    path = Path(__file__).resolve().parents[3] / T95_RISK_CALIBRATION_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not raw.get("принята"):
        return None
    try:
        return (float(raw["поправка"]["a"]), float(raw["поправка"]["b"]),
                float(raw.get("частота_на_обучении", float("nan"))))
    except (KeyError, TypeError, ValueError):
        return None


def _platt(p: float, a: float, b: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return 1.0 / (1.0 + math.exp(-(a + b * math.log(p / (1 - p)))))


def t95_violation_risk(t95: float, age_hours: float | None, limit: float) -> float:
    """Вероятность, что Т95 выше предела, — одна функция для агента и оптимизатора.

    Одна величина не может иметь два разных значения в двух карточках одного
    цикла: карточка качества и карточки вариантов берут её отсюда.
    """
    raw = spec_risk_normal(t95, t95_sigma(age_hours), limit)
    calibration = _t95_risk_calibration()
    if calibration is None:
        return raw
    a, b, _ = calibration
    return _platt(raw, a, b)


def t95_note_threshold() -> float:
    """Порог заметки о Т95 в той же шкале, что и показываемая вероятность.

    Поправка монотонна, поэтому порог 0.2 по сырой вероятности переводится через
    неё же: заметка появляется В ТЕХ ЖЕ МОМЕНТАХ, что и до поправки, только с
    честным числом. Иначе после поправки она не появлялась бы никогда: на тесте
    поправленная вероятность выше 0.2 не бывает, а сырой порог отмечал 46 анализов,
    из которых превышение было в 7 — втрое чаще обычного.
    """
    calibration = _t95_risk_calibration()
    if calibration is None:
        return T95_NOTE_RAW_RISK
    a, b, _ = calibration
    return _platt(T95_NOTE_RAW_RISK, a, b)


class QualityAgent:
    """Базовая реализация. Заменяемая часть — ``predict``."""

    def __init__(self, model=None, cfg: dict | None = None, horizon_hours: float = 2.0,
                 t95_fn=None):
        self.model = model            # обученная модель (участник 1); None → персистенция
        self.cfg = cfg or load_config()
        self.horizon_hours = getattr(model, "horizon_hours", horizon_hours)
        self.limit = self.cfg["spec"]["product_sulfur_mgkg"]["max"]
        # Т95 — ВТОРОЙ обязательный показатель качества по ответу организаторов, и
        # считать его должен агент качества, а не оптимизатор. Раньше оценка жила в
        # `optimizer.default_t95_estimator`: оптимизатор сам вычислял показатель
        # качества, то есть делал чужую работу, и в `QualityAssessment` этого
        # показателя не было вовсе — ни в логах прогонов, ни на дашборде.
        #
        # Функция, а не число: оптимизатору нужен Т95 для КАЖДОГО варианта уставок.
        from nefte.agents.optimizer import default_t95_estimator

        self.t95_fn = t95_fn if t95_fn is not None else default_t95_estimator()
        self.t95_limit = float(self.cfg["spec"]["t95_c"]["max"])
        self.shelf_life_months = float(
            self.cfg["quality"].get("model_shelf_life_months", 0.0)) or None
        # порог тревоги подобран на валидации вместе с моделью; без модели —
        # консервативное значение по умолчанию
        self.alarm_threshold = float(getattr(model, "alarm_threshold", 0.2))

    def _model_age_months(self, ts) -> float | None:
        """Сколько месяцев прошло с конца обучающего периода.

        Без обученной модели возраст не определён: персистенция не стареет.
        """
        if self.model is None:
            return None
        try:
            train_end = pd.Timestamp(self.cfg["split"]["train"][1])
        except (KeyError, TypeError, ValueError):
            return None
        days = (pd.Timestamp(ts) - train_end).total_seconds() / 86400.0
        return max(0.0, days / 30.44)

    def predict(self, state: ProcessState, current: Measurement) -> tuple[float, float]:
        """Прогноз серы на горизонте и σ. Возвращает ``(mean, sigma)``."""
        if self.model is not None:
            return self.model.predict_with_sigma(state)
        if current.value is None:
            return float("nan"), float("inf")
        sigma = BASELINE_SIGMA_MGKG
        if current.source is Source.PAK:
            sigma *= 1.3          # поточный анализатор шумнее лаборатории
        if current.is_stale:
            sigma *= 1.5 + (current.age_hours or 0) / 24.0
        return float(current.value), float(sigma)

    def assess(self, state: ProcessState) -> QualityAssessment:
        current = fuse_sulfur(state, self.cfg)
        mean, sigma = self.predict(state, current)

        notes: list[str] = []
        # Ни лаборатории, ни поточного анализатора. Если обученной модели нет,
        # прогнозировать нечем. Если есть — это ровно тот случай, для которого ТЗ
        # предусматривает третий приоритет: виртуальный анализатор. Называть его
        # надо своим именем и с пониженной уверенностью, а не выдавать за измерение.
        # ...и только если модель ДЕЙСТВИТЕЛЬНО что-то выдала. Раньше примечание
        # «значение получено виртуальным анализатором» появлялось и тогда, когда
        # модель вернула NaN (например, момент раньше начала истории): объяснение
        # утверждало то, чего не было.
        from_vak = (current.source is Source.NONE and self.model is not None
                    and mean == mean)
        if current.source is Source.NONE:
            notes.append(
                "Ни ЛИМС, ни ПАК недостоверны: значение получено виртуальным "
                "анализатором (ВАК, третий приоритет по ТЗ), уверенность снижена."
                if from_vak else
                "Нет достоверного источника по сере — прогноз недоступен.")
        if current.is_frozen:
            notes.append("Поточный анализатор признан замороженным.")
        if current.is_stale:
            notes.append(f"Лабораторное значение устарело: {current.age_hours:.1f} ч.")

        source = Source.VAK if from_vak else current.source
        if mean != mean:      # NaN
            return QualityAssessment(ts=state.ts, confidence=0.0, notes=notes,
                                     source=Source.NONE,
                                     horizon_hours=self.horizon_hours)

        # если у модели есть отдельный классификатор превышения — доверяем ему:
        # редкие события он оценивает лучше, чем нормальное приближение по σ
        risk = None
        if self.model is not None and hasattr(self.model, "risk_for_state"):
            risk = self.model.risk_for_state(state)
        if risk is None:
            risk = spec_risk_normal(mean, sigma, self.limit)
        # Порог свежести берётся у ТОГО источника, который реально выбран.
        # Раньше здесь всегда стоял порог лаборатории (24 ч), и показание
        # поточного анализатора десятичасовой давности считалось свежим, хотя его
        # собственный порог — час.
        staleness = self.cfg["quality"]["staleness_hours"]
        stale_after = float(staleness.get(
            "pak" if source is Source.PAK else "lims", staleness["lims"]))
        parts = confidence_parts(sigma, source, current.age_hours, stale_after,
                                 state.data_quality.usable,
                                 model_age_months=self._model_age_months(state.ts),
                                 shelf_life_months=self.shelf_life_months)
        confidence = max(0.05, min(0.95, float(np.prod(list(parts.values())))))
        weakest = min(parts, key=parts.get)
        if parts[weakest] < 0.9:
            notes.append(f"Уверенность {confidence:.2f}; сильнее всего её снижает "
                         f"«{weakest}» (множитель {parts[weakest]:.2f}).")

        predictions = {"product_sulfur_mgkg": mean}
        intervals = {"product_sulfur_mgkg": (mean - 1.96 * sigma, mean + 1.96 * sigma)}
        risks = {"product_sulfur_mgkg": risk}

        # Т95 текущего режима. Уровень — последний лабораторный анализ, поэтому и
        # неопределённость берётся ЕГО: показатель уезжает с момента отбора пробы,
        # и разброс этого ухода — и есть σ нашего знания о текущем Т95, а вовсе не
        # точность формулы. Величина зависит от ВОЗРАСТА анализа (6.8 °C у свежего,
        # 8.1 °C у старше двух суток), поэтому берётся функцией, а не константой.
        t95 = self.t95_fn(state, {}) if self.t95_fn else None
        if t95 is not None and t95 == t95:
            t95_meas = state.quality.get("lims_t95_c")
            t95_age = t95_meas.age_hours if t95_meas is not None else None
            sigma_t95 = t95_sigma(t95_age)
            predictions["product_t95_c"] = float(t95)
            intervals["product_t95_c"] = (t95 - 1.96 * sigma_t95,
                                          t95 + 1.96 * sigma_t95)
            risks["product_t95_c"] = t95_violation_risk(t95, t95_age, self.t95_limit)
            if risks["product_t95_c"] > t95_note_threshold():
                age_text = ("" if t95_age is None
                            else f", анализу {t95_age:.0f} ч")
                calibration = _t95_risk_calibration()
                usual = ("" if calibration is None or calibration[2] != calibration[2]
                         else f" при обычной частоте {calibration[2]:.1%}")
                notes.append(
                    f"Т95 {t95:.1f} °C при пределе {self.t95_limit:.0f}: "
                    f"вероятность выхода {risks['product_t95_c']:.0%}{usual}. Уровень "
                    f"взят из последнего анализа{age_text}, за это время он уезжает "
                    f"на {sigma_t95:.1f} °C.")

        return QualityAssessment(
            ts=state.ts,
            source=source,
            predictions=predictions,
            intervals=intervals,
            spec_risk=risks,
            horizon_hours=self.horizon_hours,
            confidence=confidence,
            notes=notes,
        )


def empty_data_quality() -> DataQuality:
    return DataQuality(missing_share=0.0, usable=True)
