"""Имитационная среда: замкнутый контур «система → уставки → процесс → система».

Зачем она нужна. Всё остальное в проекте — воспроизведение истории: система
смотрит на прошлое и говорит, что сделала бы. Рекомендация при этом никогда не
применяется, и на вопрос «а что будет, если оператор послушается» ответить нечем.
Между тем именно этот вопрос решает, годится ли система для автономного контура:
разомкнутая рекомендация может быть сколь угодно разумной поштучно и при этом
раскачивать режим, если её выполнять раз за разом.

**Честная рамка, и она принципиальна.** Отклик процесса здесь считает та же
кинетика гидрообессеривания, которой пользуется оптимизатор
(`models/kinetics.py`). Значит, среда НЕ может проверить, верна ли эта кинетика:
рассуждение замкнуто само на себя. Что она может — и ради чего сделана:

* показать, **устойчив ли контур**: не гоняет ли система уставки туда-обратно,
  не ползёт ли температура вверх бесконечно;
* измерить **суммарное воздействие** на оборудование: сколько градусов и сколько
  раз система просит подвинуть за период;
* сравнить с бездействием: **сколько времени продукт вне спецификации**, если
  режим не трогать вовсе, и если выполнять рекомендации.

Всё, что среда добавляет к истории, — это последствия НАШИХ действий. Сам процесс
(сырьё, состояние катализатора, работа АВТ) берётся из записанных данных, поэтому
дрейф и возмущения в эксперименте настоящие, а не выдуманные.

Что среда отслеживает, кроме серы: **Т95**, второй обязательный показатель. За
один цикл управление добавляет к ней меньше градуса, поштучно это незаметно — а за
прогон складывается. Считается и то, какой Т95 была бы БЕЗ наших воздействий:
сравнивать надо с историей, а не с пределом, потому что Т95 гуляет и без нас.

Допущения, которые надо назвать на защите:

* уставки удерживаются ровно так, как рекомендовано, мгновенно и без ошибки
  исполнения;
* взаимное влияние установок (АВТ ↔ гидроочистка) не моделируется: сырьё
  остаётся историческим;
* постоянная времени отклика качества — единственное, что перестало быть
  допущением: 4.6 ч измерены по данным (`scripts/find_delays.py`);
* на останове установки накопленные сдвиги уставок сбрасываются: продукт не идёт,
  а после пуска режим выставляют заново.

Что сделано, чтобы контур был замкнут на самом деле, а не на словах: модель
качества системы видит последствия действий через ``FeedbackModel``, а сравнение
«без вмешательства» идёт с той же имитацией без наших сдвигов, а не с
лабораторией (`docs/HARD_CHECKS.md` §10).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from nefte.agents.schemas import Measurement, ProcessState, Source

# Постоянная времени отклика серы на изменение режима, часы. Больше не допущение:
# 4.6 ч ИЗМЕРЕНО по данным — авторегрессия первого порядка на серу ПАК с
# управляющей уставкой на входе, только train, остановы и залипший анализатор
# выкинуты (scripts/find_delays.py, reports/delays.json). Само транспортное
# запаздывание там же вышло меньше 10 минут, то есть неразличимо на сетке данных,
# поэтому чистого dead time в модели отклика нет.
RESPONSE_TAU_HOURS = 4.6

# Насколько далеко уставке позволено уехать от исторического значения за весь
# прогон. ДОПУЩЕНИЕ и одновременно защита: без него ошибка в кинетике увела бы
# режим куда угодно, и эксперимент перестал бы что-либо значить.
MAX_DRIFT = {"T5": 8.0, "T6": 8.0, "T11": 8.0, "P13": 0.3, "F26": 40.0,
             "F14": 2.0, "F25": 1500.0, "T55": 10.0}


class FeedbackModel:
    """Модель качества, которая видит последствия действий системы в имитации.

    **Зачем, и это был дефект.** Обученная модель берёт признаки из матрицы по
    метке времени среза, то есть из ИСТОРИИ. Имитатор подменял в срезе измеренную
    серу смоделированной и считал контур замкнутым, но прогноз и риск, по которым
    оркестратор решает, действовать ли, от этой подмены не зависели вовсе:
    подставленные 2 и 15 мг/кг давали одинаковый прогноз, риск и решение. Система
    не видела, что её действия уже снизили серу, и продолжала давить до предела
    дрейфа — отсюда «сера 2.6 при пределе 10» в прежних прогонах.

    Здесь к прогнозу добавляется сдвиг: насколько смоделированная сера отличается
    от уровня, который был бы без наших воздействий. Риск классификатора сдвигается
    так же, через нормальное приближение с σ модели: ``Φ(Φ⁻¹(p) + сдвиг/σ)``. При
    нулевом сдвиге модель отвечает ровно как без обёртки.
    """

    def __init__(self, model):
        self._model = model
        self.shift = 0.0

    def __getattr__(self, name):
        return getattr(self._model, name)

    def predict_with_sigma(self, state):
        mean, sigma = self._model.predict_with_sigma(state)
        return float(mean) + self.shift, sigma

    def risk_for_state(self, state):
        risk = self._model.risk_for_state(state)
        if risk is None or risk != risk or not self.shift:
            return risk
        _, sigma = self._model.predict_with_sigma(state)
        if not sigma or sigma != sigma or math.isinf(sigma):
            return risk
        from scipy.stats import norm

        p = min(max(float(risk), 1e-9), 1.0 - 1e-9)
        return float(norm.cdf(norm.ppf(p) + self.shift / float(sigma)))


def _shifted_surrogate(fn, feedback: FeedbackModel):
    """Суррогат, к прогнозу серы которого добавлен сдвиг имитации."""
    def _fn(state, moves):
        out = dict(fn(state, moves))
        value = out.get("product_sulfur_mgkg")
        if value is not None and value == value:
            out["product_sulfur_mgkg"] = float(value) + feedback.shift
        return out
    _fn.kind = getattr(fn, "kind", None)
    return _fn


@dataclass
class SimStep:
    """Одна точка замкнутого прогона."""
    ts: pd.Timestamp
    outcome: str
    sulfur_sim: float
    sulfur_hist: float | None
    offsets: dict[str, float]
    moved: dict[str, float] = field(default_factory=dict)
    # исход — то, что РЕШИЛА система; applied — то, что удалось применить. Они
    # расходятся, когда уставка упёрлась в потолок дрейфа, и путать их нельзя: в
    # остальных прогонах «держим режим» означает «система не захотела», а не
    # «не смогла».
    applied: bool = False
    # Т95 при накопленном смещении уставок. Отдельно от серы, потому что именно
    # здесь видно, не чинит ли контур серу за счёт другого обязательного показателя:
    # за один цикл прибавка к Т95 меньше градуса и незаметна, а за прогон копится.
    t95_sim: float | None = None
    # вид действия: "возврат" — шаг назад к базовому режиму, "" — всё остальное
    kind: str = ""
    # Т95 БЕЗ наших воздействий — то же, что «сера_история» для серы. Без этой
    # опорной точки доля времени выше предела ничего не говорит: Т95 гуляет сама
    # по себе, и приписывать её выход за предел нашему управлению нечестно.
    t95_hist: float | None = None
    confidence: float = 0.0
    # Сера процесса в ТОЙ ЖЕ имитации, но без наших воздействий. Сравнивать
    # смоделированную серу надо с ней, а не с лабораторией: процесс в имитации идёт
    # от уровня модели, а он глаже лабораторного ряда и реже выходит за предел.
    sulfur_untouched: float | None = None
    # на этом шаге установка стояла, и накопленные сдвиги уставок сброшены
    reset: bool = False


class ClosedLoopSimulator:
    """Прогон системы с применением её собственных рекомендаций.

    Использование::

        sim = ClosedLoopSimulator(sb, system, surrogate)
        steps = sim.run(pd.date_range(lo, hi, freq="4h"))
    """

    def __init__(self, state_builder, system, surrogate,
                 tau_hours: float = RESPONSE_TAU_HOURS,
                 max_drift: dict[str, float] | None = None,
                 t95_fn=None):
        from nefte.agents.optimizer import default_t95_estimator

        self.sb = state_builder
        self.system = system
        self.surrogate = surrogate
        self.t95_fn = t95_fn if t95_fn is not None else default_t95_estimator()
        self.tau_hours = tau_hours
        self.max_drift = dict(max_drift or MAX_DRIFT)
        # накопленные смещения уставок относительно истории
        self.offsets: dict[str, float] = {}
        self.sulfur: float | None = None
        # Процесс отвечает по ИСХОДНОМУ суррогату (он уже сохранён выше), а система
        # получает модель и суррогат, видящие последствия своих действий.
        self.feedback = self._close_quality_loop()

    def _unit_down(self, state: ProcessState) -> bool:
        """Стоит ли установка — по агенту надёжности системы, если он есть."""
        check = getattr(getattr(self.system, "reliability", None), "is_unit_down", None)
        return bool(check(state)) if callable(check) else False

    def _close_quality_loop(self) -> FeedbackModel | None:
        """Подключает к системе модель качества со сдвигом имитации. См. FeedbackModel."""
        from nefte.models.kinetics import make_kinetic_surrogate

        quality = getattr(self.system, "quality", None)
        model = getattr(quality, "model", None)
        if model is None:
            # персистенция читает измерения из среза, и подмены серы ей хватает
            return None
        feedback = FeedbackModel(model)
        quality.model = feedback
        optimizer = getattr(self.system, "optimizer", None)
        current = getattr(optimizer, "surrogate", None)
        if current is None:
            return feedback
        optimizer.surrogate = self._with_feedback(current, feedback)
        # пессимистичная кинетика оптимизатора обязана видеть тот же сдвиг, иначе
        # гарантия проверялась бы по уровню, которого система уже не наблюдает
        robust = getattr(optimizer, "robust_surrogate", None)
        if robust is not None:
            optimizer.robust_surrogate = self._with_feedback(robust, feedback)
        return feedback

    @staticmethod
    def _with_feedback(fn, feedback: "FeedbackModel"):
        """Тот же суррогат, но поверх модели, видящей последствия действий."""
        from nefte.models.kinetics import make_kinetic_surrogate

        if getattr(fn, "kind", None) == "kinetic" and hasattr(fn, "order"):
            base = fn.base_surrogate
            return make_kinetic_surrogate(
                feedback,
                base_surrogate=None if base is None else _shifted_surrogate(base, feedback),
                strength=fn.strength, tag_prefix=fn.tag_prefix, order=fn.order)
        return _shifted_surrogate(fn, feedback)

    # ------------------------------------------------------------------ #
    def _apply_offsets(self, state: ProcessState) -> ProcessState:
        """Срез, каким его увидит система, если наши уставки удерживаются."""
        if not self.offsets:
            return state
        shifted = state.model_copy(deep=True)
        for tag, delta in self.offsets.items():
            for store in (shifted.telemetry_ht, shifted.telemetry_avt):
                if store.get(tag) is not None:
                    store[tag] = float(store[tag] + delta)
                    break
        return shifted

    def _with_simulated_quality(self, state: ProcessState, value: float) -> ProcessState:
        """Подменяет измеренную серу смоделированной.

        Иначе система читала бы исторический анализ, снятый при историческом
        режиме, и замкнутого контура не получилось бы.
        """
        out = state.model_copy(deep=True)
        for key in ("lims_sulfur_mgkg", "pak_sulfur_ppm"):
            if key in out.quality:
                current = out.quality[key]
                out.quality[key] = Measurement(
                    value=float(value), unit=current.unit, source=current.source,
                    age_hours=current.age_hours, is_stale=current.is_stale,
                    is_frozen=current.is_frozen,
                    comment="значение имитационной среды")
        if "pak_sulfur_ppm" not in out.quality and "lims_sulfur_mgkg" not in out.quality:
            out.quality["pak_sulfur_ppm"] = Measurement(
                value=float(value), unit="мг/кг", source=Source.PAK, age_hours=0.0,
                comment="значение имитационной среды")
        return out

    def _with_simulated_t95(self, state: ProcessState,
                            value: float | None) -> ProcessState:
        """Подменяет лабораторную Т95 смоделированной. См. пояснение в run()."""
        if value is None or value != value or "lims_t95_c" not in state.quality:
            return state
        out = state.model_copy(deep=True)
        current = out.quality["lims_t95_c"]
        out.quality["lims_t95_c"] = Measurement(
            value=float(value), unit=current.unit, source=current.source,
            age_hours=current.age_hours, is_stale=current.is_stale,
            comment="значение имитационной среды")
        return out

    def _target_level(self, state: ProcessState) -> float | None:
        """Уровень серы, к которому процесс придёт при текущих уставках."""
        moves = {}
        for tag, delta in self.offsets.items():
            base = state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))
            if base is not None:
                moves[tag] = float(base + delta)
        value = self.surrogate(state, moves).get("product_sulfur_mgkg")
        return None if value is None or value != value else float(value)

    def _t95_level(self, base: ProcessState) -> float | None:
        """Т95 при НАКОПЛЕННОМ смещении уставок относительно истории.

        Считается на исходном (несмещённом) срезе: смещение подставляется как
        воздействие, и оценка получается относительно исторического режима, а не
        относительно уже уехавшего. Иначе накопленный уход был бы не виден —
        каждый шаг мерился бы от предыдущего и выглядел бы безобидным.
        """
        if not self.offsets:
            return self.t95_fn(base, {})
        moves = {}
        for tag, delta in self.offsets.items():
            value = base.telemetry_ht.get(tag, base.telemetry_avt.get(tag))
            if value is not None:
                moves[tag] = float(value + delta)
        return self.t95_fn(base, moves)

    def _accept(self, deltas: dict[str, float]) -> dict[str, float]:
        """Принимает рекомендованные изменения, не выпуская режим за предел дрейфа."""
        moved = {}
        for tag, delta in deltas.items():
            if abs(delta) < 1e-6:
                continue
            limit = self.max_drift.get(tag)
            new = self.offsets.get(tag, 0.0) + float(delta)
            if limit is not None:
                new = max(-limit, min(limit, new))
            applied = new - self.offsets.get(tag, 0.0)
            if abs(applied) > 1e-9:
                self.offsets[tag] = new
                moved[tag] = applied
        return moved

    # ------------------------------------------------------------------ #
    def run(self, stamps, hist_sulfur: pd.Series | None = None) -> list[SimStep]:
        """Замкнутый прогон по моментам ``stamps``."""
        steps: list[SimStep] = []
        previous_ts = None

        for ts in stamps:
            ts = pd.Timestamp(ts)
            base = self.sb.build(ts)
            # Останов обнуляет наши сдвиги уставок. Продукт не идёт, а после пуска
            # режим выставляют заново. Без сброса имитация применяла сдвиг −2 °C,
            # сделанный до останова, к реактору при 40 °C, и Аррениус при такой
            # температуре раздувал его в четырёхкратный рост серы: на окне с
            # остановом 15 «превышений» пришлись на шаги, где система отказывалась.
            reset = bool(self.offsets) and self._unit_down(base)
            if reset:
                if hasattr(self.system, "record_applied"):
                    self.system.record_applied({t: -v for t, v in self.offsets.items()})
                self.offsets = {}
            shifted = self._apply_offsets(base)

            target = self._target_level(shifted)
            if self.sulfur is None:
                self.sulfur = target if target is not None else float("nan")
            elif target is not None and previous_ts is not None:
                # апериодическое звено первого порядка: качество идёт к новому
                # уровню не мгновенно, а с постоянной времени tau
                dt = (ts - previous_ts).total_seconds() / 3600.0
                weight = 1.0 - math.exp(-dt / self.tau_hours) if dt > 0 else 0.0
                self.sulfur += (target - self.sulfur) * weight

            observed = shifted if self.sulfur != self.sulfur else \
                self._with_simulated_quality(shifted, self.sulfur)

            # Т95 подменяем ровно по той же причине, что и серу: лабораторный
            # анализ в истории снят при ИСТОРИЧЕСКОМ режиме и про наши накопленные
            # смещения ничего не знает. Без подмены оптимизатор считал текущую Т95
            # равной историческому анализу, то есть не видел собственного ухода и
            # не мог его остановить. В реальной работе анализ приходит с настоящего
            # продукта и учитывает всё, что сделал оператор, — здесь это надо
            # воспроизвести руками.
            t95_now = self._t95_level(base)
            observed = self._with_simulated_t95(observed, t95_now)

            # сдвиг для модели системы: смоделированная сера минус уровень без наших
            # воздействий в этот момент
            untouched = self.surrogate(shifted, {}).get("product_sulfur_mgkg")
            if self.feedback is not None:
                self.feedback.shift = (
                    float(self.sulfur - untouched)
                    if (self.sulfur == self.sulfur and untouched is not None
                        and untouched == untouched) else 0.0)

            # лимит частоты воздействий здесь ОСМЫСЛЕН: прогон хронологический
            rec = self.system.run(observed)
            moved = {} if (rec.abstained or rec.action is None) else \
                self._accept(rec.action.deltas)
            # оркестратор узнаёт о применённом — только так он может потом вернуть
            # режим к базовому; выданная, но не применённая рекомендация не в счёт
            if moved and hasattr(self.system, "record_applied"):
                self.system.record_applied(moved)

            hist = None
            if hist_sulfur is not None:
                window = hist_sulfur.loc[:ts]
                hist = float(window.iloc[-1]) if len(window) else None

            steps.append(SimStep(
                ts=ts,
                outcome=rec.outcome(),
                applied=bool(moved),
                sulfur_sim=float(self.sulfur),
                sulfur_hist=hist,
                t95_sim=t95_now,
                t95_hist=self.t95_fn(base, {}),
                offsets=dict(self.offsets),
                moved=moved,
                confidence=float(rec.confidence),
                sulfur_untouched=(None if untouched is None or untouched != untouched
                                  else float(untouched)),
                reset=reset,
                kind=("возврат" if (rec.action is not None and not rec.abstained
                                    and rec.action.id == "return_to_base") else ""),
            ))
            previous_ts = ts
        return steps


def summarize(steps: list[SimStep], limit: float,
              t95_limit: float | None = None) -> dict:
    """Сводка замкнутого прогона: устойчивость контура и цена вмешательств."""
    if not steps:
        return {}
    frame = pd.DataFrame([{
        "ts": s.ts, "исход": s.outcome, "сера": s.sulfur_sim,
        "сера_история": s.sulfur_hist,
        "сера_без_нас": s.sulfur_untouched,
        "Т95": s.t95_sim,
        "Т95_история": s.t95_hist,
        **{f"смещение_{k}": v for k, v in s.offsets.items()},
        **{f"шаг_{k}": v for k, v in s.moved.items()},
    } for s in steps])

    moves = [s.moved for s in steps]
    tags = sorted({t for m in moves for t in m})
    per_tag = {}
    for tag in tags:
        series = pd.Series([m.get(tag, 0.0) for m in moves])
        signs = series[series.abs() > 1e-9].map(lambda v: 1 if v > 0 else -1)
        per_tag[tag] = {
            "суммарно, ед.": round(float(series.abs().sum()), 2),
            "итоговое смещение": round(float(steps[-1].offsets.get(tag, 0.0)), 2),
            # смена знака = система передумала; много смен = раскачка.
            # fillna(0) уже исключает первое воздействие (сравнивать его не с чем),
            # поэтому вычитать единицу не нужно: с ней одно-единственное движение
            # давало «−1 смена направления».
            "смен направления": (int((signs.diff().fillna(0) != 0).sum())
                                 if len(signs) else 0),
        }

    sim = frame["сера"].dropna()
    hist = frame["сера_история"].dropna()
    return {
        "шагов": len(steps),
        "исходы": frame["исход"].value_counts().to_dict(),
        "вмешательств": int((frame["исход"] == "меняем уставки").sum()),
        "из них возвратов к базе": int(sum(1 for item in steps if item.kind == "возврат")),
        # Раскачку возврат показывает не «сменами направления» — возврат разворачивает
        # сдвиг по определению, — а качелями: режим вернули, и в течение суток по
        # тому же тегу снова сдвинули в прежнюю сторону.
        "качели после возврата": _swings_after_return(steps),
        "применено": int(sum(1 for item in steps if item.applied)),
        "сбросов сдвигов на останове": int(sum(1 for item in steps if item.reset)),
        "сера_сим": {
            "среднее": round(float(sim.mean()), 3) if len(sim) else None,
            "доля выше предела": round(float((sim > limit).mean()), 3) if len(sim) else None,
        },
        "сера_история": {
            "среднее": round(float(hist.mean()), 3) if len(hist) else None,
            "доля выше предела": round(float((hist > limit).mean()), 3) if len(hist) else None,
        },
        # Правильная точка сравнения: та же имитация без наших воздействий, на тех же
        # шагах. «Лаборатория без вмешательства» — другой ряд: 6 % превышений против
        # 0 % в имитации получались и тогда, когда система ничего не делала.
        "сера_без_вмешательства_сим": _paired_sulfur(frame, limit),
        "уставки": per_tag,
        # Второй обязательный показатель. Смотрим на него именно здесь: за один
        # цикл контур добавляет к Т95 меньше градуса, и поштучно это незаметно, а
        # за прогон складывается в реальный уход к пределу.
        #
        # Обе доли — по ОДНИМ И ТЕМ ЖЕ шагам. Раньше каждая считалась по своим
        # непустым строкам, а оценка Т95 со смещениями была пустой в 71 шаге из 175
        # (оценка не переносила пустые F9/F2), и сводка сравнивала 26 % по 104 шагам
        # с 12 % по 175. Из этой пары вырос вывод «наше управление держит Т95 выше
        # предела вдвое чаще», и он был артефактом.
        "Т95": _t95_summary(frame.dropna(subset=["Т95", "Т95_история"])["Т95"], t95_limit),
        "Т95_история": _t95_summary(
            frame.dropna(subset=["Т95", "Т95_история"])["Т95_история"], t95_limit),
        "Т95_шагов_без_оценки": int(frame["Т95"].isna().sum()),
        # Ответ на единственный вопрос, ради которого Т95 здесь считается:
        # НАШИ действия ухудшили её или улучшили?
        "Т95_наш_вклад": _t95_contribution(frame, t95_limit),
    }


def _paired_sulfur(frame: pd.DataFrame, limit: float) -> dict:
    """Сера с нашими воздействиями и без них — на одних и тех же шагах имитации."""
    pair = frame.dropna(subset=["сера", "сера_без_нас"])
    if pair.empty:
        return {}
    ours, base = pair["сера"], pair["сера_без_нас"]
    return {
        "шагов": int(len(pair)),
        "среднее без нас": round(float(base.mean()), 3),
        "среднее с нами": round(float(ours.mean()), 3),
        "выше предела без нас, шагов": int((base > limit).sum()),
        "выше предела с нами, шагов": int((ours > limit).sum()),
        "средний наш сдвиг": round(float((ours - base).mean()), 3),
    }


def _swings_after_return(steps: list[SimStep], hours: float = 24.0) -> int:
    """Сколько возвратов отменено действием ради качества в течение ``hours``."""
    count = 0
    for i, item in enumerate(steps):
        if item.kind != "возврат":
            continue
        for later in steps[i + 1:]:
            if (later.ts - item.ts).total_seconds() / 3600 > hours:
                break
            if later.kind == "возврат":
                continue
            if any(tag in later.moved and later.moved[tag] * delta < 0
                   for tag, delta in item.moved.items()):
                count += 1
                break
    return count


def _t95_contribution(frame: pd.DataFrame, limit: float | None = None) -> dict | None:
    """Разница между Т95 с нашим управлением и Т95 без него.

    Считается по совпадающим моментам: только так видно, что именно добавили мы, а
    что было в истории и без нас.

    «Перевели через предел» считается отдельно и с величиной сдвига. Лабораторный
    Т95 записан целыми градусами, и значение ровно 360.0 превышением не является, а
    360.009 — формально является. Доля «выше предела» от такого сдвига меняется, а
    продукт — нет, поэтому рядом стоит, на сколько именно мы перевели.
    """
    both = frame[["Т95", "Т95_история"]].dropna()
    if not len(both):
        return None
    delta = both["Т95"] - both["Т95_история"]
    out = {
        "средний сдвиг": round(float(delta.mean()), 2),
        "худший сдвиг": round(float(delta.max()), 2),
        "доля моментов, где мы подняли Т95": round(float((delta > 1e-9).mean()), 3),
    }
    if limit is not None:
        crossed = (both["Т95"] > limit) & (both["Т95_история"] <= limit)
        out["перевели через предел, шагов"] = int(crossed.sum())
        out["наибольший сдвиг при переводе, °C"] = (
            round(float(delta[crossed].max()), 3) if crossed.any() else 0.0)
    return out


def _t95_summary(series: pd.Series, limit: float | None) -> dict | None:
    """Куда ушёл Т95 за прогон и подошёл ли он к пределу."""
    if not len(series):
        return None
    out = {
        "начало": round(float(series.iloc[0]), 2),
        "конец": round(float(series.iloc[-1]), 2),
        "максимум": round(float(series.max()), 2),
        "уход за прогон": round(float(series.iloc[-1] - series.iloc[0]), 2),
    }
    if limit is not None:
        out["предел"] = limit
        out["доля выше предела"] = round(float((series > limit).mean()), 3)
        out["минимальный запас"] = round(float(limit - series.max()), 2)
    return out
