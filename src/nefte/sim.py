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

Допущения, которые надо назвать на защите:

* постоянная времени отклика качества 4 часа — типовая для реактора такого
  объёма, в пакете её нет;
* уставки удерживаются ровно так, как рекомендовано, мгновенно и без ошибки
  исполнения;
* взаимное влияние установок (АВТ ↔ гидроочистка) не моделируется: сырьё
  остаётся историческим.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from nefte.agents.schemas import Measurement, ProcessState, Source

# Постоянная времени отклика серы на изменение режима, часы. ДОПУЩЕНИЕ.
RESPONSE_TAU_HOURS = 4.0

# Насколько далеко уставке позволено уехать от исторического значения за весь
# прогон. ДОПУЩЕНИЕ и одновременно защита: без него ошибка в кинетике увела бы
# режим куда угодно, и эксперимент перестал бы что-либо значить.
MAX_DRIFT = {"T5": 8.0, "T6": 8.0, "T11": 8.0, "P13": 0.3, "F26": 40.0,
             "F15": 400.0, "P24": 0.3, "T55": 10.0}


@dataclass
class SimStep:
    """Одна точка замкнутого прогона."""
    ts: pd.Timestamp
    outcome: str
    sulfur_sim: float
    sulfur_hist: float | None
    offsets: dict[str, float]
    moved: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


class ClosedLoopSimulator:
    """Прогон системы с применением её собственных рекомендаций.

    Использование::

        sim = ClosedLoopSimulator(sb, system, surrogate)
        steps = sim.run(pd.date_range(lo, hi, freq="4h"))
    """

    def __init__(self, state_builder, system, surrogate,
                 tau_hours: float = RESPONSE_TAU_HOURS,
                 max_drift: dict[str, float] | None = None):
        self.sb = state_builder
        self.system = system
        self.surrogate = surrogate
        self.tau_hours = tau_hours
        self.max_drift = dict(max_drift or MAX_DRIFT)
        # накопленные смещения уставок относительно истории
        self.offsets: dict[str, float] = {}
        self.sulfur: float | None = None

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

    def _target_level(self, state: ProcessState) -> float | None:
        """Уровень серы, к которому процесс придёт при текущих уставках."""
        moves = {}
        for tag, delta in self.offsets.items():
            base = state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))
            if base is not None:
                moves[tag] = float(base + delta)
        value = self.surrogate(state, moves).get("product_sulfur_mgkg")
        return None if value is None or value != value else float(value)

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

            # лимит частоты воздействий здесь ОСМЫСЛЕН: прогон хронологический
            rec = self.system.run(observed)
            moved = {} if (rec.abstained or rec.action is None) else \
                self._accept(rec.action.deltas)

            hist = None
            if hist_sulfur is not None:
                window = hist_sulfur.loc[:ts]
                hist = float(window.iloc[-1]) if len(window) else None

            steps.append(SimStep(
                ts=ts,
                outcome=("отказ" if rec.abstained else
                         "меняем уставки" if moved else "держим режим"),
                sulfur_sim=float(self.sulfur),
                sulfur_hist=hist,
                offsets=dict(self.offsets),
                moved=moved,
                confidence=float(rec.confidence),
            ))
            previous_ts = ts
        return steps


def summarize(steps: list[SimStep], limit: float) -> dict:
    """Сводка замкнутого прогона: устойчивость контура и цена вмешательств."""
    if not steps:
        return {}
    frame = pd.DataFrame([{
        "ts": s.ts, "исход": s.outcome, "сера": s.sulfur_sim,
        "сера_история": s.sulfur_hist,
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
            # смена знака = система передумала; много смен = раскачка
            "смен направления": int((signs.diff().fillna(0) != 0).sum() - 1)
            if len(signs) else 0,
        }

    sim = frame["сера"].dropna()
    hist = frame["сера_история"].dropna()
    return {
        "шагов": len(steps),
        "исходы": frame["исход"].value_counts().to_dict(),
        "вмешательств": int((frame["исход"] == "меняем уставки").sum()),
        "сера_сим": {
            "среднее": round(float(sim.mean()), 3) if len(sim) else None,
            "доля выше предела": round(float((sim > limit).mean()), 3) if len(sim) else None,
        },
        "сера_история": {
            "среднее": round(float(hist.mean()), 3) if len(hist) else None,
            "доля выше предела": round(float((hist > limit).mean()), 3) if len(hist) else None,
        },
        "уставки": per_tag,
    }
