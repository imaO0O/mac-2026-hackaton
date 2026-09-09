"""Агент надёжности.

Прямой разметки отказов, наработки катализатора и предельных уставок в пакете НЕТ.
Поэтому оцениваем тяжесть режима прокси-метриками и объявляем это допущением
(ТЗ, п.3: «при отсутствии прямой разметки допускаются прокси-метрики с явным
описанием допущений»).

Прокси, которые используем:
* WABT реакторного блока (T5, T6, T11) — чем выше, тем жёстче режим и быстрее
  дезактивируется катализатор;
* перепад давления на Р-202 (W10) — рост означает закоксовывание/забивку слоя;
* температура на выходе печи П-3 (T55) — тепловая напряжённость печи;
* скорость изменения режима за час — резкие броски сами по себе фактор риска;
* наработка от начала цикла — возраст катализатора, восстановленный по длительным
  провалам расхода сырья (разметки остановов в пакете нет);
* нетипичность режима — многомерный детектор (`models/anomaly.py`), который ловит
  недопустимые СОЧЕТАНИЯ нормальных по отдельности значений.

Нормировка считается ТОЛЬКО по обучающему периоду. Если брать всю историю,
severity в 2026 году будет нормирован по данным 2026 года, то есть по будущему.
Пороги risk_class тоже берутся из распределения severity на обучающем периоде,
а не назначаются вручную.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nefte.agents.schemas import ProcessState, ReliabilityAssessment
from nefte.config import load_config
from nefte.models.anomaly import RegimeAnomalyDetector
from nefte.models.regime import FEED, RECYCLE_GAS, hours_since_outage


@dataclass
class SeverityNorms:
    """Нормировка прокси-факторов по историческим квантилям.

    ``fit`` считает p05/p95 по обучающему периоду — это модельный диапазон,
    а не промышленный предел. Так требует ТЗ (правило границ).
    """
    bounds: dict[str, tuple[float, float]]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str],
            q: tuple[float, float] = (0.05, 0.95),
            train: tuple[str, str] | None = None) -> "SeverityNorms":
        """``train`` — границы обучающего периода; вне его данные не используются."""
        if train is not None:
            df = df.loc[train[0]:train[1]]
        bounds = {}
        for c in columns:
            if c in df.columns:
                lo, hi = df[c].quantile(q[0]), df[c].quantile(q[1])
                bounds[c] = (float(lo), float(hi))
        return cls(bounds=bounds)

    def normalize(self, name: str, value: float | None) -> float | None:
        if value is None or value != value or name not in self.bounds:
            return None
        lo, hi = self.bounds[name]
        if hi <= lo:
            return None
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.5))

    def normalize_series(self, name: str, series: pd.Series) -> pd.Series | None:
        """Та же нормировка, но по всей истории — для калибровки порогов."""
        if name not in self.bounds:
            return None
        lo, hi = self.bounds[name]
        if hi <= lo:
            return None
        return ((series - lo) / (hi - lo)).clip(0.0, 1.5)


class ReliabilityAgent:
    """Оценивает тяжесть режима и сужает диапазоны для оптимизатора."""

    # Веса подобраны по смыслу, а не по данным: разметки отказов нет, обучать
    # веса не на чем. WABT — главный фактор жёсткости, перепад давления — прямой
    # признак состояния слоя, аномальность и скорость изменения — про безопасность
    # манёвра, наработка и печь — медленные фоновые факторы.
    WEIGHTS = {"wabt": 0.30, "dp_r202": 0.20, "anomaly": 0.15, "ramp": 0.15,
               "catalyst": 0.10, "furnace": 0.10}
    REACTOR_TEMPS = ["T5", "T6", "T11"]
    DP_TAG = "W10"
    FURNACE_TAG = "T55"
    # Пороги risk_class по умолчанию, если агент собран без истории.
    DEFAULT_THRESHOLDS = (0.5, 0.8)
    # Возраст катализатора сбрасывается только на ДЛИТЕЛЬНОМ останове: шестичасовой
    # простой катализатор не омолаживает. 48 часов — допущение, разметки замен нет.
    CATALYST_RESET_HOURS = 48.0
    # Признаки того, что установка стоит: нет сырья И реактор холодный.
    # В истории таких периодов три, реактор в них 7-43 °C вместо 360.
    DOWN_FEED_LEVEL = 0.1          # доля от медианного расхода сырья
    DOWN_TEMP_C = 150.0

    def __init__(self, norms: SeverityNorms | None = None,
                 ramp_series: pd.Series | None = None,
                 run_hours: pd.Series | None = None,
                 run_hours_scale: float | None = None,
                 detector: RegimeAnomalyDetector | None = None,
                 thresholds: tuple[float, float] | None = None,
                 feed_median: float | None = None,
                 down_series: pd.Series | None = None):
        self.norms = norms or SeverityNorms(bounds={})
        # нормированная скорость изменения режима, посчитанная по истории
        self.ramp_series = ramp_series
        # наработка от последнего останова и её масштаб (p95 обучающего периода)
        self.run_hours = run_hours
        self.run_hours_scale = run_hours_scale
        self.detector = detector or RegimeAnomalyDetector()
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        # медианный расход сырья — опорное значение для распознавания останова
        self.feed_median = feed_median
        # маска «установка стоит», посчитанная по СЫРОЙ телеметрии: в очищенном
        # срезе расход сырья на останове замаскирован, и признак не виден
        self.down_series = down_series

    # ------------------------------------------------------------------ #
    @classmethod
    def from_history(cls, avt: pd.DataFrame, ht: pd.DataFrame,
                     cfg: dict | None = None,
                     raw_ht: pd.DataFrame | None = None) -> "ReliabilityAgent":
        """Собирает агента по истории: нормировка, аномалии и пороги — только по train.

        ``raw_ht`` — телеметрия 24-2000 ДО очистки. Она нужна ровно для одного:
        восстановить остановы. Детектор достоверности справедливо считает
        замороженный на нуле расход сырья «не живым измерением» и убирает его,
        но именно этот замороженный ноль и есть факт останова. На очищенных
        данных из 10 остановов виден 1: провалы вырезаны вместе с браком.
        """
        cfg = cfg or load_config()
        train = tuple(cfg["split"]["train"])

        temps = [t for t in cls.REACTOR_TEMPS if t in ht.columns]
        frame = pd.DataFrame(index=ht.index)
        if temps:
            frame["wabt"] = ht[temps].mean(axis=1)
        if cls.DP_TAG in ht.columns:
            frame[cls.DP_TAG] = ht[cls.DP_TAG]
        if cls.FURNACE_TAG in avt.columns:
            frame[cls.FURNACE_TAG] = avt[cls.FURNACE_TAG]

        norms = SeverityNorms.fit(frame, list(frame.columns), train=train)

        # скорость изменения режима: максимальный по реакторным температурам модуль
        # изменения за час, нормированный на p95 обучающего периода
        ramp = None
        step = pd.Series(ht.index).diff().median()
        per_hour = max(int(pd.Timedelta("1h") / step), 1) if step else 6
        if temps:
            delta = ht[temps].diff(per_hour).abs().max(axis=1)
            scale = float(delta.loc[train[0]:train[1]].quantile(0.95)) or 1.0
            ramp = (delta / scale).clip(0.0, 1.5)

        # наработка от последнего останова: разметки нет, восстанавливаем по
        # длительным провалам расхода сырья (см. models/regime.py)
        run_hours, run_scale, feed_median = None, None, None
        source = raw_ht if raw_ht is not None else ht
        if FEED in source.columns:
            feed_median = float(source[FEED].median())
            run_hours = hours_since_outage(source[FEED],
                                           min_outage_hours=cls.CATALYST_RESET_HOURS,
                                           steps_per_hour=per_hour)
            run_scale = float(run_hours.loc[train[0]:train[1]].quantile(0.95)) or None

        down_series = None
        if feed_median and temps and all(t in source.columns for t in temps):
            cold = source[temps].mean(axis=1) < cls.DOWN_TEMP_C
            no_feed = source[FEED] < feed_median * cls.DOWN_FEED_LEVEL
            down_series = (cold & no_feed).reindex(ht.index).fillna(False)

        # многомерный детектор: описатели режима, а не отдельные теги
        anomaly_frame = frame.copy()
        if FEED in ht.columns:
            anomaly_frame["feed"] = ht[FEED]
            if RECYCLE_GAS in ht.columns:
                denom = ht[FEED].where(ht[FEED] > ht[FEED].median() * 0.1)
                anomaly_frame["h2_oil"] = ht[RECYCLE_GAS] / denom
        if "P13" in ht.columns:
            anomaly_frame["pressure"] = ht["P13"]
        detector = RegimeAnomalyDetector.fit(anomaly_frame, list(anomaly_frame.columns),
                                             train=train)

        agent = cls(norms=norms, ramp_series=ramp, run_hours=run_hours,
                    run_hours_scale=run_scale, detector=detector,
                    feed_median=feed_median, down_series=down_series)
        agent.thresholds = agent._calibrate_thresholds(frame, anomaly_frame, train)
        return agent

    # ------------------------------------------------------------------ #
    def _calibrate_thresholds(self, frame: pd.DataFrame, anomaly_frame: pd.DataFrame,
                              train: tuple[str, str]) -> tuple[float, float]:
        """Пороги risk_class из распределения severity на обучающем периоде.

        Раньше стояли 0.5 и 0.8 «на глаз». Привязка к квантилям означает конкретное
        обещание: «тяжёлый» режим — это 5 % самых напряжённых часов истории,
        «средний» — верхние 30 %. Такое утверждение можно проверить, в отличие от
        произвольного числа.
        """
        severity = self.severity_series(frame, anomaly_frame)
        train_part = severity.loc[train[0]:train[1]].dropna()
        if len(train_part) < 100:
            return self.DEFAULT_THRESHOLDS
        medium, high = train_part.quantile([0.70, 0.95])
        if not (0 < medium < high):
            return self.DEFAULT_THRESHOLDS
        return float(medium), float(high)

    def severity_series(self, frame: pd.DataFrame,
                        anomaly_frame: pd.DataFrame | None = None) -> pd.Series:
        """Severity по всей истории — тем же взвешиванием, что и для одного среза."""
        factors = pd.DataFrame(index=frame.index)
        for key, column in (("wabt", "wabt"), ("dp_r202", self.DP_TAG),
                            ("furnace", self.FURNACE_TAG)):
            if column in frame.columns:
                norm = self.norms.normalize_series(column, frame[column])
                if norm is not None:
                    factors[key] = norm
        if self.ramp_series is not None:
            factors["ramp"] = self.ramp_series.reindex(frame.index).clip(0.0, 1.0)
        if self.run_hours is not None and self.run_hours_scale:
            factors["catalyst"] = (self.run_hours.reindex(frame.index)
                                   / self.run_hours_scale).clip(0.0, 1.5)
        if anomaly_frame is not None and self.detector.fitted:
            distance = self.detector.distance(anomaly_frame.reindex(frame.index))
            factors["anomaly"] = pd.Series(distance / self.detector.threshold,
                                           index=frame.index).clip(0.0, 1.5)

        weights = pd.Series({k: self.WEIGHTS[k] for k in factors.columns})
        weighted = (factors * weights).sum(axis=1, min_count=1)
        norm = factors.notna().mul(weights, axis=1).sum(axis=1)
        return (weighted / norm.where(norm > 0)).clip(0.0, 1.0)

    # ------------------------------------------------------------------ #
    def _ramp_at(self, ts) -> float | None:
        if self.ramp_series is None:
            return None
        sub = self.ramp_series.loc[:ts].dropna()
        return None if sub.empty else float(sub.iloc[-1])

    def _factors(self, state: ProcessState) -> dict[str, float]:
        ht, avt = state.telemetry_ht, state.telemetry_avt
        temps = [ht.get(t) for t in self.REACTOR_TEMPS if ht.get(t) is not None]
        factors: dict[str, float] = {}

        if temps:
            n = self.norms.normalize("wabt", float(np.mean(temps)))
            if n is not None:
                factors["wabt"] = n

        for key, tag, store in (("dp_r202", self.DP_TAG, ht), ("furnace", self.FURNACE_TAG, avt)):
            n = self.norms.normalize(tag, store.get(tag))
            if n is not None:
                factors[key] = n

        ramp = self._ramp_at(state.ts)
        if ramp is not None:
            factors["ramp"] = float(np.clip(ramp, 0.0, 1.0))

        run = self._run_hours_at(state.ts)
        if run is not None and self.run_hours_scale:
            factors["catalyst"] = float(np.clip(run / self.run_hours_scale, 0.0, 1.5))

        score = self.detector.score_row(self._anomaly_inputs(state))
        if score is not None:
            factors["anomaly"] = float(np.clip(score, 0.0, 1.5))
        return factors

    def _run_hours_at(self, ts) -> float | None:
        if self.run_hours is None:
            return None
        sub = self.run_hours.loc[:ts].dropna()
        return None if sub.empty else float(sub.iloc[-1])

    def _anomaly_inputs(self, state: ProcessState) -> dict[str, float | None]:
        """Те же описатели режима, на которых обучался детектор."""
        ht, avt = state.telemetry_ht, state.telemetry_avt
        temps = [ht.get(t) for t in self.REACTOR_TEMPS if ht.get(t) is not None]
        feed = ht.get(FEED)
        gas = ht.get(RECYCLE_GAS)
        return {
            "wabt": float(np.mean(temps)) if temps else None,
            self.DP_TAG: ht.get(self.DP_TAG),
            self.FURNACE_TAG: avt.get(self.FURNACE_TAG),
            "feed": feed,
            "h2_oil": (gas / feed) if (gas is not None and feed) else None,
            "pressure": ht.get("P13"),
        }

    def is_unit_down(self, state: ProcessState) -> bool:
        """Установка стоит: сырья нет И реактор холодный.

        Одного признака мало: нулевой расход бывает и при отказе датчика, а
        низкая температура — при пуске. Вместе они однозначны.
        """
        if self.down_series is not None:
            sub = self.down_series.loc[:state.ts]
            if len(sub):
                return bool(sub.iloc[-1])

        feed = state.telemetry_ht.get(FEED)
        temps = [state.telemetry_ht.get(t) for t in self.REACTOR_TEMPS
                 if state.telemetry_ht.get(t) is not None]
        if feed is None or not temps or not self.feed_median:
            return False
        return (feed < self.feed_median * self.DOWN_FEED_LEVEL
                and float(np.mean(temps)) < self.DOWN_TEMP_C)

    def assess(self, state: ProcessState) -> ReliabilityAssessment:
        if self.is_unit_down(state):
            # Остановленной установке нельзя советовать менять уставки: любая
            # рекомендация по режиму бессмысленна, а severity, посчитанный по
            # холодному реактору, покажет обманчиво мягкий режим.
            return ReliabilityAssessment(
                ts=state.ts, severity_index=0.0, risk_class="low", admissible=False,
                factors={}, notes=["Установка остановлена: нет расхода сырья и "
                                   "реактор холодный. Управляющие рекомендации "
                                   "неприменимы до вывода на режим."],
            )

        factors = self._factors(state)
        if not factors:
            return ReliabilityAssessment(
                ts=state.ts, severity_index=0.5, risk_class="medium", admissible=True,
                notes=["Нет данных для оценки тяжести режима — принята средняя оценка."],
            )

        w = {k: self.WEIGHTS[k] for k in factors}
        total = sum(w.values())
        severity = float(np.clip(sum(factors[k] * w[k] for k in factors) / total, 0.0, 1.0))

        medium_thr, high_thr = self.thresholds
        risk_class = ("low" if severity < medium_thr
                      else "medium" if severity < high_thr else "high")
        notes = []
        constraints: dict[str, tuple[float, float]] = {}

        top = max(factors, key=factors.get)
        notes.append(f"Основной вклад в тяжесть режима: {top} ({factors[top]:.2f}); "
                     f"пороги {medium_thr:.2f}/{high_thr:.2f} — квантили обучающего периода.")

        if factors.get("anomaly", 0) > 1.0:
            parts = self.detector.contributions(self._anomaly_inputs(state))
            worst = ", ".join(f"{k} ({v:.0%})" for k, v in list(parts.items())[:3])
            notes.append(f"Режим нетипичен для истории: наибольший вклад дают {worst}. "
                         "Каждый параметр по отдельности в норме — нетипично их сочетание.")

        if factors.get("catalyst", 0) > 0.9:
            notes.append("Катализатор в конце цикла по наработке: та же глубина очистки "
                         "требует более жёсткого режима.")

        if risk_class == "high":
            notes.append("Тяжёлый режим: повышение температур запрещено до снижения severity.")
            for tag in self.REACTOR_TEMPS:
                v = state.telemetry_ht.get(tag)
                if v is not None:
                    constraints[tag] = (v - 3.0, v)      # только вниз
        elif risk_class == "medium":
            for tag in self.REACTOR_TEMPS:
                v = state.telemetry_ht.get(tag)
                if v is not None:
                    constraints[tag] = (v - 3.0, v + 1.0)

        if factors.get("ramp", 0) > 0.8:
            notes.append("Режим меняется быстро — дополнительные воздействия нежелательны.")

        return ReliabilityAssessment(
            ts=state.ts,
            severity_index=severity,
            risk_class=risk_class,
            factors=factors,
            admissible=risk_class != "high",
            constraints=constraints,
            notes=notes,
        )
