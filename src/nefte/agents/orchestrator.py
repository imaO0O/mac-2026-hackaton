"""Оркестратор: один цикл принятия решения.

Порядок ровно как в п.1 ТЗ: состояние → проверка данных → качество → надёжность →
варианты → отсев → сравнение → смешение → рекомендация либо мотивированный отказ.

Блок смешения стоит ПОСЛЕ выбора режима, а не рядом с ним: рецептуру считать
имеет смысл для того продукта, который получится в рекомендуемом режиме. Обратный
порядок — «подобрать рецептуру, чтобы вытянуть некачественный продукт» — запрещён
правилом системы: качество и технологические ограничения выше экономики.

Каждый прогон сохраняется в reports/runs/*.json: входные данные, ответы агентов и
итог — чтобы логику решения можно было проверить постфактум (требование ТЗ).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from nefte.agents.blending import BlendingAgent
from nefte.agents.optimizer import OptimizerAgent
from nefte.agents.quality import QualityAgent, fuse_sulfur
from nefte.agents.reliability import ReliabilityAgent
from nefte.agents.schemas import (
    BlendComponent,
    ProcessState,
    QualityAssessment,
    Recommendation,
    Source,
    TraceStep,
)
from nefte.config import ROOT, load_config

RUNS_DIR = ROOT / "reports" / "runs"


# Применённый сдвиг меньше этой доли разрешённого шага не возвращаем: гоняться за
# остатком в сотые доли шага значит дёргать оборудование ради шума.
RETURN_MIN_STEP_SHARE = 0.25
# Во сколько раз запас по сере у возврата больше обычного. Выбран на окне
# валидационного периода (scripts/check_return_to_base.py): 1.0 даёт весь выигрыш,
# 1.25 и 1.5 возврат почти убивают, 2.0 не возвращает ничего.
RETURN_MARGIN_FACTOR = 1.0

class Orchestrator:
    """Связывает агентов и разрешает конфликт целей."""

    def __init__(self, quality: QualityAgent, reliability: ReliabilityAgent,
                 optimizer: OptimizerAgent, cfg: dict | None = None,
                 min_confidence: float = 0.35, act_risk_threshold: float | None = None,
                 log_runs: bool = True, min_hours_between_actions: float | None = None,
                 blending: BlendingAgent | None = None,
                 components_fn: Callable[[object], list[BlendComponent]] | None = None,
                 additive_ppm: float = 0.0):
        self.quality = quality
        self.reliability = reliability
        self.optimizer = optimizer
        # Т95 считает АГЕНТ КАЧЕСТВА — это показатель качества. Оптимизатор берёт
        # у него готовую функцию, а не строит свою: две независимые реализации
        # одного показателя разъезжаются, и в этом проекте уже разъезжались
        # (два пути очистки телеметрии, пять определений исхода рекомендации).
        if getattr(quality, "t95_fn", None) is not None:
            optimizer.t95_fn = quality.t95_fn
        # Блок смешения необязателен: без него цикл работает как раньше, а
        # рекомендация просто не содержит рецептуры.
        self.blending = blending
        self.components_fn = components_fn
        self.additive_ppm = additive_ppm
        self.cfg = cfg or load_config()
        self.min_confidence = min_confidence
        # порог, начиная с которого вмешиваемся: берём подобранный вместе с моделью
        self.act_risk_threshold = (act_risk_threshold if act_risk_threshold is not None
                                   else getattr(quality, "alarm_threshold", 0.2))
        self.log_runs = log_runs
        # Ограничение частоты воздействий: дёргать уставки каждый цикл нельзя,
        # отклик качества запаздывает и режим не успевает устояться. Значение
        # живёт в конфиге рядом с остальными пределами и опирается на измеренную
        # постоянную времени канала 4.6 ч (scripts/find_delays.py). Вместе с
        # limits.max_step_per_cycle оно и задаёт предельную скорость изменения
        # режима — 0.5 °C/ч, — поэтому прятать его в значении аргумента по
        # умолчанию было неправильно: главный параметр безопасности не должен
        # зависеть от того, кто как создал оркестратор.
        self.min_hours_between_actions = float(
            self.cfg["limits"].get("min_hours_between_actions", 4.0)
            if min_hours_between_actions is None else min_hours_between_actions)
        self._last_action_ts = None
        # Сдвиги уставок, которые ДЕЙСТВИТЕЛЬНО применены: тег → суммарное изменение.
        # Копятся только через record_applied — его зовёт тот, кто применяет
        # рекомендацию (имитатор, оператор в интерфейсе). Считать сдвигом саму
        # выданную рекомендацию нельзя: в разомкнутом прогоне по истории ничего не
        # применяется, и система начала бы «возвращать» то, чего никто не делал.
        self._applied: dict[str, float] = {}
        # последнее действие РАДИ КАЧЕСТВА (не возврат): от него отсчитывается пауза
        self._last_quality_action_ts = None
        self.return_settle_hours = float(
            self.cfg["limits"].get("return_settle_hours", 14.0))
        self.return_to_base = bool(self.cfg["limits"].get("return_to_base", False))

    def record_applied(self, moved: dict[str, float]) -> None:
        """Учесть применённые изменения уставок — от них считается возврат к базе."""
        for tag, delta in moved.items():
            if abs(delta) < 1e-9:
                continue
            total = self._applied.get(tag, 0.0) + float(delta)
            if abs(total) < 1e-9:
                self._applied.pop(tag, None)
            else:
                self._applied[tag] = total

    def _return_to_base(self, state, q, r, hold, candidates):
        """Шаг назад к базовому режиму, если запас по качеству позволяет.

        Зачем. Действие рекомендуется только при риске выше порога, поэтому то, что
        однажды сделано ради серы, раньше не отменялось никогда: в имитации расход
        сырья снижался на 40 единиц и не возвращался, а сера оставалась в разы ниже
        предела — потерянный выпуск без пользы для качества.

        Условия — все сразу:

        * есть применённый сдвиг хотя бы по одному тегу больше четверти его шага;
        * с последнего действия ради качества прошло ``limits.return_settle_hours``
          (3τ ≈ 14 ч): без паузы возврат приходил через 4 часа после среза, пока
          эффект среза проявился наполовину, и контур качался;
        * риск сейчас ниже ПОЛОВИНЫ порога вмешательства — гистерезис: иначе контур
          качался бы между «вернуть» и «снова сдвинуть»;
        * вариант проходит жёсткие ограничения оптимизатора с ГАРАНТИРОВАННЫМ запасом
          по сере (``RETURN_MARGIN_FACTOR`` — множитель запаса, выбран на валидации);
        * Т95 у варианта не выше, чем у бездействия, — даже в пределах нормы. Возврат
          необязателен, и платить за него Т95 незачем. Без этого условия первая
          версия возвращала T5 вверх, пока T11 оставался поднятым, то есть снимала
          компенсацию и поднимала Т95 (средний вклад вырос с +0.12 до +0.34 °C);
        * сам оптимизатор ПРЕДПОЧИТАЕТ вариант бездействию по своей же свёртке
          критериев — выпуск, энергия, тяжесть режима, штраф за воздействие — при
          насыщенном запасе качества. Возвращать то, что критериям безразлично,
          значит дёргать оборудование: первая версия гоняла давление туда-обратно
          19 раз за месяц;
        * каждое изменение варианта уменьшает применённый сдвиг своего тега — никаких
          попутных движений.

        Сначала пробуется вернуть все сдвинутые теги сразу, по шагу каждый, затем по
        одному, начиная с самого большого сдвига. Запрет частых воздействий
        соблюдается снаружи: сюда не заходят, пока он действует.
        """
        if not self.return_to_base or not self._applied or hold is None:
            return None
        if self._last_quality_action_ts is not None:
            since = (pd.Timestamp(state.ts)
                     - pd.Timestamp(self._last_quality_action_ts)).total_seconds() / 3600
            if 0 <= since < self.return_settle_hours:
                return None
        risk = q.spec_risk.get("product_sulfur_mgkg", 0.0)
        if risk >= self.act_risk_threshold / 2:
            return None
        optimizer = self.optimizer
        steps = self.cfg["limits"]["max_step_per_cycle"]
        bounds = optimizer._effective_bounds(state, r)
        current = {t: state.telemetry_ht.get(t, state.telemetry_avt.get(t)) for t in bounds}

        def scale(tag: str) -> float | None:
            if tag.startswith("T"):
                return float(steps["temperature_c"])
            if tag.startswith("P"):
                return float(steps["pressure_mpa"])
            now = current.get(tag)
            return None if now is None else abs(float(now)) * float(steps["flow_rel"])

        shifted = []
        for tag, offset in self._applied.items():
            size = scale(tag)
            if tag not in bounds or current.get(tag) is None or not size:
                continue
            if abs(offset) > RETURN_MIN_STEP_SHARE * size:
                shifted.append((abs(offset) / size, tag, offset, size))
        if not shifted:
            return None
        shifted.sort(reverse=True)

        def back(tag, offset, size):
            return float(current[tag]) - (1.0 if offset > 0 else -1.0) * min(abs(offset), size)

        options = [{tag: back(tag, offset, size) for _, tag, offset, size in shifted}]
        if len(shifted) > 1:
            options += [{tag: back(tag, offset, size)} for _, tag, offset, size in shifted]

        t95_limit = float(self.cfg["spec"]["t95_c"]["max"])
        t95_hold = hold.predicted_quality.get("product_t95_c")
        saved = list(getattr(optimizer, "_last_violations", []))
        try:
            for moves in options:
                cand = optimizer._candidate("return_to_base", current, moves, bounds)
                cand = optimizer.evaluate(state, [cand], q, r)[0]
                if not cand.feasible or not cand.guaranteed:
                    continue
                sulfur = cand.predicted_quality.get("product_sulfur_mgkg")
                margin = float(getattr(optimizer, "_required_margin", 0.0))
                limit = float(self.cfg["spec"]["product_sulfur_mgkg"]["max"])
                if sulfur is None or sulfur + RETURN_MARGIN_FACTOR * margin > limit:
                    continue
                real = {t: d for t, d in cand.deltas.items() if abs(d) > 1e-6}
                if not real or any(self._applied.get(t, 0.0) * d >= 0 for t, d in real.items()):
                    continue
                t95 = cand.predicted_quality.get("product_t95_c")
                if t95 is not None and t95_hold is not None and t95 > t95_hold + 1e-6:
                    continue
                if t95 is not None and t95 > t95_limit:
                    continue
                # Сравнение со всеми вариантами этого цикла: свёртка нормирует
                # критерии по набору, и в паре «вариант — бездействие» любая мелочь
                # растянулась бы на весь диапазон. Копии — чтобы не менять оценки
                # альтернатив, которые покажут оператору.
                pool = [c.model_copy(deep=True) for c in candidates if c.id != cand.id]
                if not any(c.id == "hold" for c in pool):
                    pool.append(hold.model_copy(deep=True))
                pool.append(cand)
                optimizer.rank(pool)
                hold_score = next(c.score for c in pool if c.id == "hold")
                if hold_score is None or cand.score is None or cand.score <= hold_score + 1e-9:
                    continue
                return cand
        finally:
            optimizer._last_violations = saved
        return None

    # ------------------------------------------------------------------ #
    def run(self, state: ProcessState) -> Recommendation:
        q = self.quality.assess(state)
        r = self.reliability.assess(state)

        freshness = {
            key: m.age_hours for key, m in state.quality.items()
        }
        measured = fuse_sulfur(state, self.cfg)
        limit = self.cfg["spec"]["product_sulfur_mgkg"]["max"]
        state_summary = {
            # именно тот источник, по которому принято решение: раньше здесь
            # оказывался первый попавшийся в срезе — например, сера СЫРЬЯ
            "sulfur_source": q.source.value,
            # само измеренное значение: без него оператор не видит, что факт УЖЕ
            # вне спецификации, и читает карточку как разговор про будущий риск
            "sulfur_measured": (None if measured.value is None
                                else round(float(measured.value), 2)),
            "severity_index": round(r.severity_index, 3),
            "risk_class": r.risk_class,
        }
        # Факт важнее прогноза: если последнее достоверное измерение уже за
        # пределом, это и есть главная строка карточки. Раньше система говорила
        # «риск 29 %», хотя лаборатория показала 10.2 мг/кг при пределе 10.
        off_spec_now = ""
        if measured.value is not None and measured.value > limit:
            age = (f", возраст {measured.age_hours:.0f} ч"
                   if measured.age_hours is not None else "")
            off_spec_now = (f"ФАКТ ВНЕ СПЕЦИФИКАЦИИ: {measured.source.value} "
                            f"{measured.value:.2f} мг/кг при пределе {limit}{age}. ")

        # То же самое для ВТОРОГО обязательного показателя, и по той же причине.
        #
        # Оптимизатор запрещает варианты, ухудшающие Т95, но когда Т95 уже за
        # пределом, запрещать по нему бессмысленно: бездействие не «хуже себя», и
        # оно проходит как допустимое. Логика верная — а следствие было такое:
        # карточка писала «Режим устойчив, риск 8 %» и «текущий режим
        # удовлетворяет ограничениям», показывая рядом Т95 364 °C при пределе 360
        # и перечисляя «Т95 ≤ 360 (жёсткое, прогноз)» как проверенное.
        #
        # Заметка про это агентом качества СОЗДАВАЛАСЬ («вероятность выхода
        # 72 %») и никуда не попадала: q.notes доходят до карточки только по
        # ветке отказа. Третий за вечер случай «проверка есть, но не проверяет».
        t95_limit = float(self.cfg["spec"]["t95_c"]["max"])
        t95_pred = q.predictions.get("product_t95_c")
        t95_risk = q.spec_risk.get("product_t95_c", 0.0)
        t95_off_spec = t95_pred is not None and float(t95_pred) > t95_limit
        t95_alert = ""
        if t95_off_spec:
            # «Прогноз 364 при вероятности 12 %» на демо читалось как противоречие.
            # Число слева — оценка по последнему лабораторному анализу, а вероятность
            # — что показатель выше предела СЕЙЧАС: после таких анализов режим обычно
            # поправляют, и поправленная по истории вероятность это знает.
            t95_alert = (f"Т95 ЗА ПРЕДЕЛОМ по последнему анализу: {float(t95_pred):.1f} °C "
                         f"при {t95_limit:.0f}; вероятность, что он выше предела сейчас, — "
                         f"{t95_risk:.0%}. ")

        # --- отказ 1: установка не в работе -----------------------------
        # Проверяется ПЕРВОЙ: на остановленной установке устаревший ЛИМС и
        # зависший анализатор — следствия останова, а не самостоятельные причины.
        # Назвать оператору следствие вместо причины значит сбить его с толку.
        if not r.admissible and any("остановлена" in note for note in r.notes):
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem="Установка не в работе",
                abstained=True, confidence=q.confidence,
                abstain_reason="; ".join(r.notes),
            )
            return self._finish(rec, state, q, r, blend=False,
                                rule="отказ: установка не в работе")

        # --- отказ 2: данные непригодны ---------------------------------
        if not state.data_quality.usable or q.confidence < self.min_confidence:
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem="Недостаточно достоверных данных для оценки качества",
                abstained=True, confidence=q.confidence,
                # причины — законченные фразы с точкой; склейка через «; » давала «ч.; »
                abstain_reason=" ".join(
                    n if n.rstrip().endswith((".", "!", "?")) else n.rstrip() + "."
                    for n in q.notes + state.data_quality.notes if n)
                or "низкая уверенность прогноза",
            )
            return self._finish(rec, state, q, r, blend=False,
                                rule="отказ: данные недостоверны или уверенность ниже "
                                     f"{self.min_confidence:.2f}")

        risk = q.spec_risk.get("product_sulfur_mgkg", 0.0)
        pred = q.predictions.get("product_sulfur_mgkg")

        candidates = self.optimizer.propose(state, q, r)

        # --- отказ 3: допустимых вариантов нет --------------------------
        if not candidates:
            reasons = ["Ни один вариант не проходит жёсткие ограничения",
                       self.optimizer.rejection_summary()]
            reasons += [n for n in r.notes if n]
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem=off_spec_now + f"Риск нарушения спецификации по сере: {risk:.0%}",
                abstained=True, confidence=q.confidence,
                abstain_reason=". ".join(x for x in reasons if x) + ". "
                               "Требуется решение технолога.",
            )
            return self._finish(
                rec, state, q, r, optimized=True,
                rule="отказ: ни один вариант не проходит жёсткие ограничения")

        best = candidates[0]
        # точка отсчёта — всегда оценённое бездействие, даже если оно недопустимо
        hold = self.optimizer.last_hold() or next(
            (c for c in candidates if c.id == "hold"), None)

        # Воздействовали недавно — ждём отклика, а не накладываем новое изменение.
        # Переступаем через лимит только если продукт УЖЕ вне спецификации: тогда
        # ждать нельзя. Просто высокий риск ожидания не отменяет.
        already_off_spec = pred is not None and pred > self.cfg["spec"]["product_sulfur_mgkg"]["max"]
        if self._too_soon(state.ts) and hold is not None and not already_off_spec:
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem=off_spec_now + t95_alert + f"Риск нарушения спецификации: {risk:.0%}",
                action=hold, confidence=q.confidence,
                expected_effect={"сера, мг/кг": round(pred, 2) if pred else "н/д"},
                checked_constraints=self._constraint_log(),
                explanation=(
                    f"Предыдущее воздействие было менее {self.min_hours_between_actions:.0f} ч "
                    "назад. Качество реагирует на изменение режима с запаздыванием, "
                    "поэтому ждём отклика вместо нового вмешательства. "
                    "Спецификация при этом не нарушена."),
                alternatives=self.optimizer.diverse_alternatives(candidates, 3),
            )
            return self._finish(
                rec, state, q, r, optimized=True,
                rule=("запрет частых воздействий: прошло меньше "
                      f"{self.min_hours_between_actions:g} ч"))

        # --- возврат к базовому режиму: запас позволяет отыграть сделанное ---
        if risk < self.act_risk_threshold and hold is not None and hold.feasible:
            back = self._return_to_base(state, q, r, hold, candidates)
            if back is not None:
                shifted = ", ".join(f"{t} {v:+.2f}" for t, v in sorted(self._applied.items()))
                rec = Recommendation(
                    ts=state.ts, state_summary=state_summary, freshness=freshness,
                    problem=off_spec_now + t95_alert
                            + f"Возврат к базовому режиму: риск {risk:.0%}, запас по сере "
                              f"позволяет отыграть ранее применённые изменения",
                    action=back, confidence=q.confidence,
                    expected_effect=self._effect(back, hold, r),
                    checked_constraints=self._constraint_log(t95_off_spec),
                    explanation=(
                        f"Ранее применённые изменения уставок: {shifted}. Качество сейчас "
                        "с запасом, поэтому шаг назад к исходному режиму возвращает выпуск "
                        "и не требует жертвовать серой: вариант проходит те же жёсткие "
                        "ограничения с гарантированным запасом, а риск ниже половины порога "
                        "вмешательства. Возвращаем не больше одного шага за цикл."),
                    alternatives=self.optimizer.diverse_alternatives(candidates, 3),
                )
                self._last_action_ts = state.ts
                return self._finish(
                    rec, state, q, r, optimized=True,
                    rule="возврат к базовому режиму: запас по сере позволяет")

        # --- нормальный режим: не создаём лишних воздействий ------------
        if risk < self.act_risk_threshold and hold is not None and hold.feasible:
            if t95_off_spec:
                # «Режим устойчив» здесь было бы неправдой: обязательных
                # показателя три, и один из них вне спецификации.
                headline = (f"Сера спокойна (риск {risk:.0%}), но Т95 за пределом")
                why = ("По сере изменение уставок не требуется. Т95 за пределом уже "
                       "сейчас, и система его НЕ оптимизирует: она только запрещает "
                       "варианты, которые ухудшают Т95 ради серы. Решение по Т95 — "
                       "за технологом.")
            elif off_spec_now:
                # Измерение уже выше предела, а риск модели низкий. Карточка писала
                # «ФАКТ ВНЕ СПЕЦИФИКАЦИИ» и тут же «Режим устойчив… изменение
                # уставок не требуется». На часовом прогоне теста так было в 347
                # моментах из 5233. Само решение проверено и оставлено: после пробы
                # выше предела следующая выше в 17–26 % случаев по периодам, и
                # вероятность модели в таких моментах не занижена (обучение 0.23
                # против 0.28, валидация 0.13 против 0.19 на 7 событиях, тест 0.18
                # против 0.16). Противоречивой была только карточка.
                headline = (f"Последнее измерение выше предела, но прогноз по текущим "
                            f"данным {pred:.2f} мг/кг, риск {risk:.0%} — ниже порога "
                            f"вмешательства {self.act_risk_threshold:.0%}"
                            if pred is not None else
                            f"Последнее измерение выше предела, но риск {risk:.0%} — ниже "
                            f"порога вмешательства {self.act_risk_threshold:.0%}")
                age = ("" if measured.age_hours is None
                       else f", отобранную {measured.age_hours:.0f} ч назад")
                repeat = (" Одиночное превышение по лаборатории в следующей пробе "
                          "повторяется реже, чем в одном случае из трёх."
                          if measured.source is Source.LIMS else "")
                why = (f"Измерение описывает пробу{age}, а решение — о ближайших часах; "
                       "это измерение уже входит в данные, по которым считается "
                       f"прогноз.{repeat} Режим не меняем и следим за следующим "
                       "анализом; если превышение подтвердится, решение за технологом.")
            else:
                headline = ("Режим устойчив, риск выхода за спецификацию "
                            f"{risk:.0%}" if risk < self.act_risk_threshold / 2 else
                            f"Риск {risk:.0%} — ниже порога вмешательства "
                            f"{self.act_risk_threshold:.0%}, режим держим под наблюдением")
                why = ("Текущий режим удовлетворяет ограничениям, "
                       "изменение уставок не требуется.")
            rec = Recommendation(
                ts=state.ts, state_summary=state_summary, freshness=freshness,
                problem=off_spec_now + t95_alert + headline,
                action=hold, confidence=q.confidence,
                expected_effect=self._effect(hold, hold, r),
                checked_constraints=self._constraint_log(t95_off_spec),
                explanation=why,
                alternatives=self.optimizer.diverse_alternatives(candidates, 3),
            )
            return self._finish(
                rec, state, q, r, optimized=True,
                rule=(f"риск {risk:.0%} ниже порога {self.act_risk_threshold:.0%}, "
                      "текущий режим допустим — держим"))

        # --- есть риск: рекомендуем действие ----------------------------
        # Действие при риске НИЖЕ порога бывает, когда у бездействия нет
        # гарантированного запаса по сере. На тесте это 10 случаев из 99, и во всех
        # запас нарушен на сотые: прогноз плюс запас 10.01–10.06 при пределе 10, а
        # действие — поправка в доли шага. Решение верное, но карточка с «риск 16 %»
        # и действием выглядела противоречием, поэтому причина называется прямо.
        below = ""
        if risk < self.act_risk_threshold:
            hold_sulfur = (hold.predicted_quality.get("product_sulfur_mgkg")
                           if hold is not None else None)
            margin = float(getattr(self.optimizer, "_required_margin", 0.0))
            why = (f"прогноз {hold_sulfur:.2f} + запас {margin:.2f} = "
                   f"{hold_sulfur + margin:.2f} мг/кг при пределе {limit}"
                   if hold_sulfur is not None else "бездействие не проходит ограничения")
            below = (f"Риск {risk:.0%} ниже порога вмешательства "
                     f"{self.act_risk_threshold:.0%}, но у текущего режима нет "
                     f"гарантированного запаса по сере: {why}. Нужна небольшая поправка. ")
        rec = Recommendation(
            ts=state.ts, state_summary=state_summary, freshness=freshness,
            problem=off_spec_now + t95_alert + below
                    + f"Риск нарушения спецификации по сере: {risk:.0%} "
                      f"(прогноз {pred:.2f} мг/кг при пределе {limit})",
            action=best,
            expected_effect=self._effect(best, hold, r),
            checked_constraints=self._constraint_log(t95_off_spec),
            explanation=self._explain(best, candidates, q, r,
                                      q.confidence * (1.0 if best.guaranteed else 0.6)),
            confidence=q.confidence * (1.0 if best.guaranteed else 0.6),
            alternatives=self.optimizer.diverse_alternatives(candidates[1:], 3),
        )
        if any(abs(d) > 1e-6 for d in best.deltas.values()):
            self._last_action_ts = state.ts
            self._last_quality_action_ts = state.ts
        return self._finish(
            rec, state, q, r, optimized=True,
            rule=(f"риск {risk:.0%} не ниже порога {self.act_risk_threshold:.0%} — "
                  "выбираем лучший допустимый вариант"
                  if risk >= self.act_risk_threshold else
                  "у бездействия нет гарантированного запаса — нужна поправка"))

    # ------------------------------------------------------------------ #
    def _too_soon(self, ts) -> bool:
        if self._last_action_ts is None:
            return False
        hours = (pd.Timestamp(ts) - pd.Timestamp(self._last_action_ts)).total_seconds() / 3600
        return 0 <= hours < self.min_hours_between_actions

    def _effect(self, best, hold, r) -> dict:
        """Эффект показываем относительно бездействия — иначе он ни о чём не говорит."""
        out: dict[str, float | str] = {
            "сера, мг/кг": round(best.predicted_quality.get("product_sulfur_mgkg", float("nan")), 2),
            "тяжесть режима": round(r.severity_index, 2),
        }
        # Т95 — второй обязательный показатель. Показываем рядом с серой: оператор
        # должен видеть обе стороны размена, а не только ту, ради которой двигают
        # режим.
        t95 = best.predicted_quality.get("product_t95_c")
        if t95 is not None:
            out["Т95, °C"] = round(float(t95), 1)
        if hold is not None:
            base = hold.predicted_quality.get("product_sulfur_mgkg")
            cur = best.predicted_quality.get("product_sulfur_mgkg")
            if base is not None and cur is not None:
                out["сера к бездействию"] = round(cur - base, 2)
            # Т95 тоже к бездействию, а не только абсолютом. Без этого размен
            # виден наполовину: оператор читает «сера −1.93, выпуск −3 %» и
            # «Т95 355», из чего нельзя понять, двинул ли ход разгонку и в какую
            # сторону. А ход по температуре двигает её всегда: в исправленной
            # формуле ВАК коэффициент по Т6 равен 0.50.
            t95_base = hold.predicted_quality.get("product_t95_c")
            if t95_base is not None and t95 is not None:
                out["Т95 к бездействию"] = round(float(t95) - float(t95_base), 2)
            if hold.throughput and best.throughput is not None:
                out["выпуск, %"] = round((best.throughput / hold.throughput - 1) * 100, 2)
            if hold.energy_proxy and best.energy_proxy is not None:
                out["энергия, %"] = round((best.energy_proxy / hold.energy_proxy - 1) * 100, 2)
        return out

    def _constraint_log(self, t95_off_spec: bool = False) -> list[str]:
        """Что именно проверено — и, столь же важно, что НЕ проверено.

        Список обязан стареть вместе с кодом. Пока модели Т95 не было, здесь
        честно стояло «Т95 при смене режима не прогнозируется»; теперь она есть, и
        оставить прежнюю строку значило бы соврать в другую сторону. Оператор
        читает этот список как перечень того, за что система отвечает.

        Из показателей спецификации при смене режима прогнозируются два: сера
        (моделью) и Т95 (по формуле виртуального анализатора — уровень из
        лаборатории, приращение из формулы). Плотность, ПТФ и цетановое число
        моделей не имеют и проверяются по последнему лабораторному анализу.
        """
        spec = self.cfg["spec"]
        # Когда Т95 УЖЕ за пределом, писать «Т95 ≤ 360 (жёсткое)» без оговорки
        # значит обещать выполнение того, что не выполняется. Ограничение при этом
        # действует — но в ослабленном виде: оно запрещает ухудшать Т95, а не
        # обязывает вернуть его в норму.
        t95_line = (f"Т95 ≤ {spec['t95_c']['max']} °C (жёсткое, прогноз: уровень из "
                    "лаборатории, приращение по формуле ВАК)")
        if t95_off_spec:
            t95_line = (f"Т95 УЖЕ выше {spec['t95_c']['max']} °C: варианты, "
                        "ухудшающие его, запрещены, но вернуть в норму система не "
                        "берётся — это решение технолога")
        out = [f"сера ≤ {spec['product_sulfur_mgkg']['max']} мг/кг (жёсткое, прогноз)",
               t95_line,
               "уставки внутри модельного диапазона (допущение, p05–p95 истории)",
               "шаг изменения за цикл ограничен"]
        if self.blending is not None:
            out.append(
                f"цетановое число ≥ {spec['cetane_number']['min']}, плотность и ПТФ — "
                "по последнему анализу, при смене режима НЕ прогнозируются")
        # Строку про сумму долей добавляет _finish, когда рецептура действительно
        # посчитана: дублировать её здесь значит обещать проверку там, где
        # смешение вообще не считалось.
        return out

    def _explain(self, best, candidates, q, r, confidence: float | None = None) -> str:
        n_feas = len(candidates)
        n_front = len(self.optimizer.pareto_front(candidates))
        moves = ", ".join(f"{t} {d:+.3f}" if abs(d) < 0.01 else f"{t} {d:+.2f}"
                          for t, d in best.deltas.items() if abs(d) > 1e-6)
        text = (
            f"Из {n_feas} допустимых вариантов ({n_front} на фронте Парето) выбран "
            f"{best.id}: {moves or 'без изменений'}. Он даёт наибольший запас по сере "
            f"при тяжести режима {r.severity_index:.2f} ({r.risk_class}). "
            f"Уверенность {confidence if confidence is not None else q.confidence:.2f}."
        )
        # конфликт целей: качество тянет вверх по температуре, надёжность — вниз
        if not best.guaranteed:
            text += (" ВНИМАНИЕ: запас по неопределённости не гарантирован — вариант "
                     "снижает серу относительно бездействия, но остаётся близко к пределу. "
                     "Нужен контрольный лабораторный анализ.")
        if r.constraints:
            limited = ", ".join(sorted(r.constraints))
            text += (f" Агент надёжности ограничил {limited}, поэтому вариант выбран "
                     f"внутри суженного диапазона, а не по максимуму качества.")
        if getattr(self.optimizer.surrogate, "kind", None) == "kinetic":
            # Главное допущение отклика — порядок реакции, а не энергия активации:
            # первый порядок даёт −22 % серы на градус при литературных 5–10 %
            # (scripts/check_kinetic_order.py). Называть надо оба.
            text += (" Уровень серы взят у модели, а ПРИРАЩЕНИЕ от изменения уставок "
                     "посчитано по кинетике гидрообессеривания — это допущение "
                     "(первый порядок, энергия активации 100 кДж/моль), а не "
                     "измеренный отклик; реальный отклик может быть слабее, и тогда "
                     "понадобится больше шагов.")
        return text

    # ------------------------------------------------------------------ #
    def _attach_blend(self, rec: Recommendation, state: ProcessState,
                      q: QualityAssessment) -> None:
        """Рецептура смешения для того продукта, который даст выбранный режим.

        Смешение НЕ используется как способ вытянуть некачественный продукт: если
        прогноз серы уже за пределом, допустимой рецептуры не существует, и агент
        это показывает — разбавлять товарное ДТ прямогонкой под Евро-5 нельзя
        (предельная доля порядка сотых долей процента, docs/BLENDING.md).
        """
        if self.blending is None or self.components_fn is None:
            return
        sulfur = self._blend_basis(rec, q)
        if sulfur is None or sulfur != sulfur:
            return
        components = self.components_fn(state.ts)
        if not components:
            return

        recipe = self.blending.with_forecast(components, sulfur,
                                             additive_ppm=self.additive_ppm)
        rec.blend = recipe
        rec.checked_constraints.append(
            f"доли компонентов смешения дают {recipe.fractions_sum() * 100:.1f} % "
            "(жёсткое требование ТЗ)")
        if not recipe.feasible:
            rec.explanation = (rec.explanation + " Смешением это не компенсируется: "
                               + "; ".join(recipe.violations) + ".").strip()
        elif len(recipe.fractions) > 1:
            rec.expected_effect["выпуск смеси, т/ч"] = round(recipe.throughput_tph, 1)

    @staticmethod
    def _blend_basis(rec: Recommendation, q: QualityAssessment) -> float | None:
        """Сера, на которой считается рецептура: прогноз выбранного варианта.

        Если действие не выбрано (отказ), берём прогноз агента качества для
        текущего режима — вопрос «спасёт ли рецептура» задаётся именно к нему.
        """
        if rec.action is not None:
            value = rec.action.predicted_quality.get("product_sulfur_mgkg")
            if value is not None:
                return float(value)
        value = q.predictions.get("product_sulfur_mgkg")
        return None if value is None else float(value)

    def _trace(self, rec: Recommendation, state, q, r, rule: str,
               optimized: bool) -> list[TraceStep]:
        """Путь решения: что вернул каждый агент и какое правило сработало."""
        steps: list[TraceStep] = []
        dq = state.data_quality
        lims, pak = state.quality.get("lims_sulfur_mgkg"), state.quality.get("pak_sulfur_ppm")

        def measurement(m, name: str) -> str:
            if m is None or m.value is None:
                return f"{name} нет"
            age = "" if m.age_hours is None else f", {m.age_hours:.0f} ч"
            flag = ", завис" if getattr(m, "is_frozen", False) else ""
            return f"{name} {m.value:.2f}{age}{flag}"

        steps.append(TraceStep(
            agent="срез состояния",
            summary=(f"{measurement(lims, 'ЛИМС')}; {measurement(pak, 'ПАК')}; пропусков "
                     f"{dq.missing_share:.1%}; заглушек в {len(dq.sentinel_tags)} тегах, "
                     f"«полок» в {len(dq.frozen_tags)}"
                     + ("" if dq.usable else "; срез непригоден")),
            details={"возраст источников, ч": dict(rec.freshness), "замечания": list(dq.notes)}))

        sulfur = q.predictions.get("product_sulfur_mgkg")
        t95 = q.predictions.get("product_t95_c")
        steps.append(TraceStep(
            agent="агент качества",
            summary=(f"источник {q.source.value}; прогноз серы "
                     + ("нет" if sulfur is None else f"{sulfur:.2f} мг/кг")
                     + f", риск {q.spec_risk.get('product_sulfur_mgkg', 0.0):.0%}; "
                     + ("" if t95 is None else f"Т95 {t95:.1f} °C; ")
                     + f"уверенность {q.confidence:.2f}"),
            details={"прогнозы": dict(q.predictions), "риски": dict(q.spec_risk),
                     "замечания": list(q.notes)}))

        steps.append(TraceStep(
            agent="агент надёжности",
            summary=(f"тяжесть режима {r.severity_index:.2f} ({r.risk_class}); режим "
                     + ("допустим" if r.admissible else "НЕДОПУСТИМ")
                     + ("; ограничил " + ", ".join(sorted(r.constraints)) if r.constraints
                        else "")),
            details={"факторы": dict(r.factors), "ограничения": {
                k: list(v) for k, v in r.constraints.items()}, "замечания": list(r.notes)}))

        if optimized:
            stats = self.optimizer.last_stats()
            chosen = rec.action.id if rec.action is not None else "нет"
            summary = (f"вариантов {stats.get('вариантов', 0)}, допустимых "
                       f"{stats.get('допустимых', 0)}, с гарантированным запасом "
                       f"{stats.get('с запасом', 0)}, на фронте Парето "
                       f"{stats.get('на фронте Парето', 0)}; лучший — {chosen}")
            if not stats.get("допустимых"):
                summary += "; отсев: " + self.optimizer.rejection_summary()
            steps.append(TraceStep(agent="оптимизатор", summary=summary, details=stats))

        if rec.blend is not None:
            steps.append(TraceStep(
                agent="смешение",
                summary=(f"рецептура {'допустима' if rec.blend.feasible else 'НЕДОПУСТИМА'}, "
                         f"сумма долей {rec.blend.fractions_sum() * 100:.1f} %, выпуск "
                         f"{rec.blend.throughput_tph:.1f} т/ч"),
                details={"доли": dict(rec.blend.fractions),
                         "нарушения": list(rec.blend.violations)}))

        steps.append(TraceStep(agent="оркестратор",
                               summary=f"{rule or 'правило не названо'} → {rec.outcome()}"))
        return steps

    def _finish(self, rec: Recommendation, state, q, r,
                blend: bool = True, rule: str = "", optimized: bool = False
                ) -> Recommendation:
        if blend:
            self._attach_blend(rec, state, q)
        rec.trace = self._trace(rec, state, q, r, rule, optimized)
        if self.log_runs:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            path: Path = RUNS_DIR / f"{state.ts:%Y%m%dT%H%M}.json"
            payload = {
                "state": json.loads(state.model_dump_json()),
                "quality": json.loads(q.model_dump_json()),
                "reliability": json.loads(r.model_dump_json()),
                "recommendation": json.loads(rec.model_dump_json()),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                       default=str), encoding="utf-8")
        return rec
