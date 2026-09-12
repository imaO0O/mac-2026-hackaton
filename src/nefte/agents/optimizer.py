"""Агент оптимизации.

Генерирует варианты изменения режима, отсекает недопустимые и ранжирует
оставшиеся. Жёсткие ограничения проверяются ДО ранжирования: никакая экономика
не может «перевесить» нарушение спецификации (ключевой принцип ТЗ).

Поиск устроен в три слоя, все считаются на CPU за доли секунды:
1. «ничего не делать» — обязательный кандидат, чтобы система не создавала
   лишних воздействий в устойчивом режиме;
2. покоординатная сетка — по одному изменению за раз, такие рекомендации
   оператору понятнее всего;
3. случайные точки плюс локальный поиск вокруг лучшего — для комбинаций.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

from nefte.agents.quality import T95_SIGMA_C, spec_risk_normal
from nefte.agents.schemas import (
    Candidate,
    ProcessState,
    QualityAssessment,
    ReliabilityAssessment,
)
from nefte.config import load_config
from nefte.models.regime import implied_t6
from nefte.models.vak import point_evaluator

# Во сколько σ закладываем запас по качеству: рекомендация должна оставаться
# допустимой не только в среднем, но и при разумно плохом исходе.
SAFETY_SIGMAS = 1.0

# Сетка, на которой варианты вообще считаются различными по критерию: доля
# диапазона значений среди допустимых вариантов. 0.2 — это пять градаций на
# критерий, то есть «заметно лучше / лучше / так же / хуже / заметно хуже».
#
# Это выбор ради читаемости, а не оценка точности, и его надо называть допущением.
# Причина: при четырёх непрерывных критериях строгий фронт Парето вырождается —
# недоминируемыми оказываются 97 % вариантов, и фронт перестаёт что-либо значить.
# Оператор всё равно различает варианты грубо, поэтому и сравниваем грубо.
PARETO_EPS = 0.2

# Доля от суммарного допустимого шага за цикл, начиная с которой два варианта
# считаются РАЗНЫМИ для показа оператору. ДОПУЩЕНИЕ: 0.15 — наш выбор.
ALTERNATIVE_SPREAD = 0.15


class QualitySurrogate(Protocol):
    """Модель «режим → качество», которой пользуется оптимизатор.

    Реализует участник 1 поверх обученной модели агента качества.
    """

    def __call__(self, state: ProcessState, moves: dict[str, float]) -> dict[str, float]:
        ...


def linear_surrogate(sensitivities: dict[str, float], base_key: str = "product_sulfur_mgkg"
                     ) -> QualitySurrogate:
    """Заглушка-суррогат: линейный отклик качества на изменение уставок.

    Нужна только чтобы цикл работал до появления обученной модели.
    Коэффициенты — ДОПУЩЕНИЕ, в демо не использовать без пометки.
    """

    def _fn(state: ProcessState, moves: dict[str, float]) -> dict[str, float]:
        base = state.quality.get("pak_sulfur_ppm")
        value = base.value if base and base.value is not None else 8.5
        for tag, new in moves.items():
            cur = state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))
            if cur is not None and tag in sensitivities:
                value += sensitivities[tag] * (new - cur)
        return {base_key: float(value)}

    return _fn


def default_t95_estimator(target: str = "24-2000:GODT:T95"
                          ) -> Callable[[ProcessState, dict[str, float]], float | None]:
    """Т95 продукта при заданных уставках: УРОВЕНЬ из лаборатории, ПРИРАЩЕНИЕ из формулы.

    Формула виртуального анализатора взята с листа «ВАК» с поправкой организаторов.
    Использовать её абсолютное значение как оценку Т95 нельзя, и это измерено: на
    400 последних анализах смещение всего −0.8 °C, но MAE 5.5 °C при разбросе самой
    лаборатории 6.5 °C и корреляции 0.30. То есть УРОВЕНЬ формула держит, а вот
    попадание в конкретное значение — нет, и жёсткий предел 360 °C по такому числу
    был бы ложной точностью.

    Зато коэффициенты формулы — это отклик, а он структурный: температура Р-202
    входит в Т95 с множителем 0.50, то есть +2 °C ради серы дают +1 °C к Т95.
    Поэтому берём то же правило, что и в кинетическом суррогате: уровень из
    измерения, приращение из модели. В разности лабораторный член формулы
    сокращается, и остаётся ровно чувствительность к уставкам::

        Т95(вариант) = Т95(лаборатория) + [ВАК(вариант) − ВАК(текущий режим)]

    Опорное значение приходит из среза, то есть уже с задержкой публикации. Оно
    стареет: между анализами до суток, и на столько же устаревает оценка.

    Возвращает None, когда посчитать не из чего. None означает «не знаем», и выше
    по коду разбирается отдельно: молча подставить ноль нельзя — ноль прошёл бы
    проверку предела.
    """
    evaluator = point_evaluator(target)

    def _at(state: ProcessState, moves: dict[str, float]) -> float | None:
        current = {**state.telemetry_avt, **state.telemetry_ht}
        # T6 никто не задаёт уставкой: это температура ниже по потоку, следствие
        # того, что сделали с T5 и T11. В формулу Т95 входит именно она, поэтому
        # без модели связи проверка была мёртвой — система двигала T5, формула
        # этого не видела. Связь измерена: ΔT6 = 0.72·ΔT5 + 0.18·ΔT11.
        derived = implied_t6(current, moves)
        values: dict[str, float] = {}
        for tag in evaluator.tags:
            if tag.startswith("LIMS_"):
                values[tag] = 0.0        # в разности сокращается
                continue
            if tag == "T6" and derived is not None:
                values[tag] = derived
                continue
            values[tag] = moves.get(tag, current.get(tag))
        return evaluator(values)

    # Опорная точка «текущий режим» от варианта не зависит, а оптимизатор зовёт
    # оценку для каждого из двухсот кандидатов подряд с одним и тем же срезом.
    # Пересчитывать её каждый раз — ровно вдвое лишней работы, поэтому держим
    # кэш на одну запись: он попадает почти всегда и не может устареть, потому
    # что ключ включает сам момент среза.
    cached_key: tuple | None = None
    cached_base: float | None = None

    def _base_at(state: ProcessState) -> float | None:
        nonlocal cached_key, cached_base
        # Ключ — момент среза И значения тех тегов, которые входят в формулу.
        # Одного момента мало: имитационная среда подсовывает на тот же момент
        # смещённый режим, и кэш по времени вернул бы чужое значение.
        key = (state.ts, tuple(
            (state.telemetry_ht.get(t, state.telemetry_avt.get(t)))
            for t in evaluator.tags if not t.startswith("LIMS_")))
        if key != cached_key:
            cached_key, cached_base = key, _at(state, {})
        return cached_base

    def _fn(state: ProcessState, moves: dict[str, float]) -> float | None:
        if evaluator is None:
            return None
        lab = state.quality.get("lims_t95_c")
        if lab is None or lab.value is None:
            return None
        if not moves:
            return float(lab.value)
        base, moved = _base_at(state), _at(state, moves)
        if base is None or moved is None:
            return None
        return float(lab.value) + (moved - base)

    return _fn


# --------------------------------------------------------------------------- #
# производительность и энергозатраты
# --------------------------------------------------------------------------- #

def default_throughput(feed_tag: str = "F26", product_tag: str = "F17"
                       ) -> Callable[[ProcessState, dict[str, float]], float]:
    """Выпуск гидроочищенного ДТ, т/ч.

    Фактических материальных балансов в пакете нет, поэтому считаем, что выпуск
    меняется пропорционально расходу сырья на установку. ДОПУЩЕНИЕ, помечено в
    docs/DATA_NOTES.md.
    """

    def _fn(state: ProcessState, moves: dict[str, float]) -> float:
        product = state.telemetry_ht.get(product_tag)
        feed_now = state.telemetry_ht.get(feed_tag)
        if product is None:
            return 0.0
        feed_new = moves.get(feed_tag, feed_now)
        if not feed_now or feed_new is None:
            return float(product)
        return float(product * feed_new / feed_now)

    return _fn


def default_energy_proxy() -> Callable[[ProcessState, dict[str, float]], float]:
    """Прозрачный стоимостной прокси энергозатрат (безразмерный).

    Экономических данных в пакете нет. Складываем то, что физически тянет за
    собой топливо и пар, с явными весами:

    * температура на выходе печи П-3 сверх 370 °C — топливо печи;
    * реакторные температуры сверх 350 °C — нагрев ГСС;
    * расход сырья — прокачка и нагрев потока;
    * расход пара в колонны АВТ.

    Все коэффициенты — ДОПУЩЕНИЕ. Важно не абсолютное значение, а то, что
    сравнение вариантов между собой воспроизводимо.
    """

    def _fn(state: ProcessState, moves: dict[str, float]) -> float:
        def value(tag: str) -> float | None:
            if tag in moves:
                return moves[tag]
            return state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))

        energy = 0.0
        furnace = value("T55")
        if furnace is not None:
            energy += max(furnace - 370.0, 0.0) * 1.0
        for tag in ("T5", "T6", "T11"):
            temp = value(tag)
            if temp is not None:
                energy += max(temp - 350.0, 0.0) * 0.3
        feed = value("F26")
        if feed is not None:
            energy += feed * 0.02
        for tag in ("F26", "F27", "F28", "F29"):
            steam = state.telemetry_avt.get(tag)
            if steam is not None:
                energy += max(steam, 0.0) * 0.01
        return float(energy)

    return _fn


# --------------------------------------------------------------------------- #

class OptimizerAgent:
    """Сэмплирование кандидатов в допустимой области + многокритериальный выбор."""

    def __init__(self, bounds: dict[str, tuple[float, float]],
                 surrogate: QualitySurrogate,
                 cfg: dict | None = None,
                 throughput_fn: Callable[[ProcessState, dict[str, float]], float] | None = None,
                 energy_fn: Callable[[ProcessState, dict[str, float]], float] | None = None,
                 grid_levels: int = 5,
                 reliability_agent=None,
                 t95_fn: Callable[[ProcessState, dict[str, float]], float | None] | None = None):
        # Агент надёжности нужен, чтобы пересчитать тяжесть режима под каждый
        # вариант. Необязателен: без него severity берётся текущий, как раньше,
        # и это честно видно по тому, что критерий перестаёт различать варианты.
        self.reliability_agent = reliability_agent
        self.bounds = bounds                 # модельные диапазоны (допущение!)
        self.surrogate = surrogate
        self.cfg = cfg or load_config()
        self.throughput_fn = throughput_fn or default_throughput()
        self.energy_fn = energy_fn or default_energy_proxy()
        # Т95 — второй обязательный показатель. Он не «ещё один критерий»: по
        # исправленной формуле ВАК температура Р-202 входит в Т95 с коэффициентом
        # 0.50, то есть каждые +2 °C ради серы дают +1 °C к Т95. Запас до предела
        # 360 °C бывает в пять градусов, так что несколько шагов подряд выводят
        # продукт за спецификацию по другому показателю. Без этой проверки система
        # чинила одно за счёт другого и не знала об этом.
        #
        # ВЛАДЕЛЕЦ оценки — агент качества: Т95 это показатель качества, и считать
        # его оптимизатору не по чину. Оркестратор при сборке подменяет эту функцию
        # на ту, что у агента качества, чтобы не появилось двух независимых
        # реализаций одного показателя — в этом проекте они уже расходились дважды.
        # Значение по умолчанию нужно лишь для прогонов, где оптимизатор создают
        # в одиночку.
        self.t95_fn = t95_fn if t95_fn is not None else default_t95_estimator()
        self.grid_levels = grid_levels
        self.seed = int(self.cfg["optimization"]["random_seed"])
        # причины отсева последнего прогона — оркестратор объясняет ими отказ
        self._last_violations: list[str] = []
        # требуемый запас по неопределённости с последнего прогона
        self._required_margin: float = SAFETY_SIGMAS * 1.7
        # оценённый вариант «ничего не делать»: нужен как точка отсчёта даже тогда,
        # когда сам он недопустим (текущий режим уже у предела)
        self._last_hold: Candidate | None = None

    # ------------------------------------------------------------------ #
    def _rng(self, state: ProcessState, stream: int = 0) -> np.random.Generator:
        """Генератор, привязанный к МОМЕНТУ, а не к порядку вызовов.

        Раньше генератор жил в агенте и продвигался от вызова к вызову: один и
        тот же срез при повторном прогоне давал ДРУГУЮ рекомендацию. Это ломало
        обещание воспроизводимости из README и вдобавок портило обе проверки
        устойчивости — базовый прогон и возмущённые считались при разном
        состоянии генератора, так что в «чувствительность к весам» попадал ещё и
        случайный разброс.

        ``stream`` разделяет независимые потоки: сетка кандидатов и локальное
        уточнение не должны брать одни и те же числа.
        """
        moment = int(pd.Timestamp(state.ts).value)
        return np.random.default_rng([self.seed, moment, stream])

    def _severity_for(self, state: ProcessState, reliability: ReliabilityAssessment,
                      moves: dict[str, float]) -> float:
        agent = self.reliability_agent
        if agent is None or not hasattr(agent, "severity_for"):
            return reliability.severity_index
        return float(agent.severity_for(state, moves))

    def _effective_bounds(self, state: ProcessState,
                          reliability: ReliabilityAssessment) -> dict[str, tuple[float, float]]:
        """Пересечение модельного диапазона, ограничения агента надёжности и шага."""
        steps = self.cfg["limits"]["max_step_per_cycle"]
        out: dict[str, tuple[float, float]] = {}
        for tag, (lo, hi) in self.bounds.items():
            cur = state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))
            if cur is None:
                continue
            step = steps["temperature_c"] if tag.startswith("T") else (
                steps["pressure_mpa"] if tag.startswith("P") else abs(cur) * steps["flow_rel"])
            lo_e, hi_e = max(lo, cur - step), min(hi, cur + step)
            if tag in reliability.constraints:
                r_lo, r_hi = reliability.constraints[tag]
                lo_e, hi_e = max(lo_e, r_lo), min(hi_e, r_hi)
            if hi_e >= lo_e:
                out[tag] = (lo_e, hi_e)
        return out

    @staticmethod
    def _candidate(idx: str, current: dict[str, float], moves: dict[str, float],
                   bounds: dict[str, tuple[float, float]] | None = None) -> Candidate:
        """Собирает вариант, дотягивая незатронутые теги внутрь допустимой области.

        Если текущее значение тега уже вне диапазона, разрешённого агентом
        надёжности, «не трогать его» — не нейтральное действие: вариант остался бы
        недопустимым. Поэтому такой тег возвращается к ближайшей границе, и это
        честно показывается оператору как отдельное изменение.
        """
        full: dict[str, float] = {}
        for tag, value in current.items():
            if value is None:
                continue
            new_value = moves.get(tag, value)
            if bounds and tag in bounds:
                lo, hi = bounds[tag]
                new_value = min(max(new_value, lo), hi)
            full[tag] = float(new_value)
        deltas = {t: full[t] - current[t] for t in full if current.get(t) is not None}
        return Candidate(id=idx, moves=full, deltas=deltas)

    def generate(self, state: ProcessState, reliability: ReliabilityAssessment,
                 n: int | None = None) -> list[Candidate]:
        """«Ничего не делать» + покоординатная сетка + случайные комбинации."""
        n = n or self.cfg["optimization"]["n_candidates"]
        bounds = self._effective_bounds(state, reliability)
        current = {t: state.telemetry_ht.get(t, state.telemetry_avt.get(t)) for t in bounds}

        cands = [self._candidate("hold", current, {})]

        # покоординатно: одно изменение за раз — самая понятная оператору форма
        for tag, (lo, hi) in bounds.items():
            for level in np.linspace(lo, hi, self.grid_levels):
                if current[tag] is None or abs(level - current[tag]) < 1e-9:
                    continue
                cands.append(self._candidate(f"{tag}={level:.2f}", current,
                                             {tag: float(level)}, bounds))

        # комбинации
        rng = self._rng(state, stream=0)
        n_random = max(n - len(cands), 0)
        for i in range(n_random):
            moves = {tag: float(rng.uniform(lo, hi)) for tag, (lo, hi) in bounds.items()}
            cands.append(self._candidate(f"mix_{i:03d}", current, moves, bounds))
        return cands

    def refine(self, state: ProcessState, reliability: ReliabilityAssessment,
               best: Candidate, n: int = 40, shrink: float = 0.25) -> list[Candidate]:
        """Локальный поиск вокруг лучшего кандидата — уточнение без новых рисков."""
        bounds = self._effective_bounds(state, reliability)
        current = {t: state.telemetry_ht.get(t, state.telemetry_avt.get(t)) for t in bounds}
        rng = self._rng(state, stream=1)
        out = []
        for i in range(n):
            moves = {}
            for tag, (lo, hi) in bounds.items():
                span = (hi - lo) * shrink
                center = best.moves.get(tag, current.get(tag))
                if center is None:
                    continue
                moves[tag] = float(np.clip(rng.normal(center, span / 2), lo, hi))
            out.append(self._candidate(f"local_{i:03d}", current, moves, bounds))
        return out

    # ------------------------------------------------------------------ #
    def evaluate(self, state: ProcessState, cands: list[Candidate],
                 quality: QualityAssessment,
                 reliability: ReliabilityAssessment) -> list[Candidate]:
        """Прогноз качества и проверка жёстких ограничений для каждого кандидата."""
        limit = self.cfg["spec"]["product_sulfur_mgkg"]["max"]

        # σ прогноза берём из интервала агента качества: запас должен опираться на
        # измеренную неопределённость, а не на константу
        interval = quality.intervals.get("product_sulfur_mgkg")
        sigma = (interval[1] - interval[0]) / (2 * 1.96) if interval else 1.7
        margin = SAFETY_SIGMAS * float(sigma)
        # запас, дальше которого улучшать качество бессмысленно: он нужен ранжированию
        self._required_margin = margin

        # прогноз для «ничего не делать» — точка отсчёта: если текущий режим уже
        # близок к пределу, отказываться от улучшающего действия неправильно
        hold_pred = None
        for c in cands:
            if c.id == "hold":
                hold_pred = self.surrogate(state, c.moves).get("product_sulfur_mgkg")
                break

        t95_limit = self.cfg["spec"]["t95_c"]["max"]
        # Т95 текущего режима: с ним сравниваем вариант. Если Т95 УЖЕ за пределом,
        # запрещать варианты по этому признаку бессмысленно — тогда важно лишь то,
        # что вариант не делает хуже.
        t95_now = self.t95_fn(state, {}) if self.t95_fn else None

        for c in cands:
            pred = self.surrogate(state, c.moves)
            c.predicted_quality = pred
            sulfur = pred.get("product_sulfur_mgkg")
            violations = []
            guaranteed = False

            # Второй обязательный показатель. Проверяем ДО ранжирования: вариант,
            # который чинит серу ценой Т95, недопустим, а не «чуть хуже по баллам».
            t95 = self.t95_fn(state, c.moves) if self.t95_fn else None
            if t95 is not None:
                pred["product_t95_c"] = float(t95)
                worse = t95_now is not None and t95 > t95_now + 1e-9
                if t95 > t95_limit and (t95_now is None or t95_now <= t95_limit or worse):
                    violations.append(
                        f"Т95 {t95:.1f} °C выходит за {t95_limit} — вариант чинит серу "
                        f"за счёт другого обязательного показателя")

            if sulfur is None or sulfur != sulfur:
                violations.append("нет прогноза качества")
            else:
                guaranteed = sulfur + margin <= limit
                improving = hold_pred is not None and sulfur < hold_pred - 1e-9
                if not guaranteed and not improving:
                    violations.append(
                        f"сера {sulfur:.2f} + запас {margin:.2f} мг/кг выходит за {limit} "
                        f"и вариант не лучше бездействия")
            if not reliability.admissible:
                violations.append("режим признан недопустимым агентом надёжности")

            c.guaranteed = bool(guaranteed)
            # Вероятность, а не флаг «выше предела». Раньше здесь стоял 0/1, и в
            # карточке оператора все альтернативы выглядели одинаково безрисковыми,
            # хотя запас у них разный. σ берём ту же, что и для запаса: другой
            # оценки неопределённости у суррогата нет, и выдумывать её нельзя.
            c.spec_risk = {"product_sulfur_mgkg":
                           spec_risk_normal(sulfur, float(sigma), limit)
                           if sulfur == sulfur else 1.0}
            # Т95 — такой же обязательный показатель, и у вариантов он тоже должен
            # нести ВЕРОЯТНОСТЬ, а не только значение. Иначе в карточке оператора
            # альтернативы сравнимы по сере и несравнимы по разгонке, хотя
            # ограничение действует по обеим. σ здесь — неопределённость нашего
            # знания о текущем Т95 (суточный уход показателя, 6.64 °C измерено),
            # а не точность формулы: уровень мы берём из последнего анализа.
            if t95 is not None:
                c.spec_risk["product_t95_c"] = spec_risk_normal(
                    float(t95), T95_SIGMA_C, t95_limit)
            c.throughput = self.throughput_fn(state, c.moves)
            c.energy_proxy = self.energy_fn(state, c.moves)
            # Тяжесть режима У ЭТОГО варианта, а не у текущего: иначе критерий
            # severity в свёртке и на фронте Парето вырождается в константу.
            c.severity_index = self._severity_for(state, reliability, c.moves)
            c.violations = violations
            c.feasible = not violations

        self._last_violations = [v for c in cands for v in c.violations]
        for c in cands:
            if c.id == "hold":
                self._last_hold = c
                break
        return cands

    # ------------------------------------------------------------------ #
    def rank(self, cands: list[Candidate]) -> list[Candidate]:
        """Взвешенная свёртка + ранг Парето. Ранжируются ТОЛЬКО допустимые.

        Варианты с гарантированным запасом всегда идут выше тех, что просто
        улучшают качество: гарантия важнее свёртки критериев.
        """
        w = self.cfg["optimization"]["objective_weights"]
        limit = self.cfg["spec"]["product_sulfur_mgkg"]["max"]
        feas = [c for c in cands if c.feasible]
        if not feas:
            return []

        def norm(values: list[float]) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            rng = np.nanmax(arr) - np.nanmin(arr)
            return (arr - np.nanmin(arr)) / rng if rng > 0 else np.full(len(arr), 0.5)

        # `or limit` здесь был бы ошибкой того же рода, что и в блендинге: прогноз
        # ровно 0.0 — валидное число, а не «нет прогноза».
        #
        # Запас НАСЫЩАЕТСЯ: как только прогноз ушёл ниже предела на требуемый запас
        # по неопределённости, дальнейшее углубление очистки ценности не имеет.
        # Без насыщения критерий качества всегда тянет «ещё чище», и в замкнутом
        # контуре система шаг за шагом упирала режим в предел допустимого дрейфа:
        # T5 и T11 уходили на +8 °C, а прогноз серы — к нулю (docs/HARD_CHECKS.md §7).
        # Поштучно каждый шаг был безопасен, а последовательность — нет.
        required = getattr(self, "_required_margin", SAFETY_SIGMAS * 1.7)

        def _margin(c: Candidate) -> float:
            value = c.predicted_quality.get("product_sulfur_mgkg")
            if value is None:
                return 0.0
            return min(limit - float(value), float(required))

        quality_margin = norm([_margin(c) for c in feas])
        throughput = norm([c.throughput or 0.0 for c in feas])
        energy = norm([c.energy_proxy or 0.0 for c in feas])
        severity = norm([c.severity_index or 0.0 for c in feas])

        # Подавление воздействия: при прочих равных меньшее вмешательство лучше.
        # Стандартный приём промышленных регуляторов (move suppression), и здесь он
        # нужен по конкретной причине: в замкнутом контуре без него система шаг за
        # шагом уводила уставки в упор допустимого дрейфа и начинала рыскать —
        # каждый шаг сам по себе разумен, а последовательность нет
        # (docs/HARD_CHECKS.md §7).
        steps = self.cfg["limits"]["max_step_per_cycle"]

        def _effort(c: Candidate) -> float:
            """Размер воздействия в долях разрешённого шага: теги несравнимы в единицах."""
            total = 0.0
            for tag, delta in c.deltas.items():
                if abs(delta) < 1e-9:
                    continue
                if tag.startswith("T"):
                    scale = float(steps["temperature_c"])
                elif tag.startswith("P"):
                    scale = float(steps["pressure_mpa"])
                else:
                    base = abs(c.moves.get(tag, 0.0) - delta)
                    scale = max(base * float(steps["flow_rel"]), 1e-6)
                total += abs(delta) / scale
            return total

        effort = norm([_effort(c) for c in feas])
        move_weight = float(w.get("move_penalty", 0.0))

        for i, c in enumerate(feas):
            c.score = float(w["quality_margin"] * quality_margin[i]
                            + w["throughput"] * throughput[i]
                            - w["energy_proxy"] * energy[i]
                            - w["severity"] * severity[i]
                            - move_weight * effort[i])

        # Парето: максимизируем запас и выпуск, минимизируем энергию и тяжесть.
        # Сравниваем не точные числа, а округлённые до PARETO_EPS доли диапазона:
        # различие в тысячную долю мг/кг — это не различие, а шум суррогата. Без
        # округления при четырёх критериях недоминируемыми оказываются почти все
        # варианты (150 из 152 на реальном срезе), и фронт перестаёт что-либо
        # значить оператору.
        objectives = np.column_stack([quality_margin, throughput, -energy, -severity])
        grid = np.round(objectives / PARETO_EPS)
        for i, c in enumerate(feas):
            dominated = (np.all(grid >= grid[i], axis=1)
                         & np.any(grid > grid[i], axis=1))
            c.pareto_rank = int(dominated.sum())

        return sorted(feas, key=lambda c: (not c.guaranteed, -(c.score or 0.0),
                                           c.pareto_rank or 0))

    # ------------------------------------------------------------------ #
    def last_hold(self) -> Candidate | None:
        """Оценённый вариант «ничего не делать» из последнего прогона."""
        return self._last_hold

    def rejection_summary(self) -> str:
        """Почему отсеялись варианты — человекочитаемо, для объяснения отказа.

        Оператору важно не «допустимых нет», а какое именно ограничение уперлось:
        качество, модельный диапазон или запрет агента надёжности.
        """
        if not self._last_violations:
            return "причины отсева неизвестны"
        kinds: dict[str, int] = {}
        for v in self._last_violations:
            key = ("прогноз качества с запасом выходит за предел" if "сера" in v else
                   "режим признан недопустимым по тяжести" if "недопустим" in v else
                   "нет прогноза качества" if "нет прогноза" in v else v)
            kinds[key] = kinds.get(key, 0) + 1
        top = sorted(kinds.items(), key=lambda kv: -kv[1])
        return "; ".join(f"{name} ({count})" for name, count in top[:3])

    @staticmethod
    def pareto_front(cands: list[Candidate]) -> list[Candidate]:
        """Недоминируемые варианты — то, между чем реально выбирает технолог."""
        return [c for c in cands if c.pareto_rank == 0]

    def min_alternative_distance(self) -> float:
        """Насколько альтернативы обязаны различаться, чтобы их стоило показывать.

        Доля от максимального шага за цикл, суммарно по управляющим тегам. Порог
        1e-3, стоявший здесь раньше, различием не является: на дашборде три
        «разные» альтернативы отличались в третьем знаке, и график их разброса
        показывал шум вместо выбора.
        """
        steps = self.cfg["limits"]["max_step_per_cycle"]
        total = 0.0
        for tag, (lo, hi) in self.bounds.items():
            if tag.startswith("T"):
                total += float(steps["temperature_c"])
            elif tag.startswith("P"):
                total += float(steps["pressure_mpa"])
            else:                     # расходы: шаг задан долей от значения
                total += abs(hi - lo) * float(steps["flow_rel"])
        return max(total * ALTERNATIVE_SPREAD, 1e-3)

    def diverse_alternatives(self, cands: list[Candidate], k: int = 3,
                             min_distance: float | None = None) -> list[Candidate]:
        """k различающихся альтернатив: показывать три почти одинаковых бессмысленно."""
        threshold = (self.min_alternative_distance() if min_distance is None
                     else min_distance)
        out: list[Candidate] = []
        for c in cands:
            if len(out) >= k:
                break
            if all(_distance(c, other) > threshold for other in out):
                out.append(c)
        # Если настолько разных вариантов нет — это факт, а не повод показывать
        # похожие: добираем ближайшие, но честно, начиная с лучшего.
        for c in cands:
            if len(out) >= k:
                break
            if c not in out:
                out.append(c)
        return out

    def propose(self, state: ProcessState, quality: QualityAssessment,
                reliability: ReliabilityAssessment) -> list[Candidate]:
        cands = self.evaluate(state, self.generate(state, reliability), quality, reliability)
        ranked = self.rank(cands)
        if not ranked:
            return []
        extra = self.evaluate(state, self.refine(state, reliability, ranked[0]),
                              quality, reliability)
        return self.rank(cands + extra)


def _distance(a: Candidate, b: Candidate) -> float:
    """Насколько два варианта различаются по управляющим воздействиям."""
    tags = set(a.deltas) | set(b.deltas)
    return sum(abs(a.deltas.get(t, 0.0) - b.deltas.get(t, 0.0)) for t in tags)
