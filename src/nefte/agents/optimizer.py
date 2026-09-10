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

from nefte.agents.quality import spec_risk_normal
from nefte.agents.schemas import (
    Candidate,
    ProcessState,
    QualityAssessment,
    ReliabilityAssessment,
)
from nefte.config import load_config

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
                 reliability_agent=None):
        # Агент надёжности нужен, чтобы пересчитать тяжесть режима под каждый
        # вариант. Необязателен: без него severity берётся текущий, как раньше,
        # и это честно видно по тому, что критерий перестаёт различать варианты.
        self.reliability_agent = reliability_agent
        self.bounds = bounds                 # модельные диапазоны (допущение!)
        self.surrogate = surrogate
        self.cfg = cfg or load_config()
        self.throughput_fn = throughput_fn or default_throughput()
        self.energy_fn = energy_fn or default_energy_proxy()
        self.grid_levels = grid_levels
        self.rng = np.random.default_rng(self.cfg["optimization"]["random_seed"])
        # причины отсева последнего прогона — оркестратор объясняет ими отказ
        self._last_violations: list[str] = []
        # оценённый вариант «ничего не делать»: нужен как точка отсчёта даже тогда,
        # когда сам он недопустим (текущий режим уже у предела)
        self._last_hold: Candidate | None = None

    # ------------------------------------------------------------------ #
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
        n_random = max(n - len(cands), 0)
        for i in range(n_random):
            moves = {tag: float(self.rng.uniform(lo, hi)) for tag, (lo, hi) in bounds.items()}
            cands.append(self._candidate(f"mix_{i:03d}", current, moves, bounds))
        return cands

    def refine(self, state: ProcessState, reliability: ReliabilityAssessment,
               best: Candidate, n: int = 40, shrink: float = 0.25) -> list[Candidate]:
        """Локальный поиск вокруг лучшего кандидата — уточнение без новых рисков."""
        bounds = self._effective_bounds(state, reliability)
        current = {t: state.telemetry_ht.get(t, state.telemetry_avt.get(t)) for t in bounds}
        out = []
        for i in range(n):
            moves = {}
            for tag, (lo, hi) in bounds.items():
                span = (hi - lo) * shrink
                center = best.moves.get(tag, current.get(tag))
                if center is None:
                    continue
                moves[tag] = float(np.clip(self.rng.normal(center, span / 2), lo, hi))
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

        # прогноз для «ничего не делать» — точка отсчёта: если текущий режим уже
        # близок к пределу, отказываться от улучшающего действия неправильно
        hold_pred = None
        for c in cands:
            if c.id == "hold":
                hold_pred = self.surrogate(state, c.moves).get("product_sulfur_mgkg")
                break

        for c in cands:
            pred = self.surrogate(state, c.moves)
            c.predicted_quality = pred
            sulfur = pred.get("product_sulfur_mgkg")
            violations = []
            guaranteed = False

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

        quality_margin = norm([limit - (c.predicted_quality.get("product_sulfur_mgkg") or limit)
                               for c in feas])
        throughput = norm([c.throughput or 0.0 for c in feas])
        energy = norm([c.energy_proxy or 0.0 for c in feas])
        severity = norm([c.severity_index or 0.0 for c in feas])

        for i, c in enumerate(feas):
            c.score = float(w["quality_margin"] * quality_margin[i]
                            + w["throughput"] * throughput[i]
                            - w["energy_proxy"] * energy[i]
                            - w["severity"] * severity[i])

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

    @staticmethod
    def diverse_alternatives(cands: list[Candidate], k: int = 3) -> list[Candidate]:
        """k различающихся альтернатив: показывать три почти одинаковых бессмысленно."""
        out: list[Candidate] = []
        for c in cands:
            if len(out) >= k:
                break
            if all(_distance(c, other) > 1e-3 for other in out):
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
