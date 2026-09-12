"""Модель качества: прогноз серы в товарном ДТ + честный интервал.

Три градиентных бустинга CatBoost на CPU (обучение — минуты, GPU не нужен):
* q50 — медианный прогноз (MAE-устойчивый),
* q10 и q90 — границы интервала.

σ восстанавливается из ширины интервала, а не задаётся константой: у оркестратора
вероятность нарушения спецификации считается именно по ней, поэтому интервал
должен быть измеренным, а не назначенным.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nefte.agents.schemas import ProcessState
from nefte.config import ROOT

MODELS_DIR = ROOT / "models"

# Физика, зашитая в модель монотонными ограничениями CatBoost.
# −1: рост признака СНИЖАЕТ серу в продукте, +1: повышает.
# Смысл не в точности, а в направлении: без этого дерево кусочно-постоянно и
# на сдвиг уставки может ответить нулём или ответить в неверную сторону, а
# оптимизатор потом «улучшает» качество, понижая температуру реактора.
PHYSICS_MONOTONE: dict[str, int] = {
    # глубже режим — чище продукт
    "reg_wabt": -1, "reg_wabt_mean36": -1, "reg_wabt_mean144": -1,
    "reg_kinetic": -1, "reg_kinetic_mean36": -1, "reg_kinetic_mean144": -1,
    "reg_drive": -1,
    "reg_h2_partial": -1, "reg_h2_partial_mean36": -1, "reg_h2_partial_mean144": -1,
    "reg_h2_oil": -1, "reg_makeup_h2_oil": -1,
    # экзотерма — следствие глубины реакций: больше тепловыделение, чище продукт.
    # Без этого знака рост T11 «улучшал» серу через незажатый признак разности.
    "reg_dt_react": -1,
    "ht_T5": -1, "ht_T6": -1, "ht_T11": -1, "ht_P13": -1,
    # выше нагрузка и охлаждение слоя — меньше время контакта и глубина очистки
    "ht_F26": 1, "reg_quench_ratio": 1, "ht_F15": 1,
    # катализатор стареет — при тех же условиях сера растёт
    "reg_run_hours": 1,
}
# Направление влияния для Т95 — ВТОРОГО обязательного показателя.
#
# Знаки взяты не из общих соображений: исправленная организаторами формула
# виртуального анализатора `24-2000:GODT:T95` содержит температуру Р-202 с
# коэффициентом +0.50, то есть глубже режим — тяжелее хвост разгонки. По данным
# направление то же: корреляция Т95 с T5 +0.14, с расходом сырья +0.21.
#
# Знаков здесь МЕНЬШЕ, чем у серы, и это намеренно. По сере физика гидроочистки
# известна по каждому каналу; по Т95 уверенно известен только знак температуры и
# нагрузки, а остальное — догадки. Зашивать догадку в ограничение модели хуже,
# чем оставить признак свободным: ограничение нельзя переучить данными.
PHYSICS_MONOTONE_T95: dict[str, int] = {
    # выше температура реактора — тяжелее продукт
    "ht_T5": 1, "ht_T6": 1, "ht_T11": 1,
    "reg_wabt": 1, "reg_wabt_mean36": 1, "reg_wabt_mean144": 1,
    # выше нагрузка — хуже отпарка лёгких, хвост уходит вверх
    "ht_F26": 1,
}

# Наборы знаков по показателям: модель выбирает свой по имени цели.
PHYSICS_BY_TARGET: dict[str, dict[str, int]] = {
    "sulfur": PHYSICS_MONOTONE,
    "t95": PHYSICS_MONOTONE_T95,
}

# Признаки, которые НЕ лежат в матрице, а считаются в момент запроса: они зависят
# от того, когда спросили, а не только от истории. Матрица строится по сетке
# времени и такого столбца содержать не может.
SERVE_COMPUTED = ("feature_age_h",)

# 90 % интервал: σ = (q90 - q10) / (2 * 1.2816)
Z90 = 1.2815515655446004


def conformal_sigma_scale(pred: pd.DataFrame, y: pd.Series, target: float = 0.8) -> float:
    """Во сколько раз растянуть σ, чтобы покрытие на валидации совпало с номиналом.

    Квантильные модели — и бустинг, и нейросеть — систематически дают слишком
    узкий интервал на новых данных. Поправка берётся из распределения нормированной
    ошибки |y − q50| / σ. Оркестратор считает риск по σ, поэтому её честность важнее
    точности точечного прогноза.
    """
    norm_err = ((pred["q50"] - y).abs() / pred["sigma"]).replace([np.inf], np.nan).dropna()
    if norm_err.empty:
        return 1.0
    return float(max(norm_err.quantile(target) / Z90, 0.1))


def interval_risk(pred: pd.DataFrame, limit: float) -> pd.Series:
    """P(значение > limit) по прогнозу и σ в нормальном приближении."""
    return pd.Series(1 - _normal_cdf((limit - pred["q50"]) / pred["sigma"]),
                     index=pred.index)


def pick_alarm_threshold(risk, over, beta: float = 1.5,
                         min_lift: float = 1.5) -> tuple[float, bool]:
    """Порог тревоги по валидации: максимум F-beta при достаточной точности.

    Возвращает ``(порог, надёжна ли тревога)``. Второе значение False означает,
    что ни один порог не даёт precision выше ``min_lift`` от базовой частоты
    нарушений: тогда единственное честное правило — сигналить, только если сам
    точечный прогноз выше предела.
    """
    risk = np.asarray(risk, dtype=float)
    over = np.asarray(over, dtype=bool)
    if len(risk) == 0 or over.sum() == 0:
        return 0.5, False
    base_rate = float(over.mean())
    grid = np.unique(np.quantile(risk, np.linspace(0.40, 0.99, 60)).round(4))

    best, best_f = None, -1.0
    for thr in grid:
        alarm = risk > thr
        tp = int((alarm & over).sum())
        fp = int((alarm & ~over).sum())
        fn = int((~alarm & over).sum())
        if tp == 0:
            continue
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        f = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        if f > best_f and precision >= min_lift * base_rate:
            best, best_f = float(thr), f
    if best is None:
        return 0.5, False
    return best, True


def pick_threshold_for_budget(risk, budget: float) -> float:
    """Порог, при котором тревога срабатывает не чаще, чем на доле ``budget`` моментов.

    Зачем он нужен рядом с F-beta. Порог по F-beta — это компромисс, который модель
    выбирает сама, и у разных моделей он даёт РАЗНУЮ частоту вмешательств: у
    бустинга получилось 30 % моментов, у нейросети 60 %. Сравнивать их после этого
    нельзя: та, что кричит вдвое чаще, поймает больше превышений просто поэтому.

    Бюджет тревог задаёт рабочую точку явно и одинаково для всех моделей, а сам он
    — решение технолога («сколько вмешательств в спокойный режим мы готовы
    терпеть»), а не свойство данных. Поэтому он вынесен в конфиг и помечен
    допущением.
    """
    risk = np.asarray(risk, dtype=float)
    if len(risk) == 0:
        return 0.5
    budget = float(min(max(budget, 1e-3), 1.0))
    return float(np.quantile(risk, 1.0 - budget))


def rolling_budget_threshold(risk: pd.Series, budget: float,
                             window_days: int = 30,
                             min_periods: int = 20) -> pd.Series:
    """Порог тревоги по бюджету, пересчитываемый на скользящем окне.

    Зачем он нужен вместо одного числа с валидации. Порог по бюджету — это
    квантиль РАСПРЕДЕЛЕНИЯ РИСКА, и он осмыслен ровно настолько, насколько это
    распределение стабильно. У бустинга оно стабильно (средний риск 0.143 на
    валидации и 0.150 на тесте), и фиксированный порог держит бюджет: 29.9 % и
    30.5 % моментов с тревогой. А у нейросети средний риск на тесте УДВАИВАЕТСЯ
    (0.148 → 0.325), и тот же приём даёт 68.7 % вместо 30 % — система вмешивается
    в две трети спокойных моментов.

    Скользящий порог считается по собственному выходу модели за прошедшее окно и
    **не требует меток**: лаборатория приходит раз в сутки с задержкой, а квантиль
    риска известен сразу. Поэтому приём применим в эксплуатации, а не только в
    отчёте.

    ``shift(1)`` обязателен: порог для момента t считается по риску СТРОГО до t.
    Без сдвига текущее значение участвует в собственном пороге — утечка того же
    рода, что ловилась в признаках.
    """
    quantile = 1.0 - float(min(max(budget, 1e-3), 1.0))
    return (risk.rolling(f"{int(window_days)}D", min_periods=min_periods)
            .quantile(quantile).shift(1))


def interval_metrics(pred: pd.DataFrame, y: pd.Series, risk: pd.Series,
                     limit: float, alarm_threshold: float,
                     risk_thresholds: tuple[float, ...] = (0.2, 0.5)) -> dict:
    """Точность, покрытие интервала и качество тревоги. Общее для всех моделей.

    Решение оператору принимается по ВЕРОЯТНОСТИ превышения, а не по точечному
    прогнозу, поэтому precision/recall считаются на нескольких порогах.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    err = pred["q50"] - y
    half = Z90 * pred["sigma"]
    inside = ((y >= pred["q50"] - half) & (y <= pred["q50"] + half)).mean()
    over_true = y > limit

    out = {
        "n": int(len(y)),
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "coverage_80": float(inside),
        "n_over_limit": int(over_true.sum()),
        "alarm_threshold": float(alarm_threshold),
        # Доля моментов с тревогой — то, что бюджет и обещает. Без этого числа
        # нельзя заметить, что порог, выбранный на валидации, на другом периоде
        # означает другую частоту вмешательств. У нейросети на горизонте 2 ч он
        # означал 68.7 % вместо обещанных 30 %, и увидеть это было негде.
        "alarm_rate": float((risk > alarm_threshold).mean()),
    }
    for thr in tuple(risk_thresholds) + (round(float(alarm_threshold), 2),):
        alarm = risk > thr
        tp = int((alarm & over_true).sum())
        fp = int((alarm & ~over_true).sum())
        fn = int((~alarm & over_true).sum())
        out[f"precision@{thr}"] = float(tp / (tp + fp)) if tp + fp else None
        out[f"recall@{thr}"] = float(tp / (tp + fn)) if tp + fn else None

    if over_true.nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(over_true.astype(int), risk))
        out["pr_auc"] = float(average_precision_score(over_true.astype(int), risk))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    out["base_rate"] = float(over_true.mean())

    key = round(float(alarm_threshold), 2)
    out["spec_precision"] = out.get(f"precision@{key}")
    out["spec_recall"] = out.get(f"recall@{key}")
    return out


@dataclass
class SulfurModel:
    """Виртуальный анализатор серы, приведённый к лабораторной шкале."""

    horizon_hours: float = 2.0
    iterations: int = 600
    learning_rate: float = 0.05
    depth: int = 6
    seed: int = 42
    features: list[str] = field(default_factory=list)
    # калибровка ширины интервала по валидации (конформная поправка)
    sigma_scale: float = 1.0
    models: dict = field(default_factory=dict)
    # отдельный классификатор P(сера > предела): решение принимается по нему,
    # потому что вероятность из квантилей систематически занижает редкие события
    clf: object | None = None
    limit: float = 10.0
    # порог тревоги подбирается по валидации, а не берётся «на глаз»
    alarm_threshold: float = 0.5
    # калибровка Платта: сырой скор источника риска → честная вероятность
    risk_calibration: tuple[float, float] | None = None
    # почему поправка не применена, если не применена: пустая строка — применена
    risk_calibration_note: str = ""
    # чем считаем вероятность нарушения: "classifier" или "interval" (выбор по val)
    risk_source: str = "classifier"
    # можно ли вообще доверять тревоге: False, если ни один порог на валидации
    # не даёт precision заметно выше базовой частоты нарушений
    alarm_reliable: bool = True
    # порог, который выбрал бы компромисс F-beta — оставляем в отчёте для сравнения
    alarm_threshold_fbeta: float | None = None
    # зашивать ли в модель физическое направление отклика
    monotone: bool = True
    # какой показатель прогнозируем: от этого зависят знаки физики и имя файла
    target: str = "sulfur"
    # Сдвиг цели при обучении. НЕ косметика: с монотонными ограничениями CatBoost
    # теряет автоматическое начальное приближение и начинает подъём от нуля. На
    # сере (уровень 8.5) это стоило смещения −0.28 мг/кг, а на Т95 (уровень 347)
    # модель не доезжала до уровня вовсе — MAE 90 против 5.3. Учим на отклонении
    # от медианы обучающей выборки и возвращаем сдвиг при прогнозе; физика в
    # ограничениях от этого не меняется, а начальная точка перестаёт быть нулём.
    y_offset: float = 0.0
    metrics: dict = field(default_factory=dict)
    # матрица признаков на регулярной сетке; нужна, чтобы отдать прогноз по ProcessState
    feature_matrix: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None,
            top_features: int | None = None,
            must_keep: list[str] | None = None) -> "SulfurModel":
        """Обучение. ``top_features`` включает отбор признаков в два прохода.

        На 983 анализах и 354 признаках бустинг переобучается (AUC 0.80 на train
        против 0.57 на val). Первый проход нужен только чтобы измерить важность,
        второй учится на сокращённом наборе.
        """
        if top_features:
            probe = SulfurModel(iterations=self.iterations, learning_rate=self.learning_rate,
                                depth=self.depth, seed=self.seed, limit=self.limit,
                                target=self.target, monotone=self.monotone)
            probe.fit(X, y, X_val, y_val)
            imp = probe.feature_importance(top_features)
            keep = list(imp[imp > 0].index) or list(X.columns[:top_features])
            # управляющие теги оставляем принудительно: без них модель не реагирует
            # на уставки и не годится оптимизатору как суррогат «режим → качество»
            for col in (must_keep or []):
                if col in X.columns and col not in keep:
                    keep.append(col)
            X = X[keep]
            X_val = X_val[keep] if X_val is not None else None

        from catboost import CatBoostRegressor

        self.features = list(X.columns)
        constraints = self._monotone_constraints()
        # сдвиг считаем ТОЛЬКО по обучающей выборке: медиана валидации — это уже
        # подглядывание, пусть и слабое
        self.y_offset = float(np.median(y)) if constraints else 0.0
        y_fit = y - self.y_offset
        eval_set = ((X_val[self.features], y_val - self.y_offset)
                    if X_val is not None else None)

        for name, loss in [("q50", "Quantile:alpha=0.5"),
                           ("q10", "Quantile:alpha=0.1"),
                           ("q90", "Quantile:alpha=0.9")]:
            model = CatBoostRegressor(
                loss_function=loss, iterations=self.iterations,
                learning_rate=self.learning_rate, depth=self.depth,
                random_seed=self.seed, verbose=False, allow_writing_files=False,
                task_type="CPU", monotone_constraints=constraints,
            )
            model.fit(X, y_fit, eval_set=eval_set, use_best_model=eval_set is not None)
            self.models[name] = model

        self._fit_classifier(X, y, X_val, y_val)
        return self

    def _monotone_constraints(self) -> list[int] | None:
        """Направление влияния каждого признака: −1, 0 или +1.

        Ограничение накладывается только на признаки с ясной физикой; остальные
        обучаются свободно. Классификатор превышения использует те же знаки:
        что повышает серу, то повышает и риск выйти за спецификацию.
        """
        if not self.monotone:
            return None
        physics = PHYSICS_BY_TARGET.get(self.target, PHYSICS_MONOTONE)
        constraints = [physics.get(f, 0) for f in self.features]
        return constraints if any(constraints) else None

    def _fit_classifier(self, X, y, X_val=None, y_val=None) -> None:
        """Классификатор превышения спецификации. Классы несбалансированы (~15 %),
        поэтому веса балансируются, а порог тревоги подбирает оркестратор."""
        from catboost import CatBoostClassifier

        target = (y > self.limit).astype(int)
        if target.nunique() < 2:
            self.clf = None
            return
        eval_set = None
        if X_val is not None and (y_val > self.limit).nunique() > 1:
            eval_set = (X_val[self.features], (y_val > self.limit).astype(int))
        clf = CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, random_seed=self.seed, verbose=False,
            allow_writing_files=False, task_type="CPU",
            auto_class_weights="Balanced", loss_function="Logloss",
            monotone_constraints=self._monotone_constraints(),
        )
        clf.fit(X, target, eval_set=eval_set, use_best_model=eval_set is not None)
        self.clf = clf

    # ------------------------------------------------------------------ #
    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Прогноз для таблицы признаков: ``q50, q10, q90, sigma``."""
        Xf = X[self.features]
        out = pd.DataFrame({k: m.predict(Xf) + self.y_offset
                            for k, m in self.models.items()}, index=X.index)
        out["sigma"] = ((out["q90"] - out["q10"]) / (2 * Z90)).clip(lower=0.1) * self.sigma_scale
        return out

    def _raw_risk(self, X: pd.DataFrame) -> pd.Series:
        """Вероятность до поправки — от того источника, который выбран."""
        if self.clf is not None and self.risk_source == "classifier":
            return pd.Series(self.clf.predict_proba(X[self.features])[:, 1], index=X.index)
        return interval_risk(self.predict_frame(X), self.limit)

    def predict_risk(self, X: pd.DataFrame, raw: bool = False) -> pd.Series:
        """P(показатель > предела). Классификатор, если выбран, иначе — из интервала.

        Поправка Платта применяется к ЛЮБОМУ источнику, а не только к
        классификатору. Так было не всегда, и разница оказалась не теоретической:
        рабочая модель выбирает источником интервал, поправка подбиралась на
        скоре классификатора и не применялась НИКОГДА. В отчёте при этом лежало
        поле `risk_calibration`, из которого следовало обратное.

        Почему поправка нужна и интервалу. Вероятность из интервала — это
        нормальное приближение по q50 и σ. Приближение грубое: хвосты у ошибки
        прогноза тяжелее нормальных, и в середине шкалы вероятность
        систематически завышается. Оператору же показывается именно она, и по ней
        же сравнивается порог вмешательства.
        """
        score = self._raw_risk(X)
        if raw or not self.risk_calibration:
            return score
        a, b = self.risk_calibration
        clipped = np.clip(score, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped))
        return pd.Series(1 / (1 + np.exp(-(a * logit + b))), index=X.index)

    def select_risk_source(self, X_val: pd.DataFrame, y_val: pd.Series,
                           min_spread: float = 0.05) -> str:
        """Чем считать вероятность нарушения: классификатором или интервалом.

        Выбор делается на валидации по PR-AUC, но только среди источников, у которых
        вероятность вообще меняется: после калибровки неинформативный классификатор
        схлопывается в константу около базовой частоты, и решать по ней нельзя.
        """
        from sklearn.metrics import average_precision_score

        over = (y_val > self.limit).astype(int)
        if over.nunique() < 2 or self.clf is None:
            self.risk_source = "interval"
            return self.risk_source

        scores, spreads = {}, {}
        for source in ("classifier", "interval"):
            self.risk_source = source
            risk = self.predict_risk(X_val)
            scores[source] = float(average_precision_score(over, risk))
            # если вероятность почти не меняется от точки к точке, она не несёт
            # информации, каким бы ни был PR-AUC: решать по ней нельзя
            spreads[source] = float(risk.quantile(0.95) - risk.quantile(0.05))

        usable = [k for k in scores if spreads[k] >= min_spread]
        self.risk_source = (max(usable, key=scores.get) if usable
                            else max(scores, key=scores.get))
        self.risk_source_scores = scores
        self.risk_source_spreads = spreads
        if not usable:
            self.alarm_reliable = False
        return self.risk_source

    def calibrate_risk(self, X_val: pd.DataFrame, y_val: pd.Series,
                       train_base_rate: float | None = None,
                       max_base_rate_shift: float = 0.2) -> tuple[float, float]:
        """Калибровка Платта на валидации: логистическая регрессия по логиту скора.

        Вызывать ПОСЛЕ ``select_risk_source``: поправка подбирается под тот
        источник, который будет работать. В обратном порядке она настраивалась на
        классификатор, а решения принимались по интервалу.

        **Поправка применяется не всегда, и это главное в этом методе.** Платт —
        преобразование, сдвигающее УРОВЕНЬ вероятности к частоте событий на том
        окне, где он подобран. Если частота на валидации нетипична, поправка
        переносит в рабочую модель артефакт периода, а не свойство модели.

        На наших данных так и вышло: превышений на обучении 15.1 %, на валидации
        19.5 % — на треть больше. Поправка, подобранная на валидации, поднимает
        вероятность под эти 19.5 % и на следующем периоде завышает её.

        Поэтому перед подбором сравниваем частоты. Расхождение больше
        ``max_base_rate_shift`` (относительных) — поправка НЕ применяется, причина
        записывается в ``risk_calibration_note``. Правило сформулировано по train и
        val, тест в решении не участвует.
        """
        from sklearn.linear_model import LogisticRegression

        self.risk_calibration = None
        self.risk_calibration_note = ""

        over_val = float((y_val > self.limit).mean())
        if train_base_rate and over_val > 0:
            shift = abs(over_val - train_base_rate) / max(train_base_rate, 1e-9)
            if shift > max_base_rate_shift:
                self.risk_calibration_note = (
                    f"поправка не применена: частота превышений на валидации "
                    f"{over_val:.1%} против {train_base_rate:.1%} на обучении "
                    f"(расхождение {shift:.0%}), подгонка уровня перенесла бы "
                    f"свойство периода, а не модели")
                return (1.0, 0.0)

        score = self.predict_risk(X_val, raw=True).to_numpy()
        score = np.clip(score, 1e-6, 1 - 1e-6)
        logit = np.log(score / (1 - score)).reshape(-1, 1)
        target = (y_val > self.limit).astype(int).to_numpy()
        if len(np.unique(target)) < 2:
            return (1.0, 0.0)
        lr = LogisticRegression(max_iter=1000).fit(logit, target)
        self.risk_calibration = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
        return self.risk_calibration

    def risk_for_state(self, state: ProcessState) -> float | None:
        row = self._row_for(state)
        return None if row is None else float(self.predict_risk(row).iloc[0])

    def select_alarm_threshold(self, X_val: pd.DataFrame, y_val: pd.Series,
                               beta: float = 1.5, min_lift: float = 1.5,
                               budget: float | None = None) -> float:
        """Порог вероятности, при котором объявляем риск нарушения.

        Два требования одновременно:
        * максимум F-beta (beta=1.5 — пропуск некондиции дороже ложной тревоги,
          но не настолько, чтобы включать сигнализацию постоянно);
        * тревога обязана быть информативной: precision не ниже ``min_lift`` от
          базовой частоты нарушений. Без этого условия оптимум вырождается в
          «тревога всегда», а ТЗ прямо требует не создавать лишних воздействий
          в устойчивом режиме.
        """
        # Если ни один порог не даёт информативной тревоги, «тревога всегда» —
        # худший из вариантов: она нарушает требование ТЗ не создавать лишних
        # воздействий. Тогда pick_alarm_threshold возвращает прозрачное правило
        # «сигнал только если сам прогноз выше предела» (P > 0.5 по интервалу).
        risk = self.predict_risk(X_val).to_numpy()
        threshold, reliable = pick_alarm_threshold(
            risk, (y_val > self.limit).to_numpy(), beta=beta, min_lift=min_lift)
        self.alarm_threshold_fbeta = threshold
        self.alarm_reliable = reliable
        # бюджет тревог, если задан, важнее компромисса F-beta: рабочая точка —
        # решение технолога, а не модели
        self.alarm_threshold = (threshold if budget is None
                                else pick_threshold_for_budget(risk, budget))
        return self.alarm_threshold

    def discrimination(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Насколько вероятность вообще информативна: ROC-AUC и PR-AUC."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        over = (y > self.limit).astype(int)
        if over.nunique() < 2:
            return {"roc_auc": None, "pr_auc": None, "base_rate": float(over.mean())}
        risk = self.predict_risk(X)
        return {"roc_auc": float(roc_auc_score(over, risk)),
                "pr_auc": float(average_precision_score(over, risk)),
                "base_rate": float(over.mean())}

    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series, target: float = 0.8) -> float:
        """Конформная поправка ширины интервала по валидации.

        Квантильные бустинги систематически дают слишком узкий интервал на новых
        данных. Берём распределение нормированной ошибки |y - q50| / σ и растягиваем
        σ так, чтобы покрытие на валидации совпало с номинальным. Оркестратор
        считает риск по σ, поэтому её честность важнее точности точечного прогноза.
        """
        self.sigma_scale = 1.0
        self.sigma_scale = conformal_sigma_scale(self.predict_frame(X_val), y_val, target)
        return self.sigma_scale

    def _row_for(self, state: ProcessState) -> pd.DataFrame | None:
        """Строка признаков, доступная на момент состояния (строго ≤ ts)."""
        if self.feature_matrix is None:
            raise RuntimeError("модель не привязана к матрице признаков: "
                               "SulfurModel.attach(features)")
        idx = self.feature_matrix.index
        pos = idx.searchsorted(pd.Timestamp(state.ts), side="right") - 1
        if pos < 0:
            return None
        row = self.feature_matrix.iloc[[pos]]
        # Признаки, которых в матрице нет по построению: они зависят не от истории,
        # а от МОМЕНТА ЗАПРОСА. Возраст строки признаков при обучении считается от
        # времени анализа, и такого столбца в матрице быть не может — его надо
        # досчитать здесь, иначе обучение и работа разъедутся.
        missing = [f for f in self.features if f in SERVE_COMPUTED]
        if missing:
            row = row.copy()
            row["feature_age_h"] = (
                (pd.Timestamp(state.ts) - idx[pos]).total_seconds() / 3600.0)
        return row

    def predict_with_sigma(self, state: ProcessState) -> tuple[float, float]:
        """Интерфейс для ``QualityAgent``: ``(среднее, σ)`` на момент состояния."""
        row = self._row_for(state)
        if row is None:
            return float("nan"), float("inf")
        pred = self.predict_frame(row)
        return float(pred["q50"].iloc[0]), float(pred["sigma"].iloc[0])

    def attach(self, feature_matrix: pd.DataFrame) -> "SulfurModel":
        """Привязка к рабочей матрице признаков.

        Раньше здесь стояло ``feature_matrix[self.features]``, и это была мина.
        Отбору признаков при обучении предлагается ``feature_age_h``, которого в
        рабочей матрице НЕТ по построению: он считается от времени анализа. Пока
        отбор его не выбирал, всё работало; при следующем переобучении он мог
        попасть в набор, и модель падала бы на загрузке — в момент, когда её уже
        подключили к циклу.
        """
        stored = [f for f in self.features if f not in SERVE_COMPUTED]
        missing = [f for f in stored if f not in feature_matrix.columns]
        if missing:
            raise RuntimeError(
                f"в рабочей матрице нет признаков модели: {missing}. "
                "Матрица и модель собраны по разным настройкам — переобучите "
                "модель или пересоберите матрицу.")
        self.feature_matrix = feature_matrix[stored]
        return self

    # ------------------------------------------------------------------ #
    def evaluate(self, X: pd.DataFrame, y: pd.Series, limit: float = 10.0,
                 risk_thresholds: tuple[float, ...] = (0.2, 0.5)) -> dict:
        """Точность, калибровка интервала и качество решения «есть риск / нет риска».

        Решение оператору принимается по ВЕРОЯТНОСТИ превышения, а не по точечному
        прогнозу, поэтому precision/recall считаем на нескольких порогах тревоги:
        при 0.5 модель почти всегда молчит, рабочий порог заметно ниже.
        """
        risk = self.predict_risk(X)
        out = {"risk_source": self.risk_source, "alarm_reliable": self.alarm_reliable}
        out.update(interval_metrics(self.predict_frame(X), y, risk,
                                    limit, self.alarm_threshold, risk_thresholds))
        out.update(probability_metrics(risk, (y > limit).astype(int)))
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def default_path(horizon_hours: float, target: str = "sulfur") -> Path:
        """Модели разных горизонтов и показателей не перетирают друг друга."""
        stem = "sulfur" if target == "sulfur" else target
        return MODELS_DIR / f"{stem}_h{horizon_hours:g}"

    def save(self, path: Path | None = None) -> Path:
        path = path or self.default_path(self.horizon_hours, self.target)
        path.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            model.save_model(str(path / f"{name}.cbm"))
        if self.clf is not None:
            self.clf.save_model(str(path / "risk.cbm"))
        meta = {"alarm_threshold_fbeta": self.alarm_threshold_fbeta,
                "horizon_hours": self.horizon_hours, "features": self.features,
                "metrics": self.metrics, "seed": self.seed,
                "sigma_scale": self.sigma_scale,
                "alarm_threshold": self.alarm_threshold, "limit": self.limit,
                "monotone": self.monotone, "target": self.target,
                "y_offset": self.y_offset,
                "risk_calibration": list(self.risk_calibration) if self.risk_calibration else None,
                "risk_calibration_note": self.risk_calibration_note,
                "risk_source": self.risk_source, "alarm_reliable": self.alarm_reliable}
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "SulfurModel":
        from catboost import CatBoostRegressor

        path = Path(path) if path else MODELS_DIR / "sulfur_h0"
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        obj = cls(horizon_hours=meta["horizon_hours"], seed=meta["seed"],
                  features=meta["features"], metrics=meta.get("metrics", {}),
                  sigma_scale=meta.get("sigma_scale", 1.0),
                  alarm_threshold=meta.get("alarm_threshold", 0.5),
                  limit=meta.get("limit", 10.0),
                  risk_calibration=tuple(meta["risk_calibration"])
                  if meta.get("risk_calibration") else None,
                  risk_calibration_note=meta.get("risk_calibration_note", ""),
                  risk_source=meta.get("risk_source", "classifier"),
                  alarm_reliable=meta.get("alarm_reliable", True),
                  alarm_threshold_fbeta=meta.get("alarm_threshold_fbeta"),
                  monotone=meta.get("monotone", True),
                  target=meta.get("target", "sulfur"),
                  y_offset=meta.get("y_offset", 0.0))
        for name in ("q50", "q10", "q90"):
            model = CatBoostRegressor()
            model.load_model(str(path / f"{name}.cbm"))
            obj.models[name] = model
        risk_path = path / "risk.cbm"
        if risk_path.exists():
            from catboost import CatBoostClassifier
            clf = CatBoostClassifier()
            clf.load_model(str(risk_path))
            obj.clf = clf
        return obj

    # ------------------------------------------------------------------ #
    def feature_importance(self, top: int = 20) -> pd.Series:
        imp = self.models["q50"].get_feature_importance()
        return pd.Series(imp, index=self.features).sort_values(ascending=False).head(top)


def _normal_cdf(z):
    from scipy.special import ndtr
    return ndtr(z)


def probability_metrics(risk: pd.Series, over: pd.Series, bins: int = 5) -> dict:
    """Честна ли вероятность как ВЕРОЯТНОСТЬ, а не как порядок.

    ROC-AUC и PR-AUC меряют различение: умение упорядочить моменты по опасности.
    Модель может прекрасно ранжировать и при этом систематически завышать
    вероятность вдвое — по этим двум метрикам не видно ничего. А оркестратор
    сравнивает вероятность с порогом, и бюджет тревог задан в тех же единицах, то
    есть всё решающее правило стоит на предположении, что 0.2 означает «примерно
    один случай из пяти».

    * ``brier`` — средний квадрат ошибки вероятности;
    * ``brier_base`` — то же у «всегда базовая частота». Проигрыш константе
      означает, что как вероятность выход использовать нельзя;
    * ``ece`` — средний по бинам разрыв «заявлено против наблюдалось»;
    * ``calibration_shift`` — перекос уровня: средняя заявленная минус фактическая
      частота. Именно он уезжает, когда меняется частота событий.
    """
    if not len(risk) or over.nunique() < 2:
        return {}
    base = float(over.mean())
    frame = pd.DataFrame({"p": risk.to_numpy(), "y": over.to_numpy()})
    try:
        frame["bin"] = pd.qcut(frame["p"], bins, duplicates="drop")
        grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"),
                                                          p=("p", "mean"),
                                                          y=("y", "mean"))
        ece = float((grouped["n"] / grouped["n"].sum()
                     * (grouped["y"] - grouped["p"]).abs()).sum())
    except ValueError:
        ece = float("nan")
    return {
        "brier": float(((frame["p"] - frame["y"]) ** 2).mean()),
        "brier_base": float(((base - frame["y"]) ** 2).mean()),
        "ece": None if ece != ece else ece,
        "calibration_shift": float(frame["p"].mean() - base),
    }


def baseline_metrics(pred: pd.Series, y: pd.Series, limit: float = 10.0) -> dict:
    """Метрики базовой «модели» без интервала — для честного сравнения."""
    common = pred.dropna().index.intersection(y.index)
    p, t = pred.loc[common], y.loc[common]
    err = p - t
    alarm = p > limit
    over = t > limit
    tp, fp, fn = int((alarm & over).sum()), int((alarm & ~over).sum()), int((~alarm & over).sum())
    return {
        "n": int(len(t)),
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
        "coverage_80": None,
        "n_over_limit": int(over.sum()),
        "spec_precision": float(tp / (tp + fp)) if tp + fp else None,
        "spec_recall": float(tp / (tp + fn)) if tp + fn else None,
    }


def make_model_surrogate(model: "SulfurModel", tag_prefix: str = "ht_"):
    """Суррогат «режим → качество» поверх обученной модели.

    Берёт строку признаков на момент состояния, подставляет уставки кандидата и
    ПЕРЕСЧИТЫВАЕТ производные признаки режима (WABT, кратность газ/сырьё,
    кинетический индекс). Без пересчёта подмена одной колонки бессмысленна:
    модель опирается на сочетания, и они остались бы от старого режима.
    """
    from nefte.models.regime import apply_moves_to_rows

    def _fn(state: ProcessState, moves: dict[str, float]) -> dict[str, float]:
        row = model._row_for(state)
        if row is None:
            return {"product_sulfur_mgkg": float("nan")}
        row = apply_moves_to_rows(row.copy(), moves, prefix=tag_prefix)
        for tag, value in moves.items():          # уставки АВТ, если такие есть
            col = f"avt_{tag}"
            if col in row.columns and f"{tag_prefix}{tag}" not in row.columns:
                row[col] = value
        return {"product_sulfur_mgkg": float(model.predict_frame(row)["q50"].iloc[0])}

    return _fn


def controllable_features(model: "SulfurModel", tags: list[str]) -> list[str]:
    """Какие из управляющих тегов реально входят в модель (для честного отчёта)."""
    cols = set(model.features)
    return [t for t in tags if {f"ht_{t}", f"avt_{t}", t} & cols]
