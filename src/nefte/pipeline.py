"""Сборка среза состояния процесса (ProcessState) на произвольный момент времени.

Здесь живёт единственное место, где источники соединяются друг с другом.
Соединение — только по времени и только «назад», возраст каждого значения
сохраняется. Никакой агент не обращается к сырым файлам напрямую.
"""
from __future__ import annotations

import pandas as pd

from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source
from nefte.config import load_config
from nefte.data.cleaning import (
    clean_lims_distillation,
    clean_lims_sulfur,
    flat_mask,
    frozen_mask,
)
from nefte.data.features import known_from
from nefte.models.dataset import PCT_TO_MGKG
from nefte.data.loaders import lims_series, load_lims, load_pak, load_telemetry
from nefte.data.validity import SignalValidity

# Доля пропусков в срезе, после которой решение принимать нельзя. ДОПУЩЕНИЕ:
# порог выбран нами, в пакете такого требования нет.
MAX_MISSING_SHARE = 0.2


def is_state_usable(missing_share: float, pak_is_frozen: bool,
                    lims_age_hours: float | None, cfg: dict | None = None) -> bool:
    """Пригоден ли срез для принятия решения.

    Правило одно на всю систему: срез непригоден, если в нём слишком много
    пропусков или если оба источника качества молчат — поточный анализатор завис,
    а лабораторный результат старше трёх нормативных сроков. Вынесено из
    ``StateBuilder.build``, чтобы «злые» сценарии проверялись тем же правилом,
    что и рабочий цикл, а не его копией.
    """
    cfg = cfg or load_config()
    stale = cfg["quality"]["staleness_hours"]
    if missing_share >= MAX_MISSING_SHARE:
        return False
    return not (pak_is_frozen and lims_age_hours is not None
                and lims_age_hours > stale["lims"] * 3)


# Имена установок в списках тегов. Код без установки ничего не значит: восемь кодов
# есть и на АВТ, и на 24-2000 с разным смыслом (docs/AVT_SCHEMES.md §4) — `F19` на
# АВТ это орошение К-2, на гидроочистке расход сырья.
UNIT_NAMES = {"avt": "АВТ", "ht": "24-2000"}
LISTED_PER_UNIT = 5


def flagged_tags(avt_flags: dict[str, list[str]], ht_flags: dict[str, list[str]],
                 reason: str) -> list[str]:
    """Теги с причиной брака ``reason`` в виде «установка:код».

    Раньше флаги двух установок сливались в один словарь по голому коду, и при
    браке одноимённых тегов на обеих установках причина АВТ перезаписывалась.
    """
    return sorted([f"avt:{t}" for t, r in avt_flags.items() if reason in r]
                  + [f"ht:{t}" for t, r in ht_flags.items() if reason in r])


def tag_listing(tags: list[str]) -> str:
    """«АВТ: D10, F12, F19; 24-2000: F2» — по установкам, не больше пяти на каждую."""
    parts = []
    for unit, name in UNIT_NAMES.items():
        codes = [t.split(":", 1)[1] for t in tags if t.startswith(f"{unit}:")]
        if not codes:
            continue
        shown = ", ".join(codes[:LISTED_PER_UNIT])
        rest = len(codes) - LISTED_PER_UNIT
        parts.append(f"{name}: {shown}" + (f" и ещё {rest}" if rest > 0 else ""))
    return "; ".join(parts)


class StateBuilder:
    """Готовит данные один раз, затем быстро отдаёт срез на любой момент."""

    def __init__(self, cfg: dict | None = None, avt_tags: list[str] | None = None,
                 ht_tags: list[str] | None = None, mask_frozen: bool = True):
        self.cfg = cfg or load_config()

        # Маски недостоверности считаются один раз на всю историю: дальше срез на
        # любой момент отдаётся мгновенно, вместе с причинами брака по каждому тегу.
        self.avt_validity = SignalValidity.build(
            load_telemetry("avt", avt_tags), unit="avt", cfg=self.cfg)
        self.ht_validity = SignalValidity.build(
            load_telemetry("ht", ht_tags), unit="ht", cfg=self.cfg)
        self.avt = self.avt_validity.clean
        self.ht = self.ht_validity.clean

        pak = load_pak()
        self.pak_sulfur = pak["sulfur_ppm"]
        # «Замороженный» поточный анализатор — отдельный флаг, а не удаление данных:
        # оператору важно видеть, что прибор врёт (случай 2026-04-15…27).
        self.pak_frozen = frozen_mask(
            self.pak_sulfur, int(self.cfg["telemetry"]["frozen_min_samples"]))
        # Второй поточный анализатор серы — тег Q21 телеметрии 24-2000. В листе
        # «КИП» пакета его описание стояло на другом коде, поэтому прибор считался
        # отсутствующим («дубликата анализатора нет»). Таблица организаторов 15.09
        # вернула ему смысл: с лабораторией он сходится лучше файла ПАК.
        # Заглушка 307 снимается общим детектором достоверности, «полка» — тем же
        # правилом, что у ПАК.
        self.q21_sulfur = (self.ht["Q21"].dropna() if "Q21" in self.ht.columns
                           else pd.Series(dtype=float))
        # Полка Q21 — не одно число, а 24.88 ± 0.04: точное равенство прежнего
        # детектора её не видело вовсе (0 % точек), и система две недели подряд
        # считала неисправный прибор живым оперативным значением. Детектор с
        # допуском, порог — правилом до счёта (scripts/check_q21_shelf.py).
        tolerance = (self.cfg.get("quality") or {}).get("q21_frozen_tolerance_mgkg")
        min_samples = int(self.cfg["telemetry"]["frozen_min_samples"])
        if not len(self.q21_sulfur):
            self.q21_frozen = pd.Series(dtype=bool)
        elif tolerance is None:
            self.q21_frozen = frozen_mask(self.q21_sulfur, min_samples)
        else:
            self.q21_frozen = flat_mask(self.q21_sulfur, min_samples, float(tolerance))

        lims = load_lims()
        # Факт: когда проба отобрана. С ним сверяются прогоны и бэктесты.
        self.lims_sulfur = clean_lims_sulfur(
            lims_series(self.cfg["quality"]["target"]["lims_source"], lims))
        # Вход решения: когда результат стал известен оператору. Разница — до
        # четырёх часов, и без неё система смотрит в будущее.
        delay = float(self.cfg["quality"].get("lims_publication_delay_hours", 0.0))
        self.lims_delay_hours = delay
        self.lims_sulfur_known = known_from(self.lims_sulfur, delay)
        # сера сырья гидроочистки: нужна кинетическому суррогату как «вход» реакции
        try:
            feed = lims_series("Гидроочистка|1|Mass.Sulfur", lims)
            # сера сырья — тоже лабораторный анализ, и публикуется так же поздно
            # % масс. → мг/кг, той же константой, что и в матрице признаков:
            # два места, где делается одно преобразование, обязаны ссылаться на
            # одно число, иначе они разъедутся.
            self.lims_feed_sulfur = known_from(feed[feed > 0] * PCT_TO_MGKG, delay)
        except KeyError:
            self.lims_feed_sulfur = None
        # Т95 — второй обязательный показатель по ответу организаторов. Он нужен не
        # для отчётности: исправленная формула ВАК даёт Т95 с коэффициентом 0.50 по
        # температуре Р-202, то есть КАЖДОЕ повышение температуры ради серы тянет
        # Т95 вверх. Без этого ряда оптимизатор не знает, что чинит одно за счёт
        # другого.
        try:
            t95 = clean_lims_distillation(lims_series("Гидроочистка|2|95%.T", lims))
            self.lims_t95 = t95
            self.lims_t95_known = known_from(t95, delay)
        except KeyError:
            self.lims_t95 = self.lims_t95_known = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _last(series: pd.Series, ts: pd.Timestamp) -> tuple[float | None, float | None]:
        """Последнее ИЗМЕРЕННОЕ значение до ``ts`` и его возраст в часах.

        Пустая проба — не измерение. Без отсева строка ЛИМС без значения выглядела
        бы свежим анализом с возрастом ноль: флаг устаревания молчал бы, а
        уверенность считалась бы по несуществующему числу. На выданных данных таких
        строк нет (0 из 873 в `_last_lims`), но правило должно стоять в коде, а не
        держаться на удаче — тесты `tests/test_last_measurement.py`.
        """
        sub = series.loc[:ts].dropna()
        if sub.empty:
            return None, None
        age_h = (ts - sub.index[-1]).total_seconds() / 3600.0
        return float(sub.iloc[-1]), float(age_h)

    def _last_lims(self, series: pd.Series,
                   ts: pd.Timestamp) -> tuple[float | None, float | None]:
        """То же для лабораторного ряда, но возраст — ОТ ОТБОРА ПРОБЫ.

        Ряды ЛИМС хранятся сдвинутыми на задержку публикации: в срез попадает
        только то, что оператор мог увидеть. Поэтому возраст «от последней
        известной точки» занижен ровно на задержку, и складывать её обратно
        обязаны все три ряда одинаково — иначе один порог означает разное для
        серы и для Т95.

        Такое расхождение здесь уже было: возраст Т95 показывался от отбора, а
        флаг устаревания считался от публикации. В 19.2 % срезов тестового
        периода оператор видел две строки ЛИМС с ОДНИМ возрастом 26 ч и ОДНИМ
        порогом 24 ч, из которых одна помечена устаревшей, а другая нет.
        """
        value, age_h = self._last(series, ts)
        if age_h is not None:
            age_h += self.lims_delay_hours
        return value, age_h

    def build(self, ts: str | pd.Timestamp) -> ProcessState:
        ts = pd.Timestamp(ts)
        stale = self.cfg["quality"]["staleness_hours"]

        avt_row = self.avt.loc[:ts].iloc[-1] if len(self.avt.loc[:ts]) else pd.Series(dtype=float)
        ht_row = self.ht.loc[:ts].iloc[-1] if len(self.ht.loc[:ts]) else pd.Series(dtype=float)

        # в срез идёт ТОЛЬКО опубликованное значение, а возраст считается от
        # ОТБОРА пробы: оператору важно, насколько старая проба, а не когда её
        # напечатали. Одной функцией для всех трёх лабораторных рядов — чтобы
        # порог означал для них одно и то же.
        lims_val, lims_age = self._last_lims(self.lims_sulfur_known, ts)
        pak_val, pak_age = self._last(self.pak_sulfur, ts)
        # зависание файлового ряда — теперь для примечания; в правило пригодности идёт
        # зависание ОПЕРАТИВНОГО прибора (см. ниже, operational_frozen)
        pak_is_frozen = bool(self.pak_frozen.loc[:ts].iloc[-1]) if len(
            self.pak_frozen.loc[:ts]) else False

        quality = {}
        if lims_val is not None:
            quality["lims_sulfur_mgkg"] = Measurement(
                value=lims_val, unit="мг/кг", source=Source.LIMS, age_hours=lims_age,
                is_stale=lims_age > stale["lims"])
        if self.lims_feed_sulfur is not None:
            feed_val, feed_age = self._last_lims(self.lims_feed_sulfur, ts)
            if feed_val is not None:
                # Порог для сырья — СВОЙ, а не кратный порогу продукта. Продукт
                # анализируют раз в сутки (95 % промежутков ровно 24 ч), сырьё —
                # по совсем другому расписанию: медиана 48 ч, 90-й процентиль
                # 715 ч. Прежний множитель ×7 зажигал флаг в 56 % срезов, и в
                # 36 % это была ЕДИНСТВЕННАЯ жалоба в stale_sources — то есть
                # список «чему нельзя верить» горел из-за нормального расписания.
                quality["lims_feed_sulfur_mgkg"] = Measurement(
                    value=feed_val, unit="мг/кг", source=Source.LIMS, age_hours=feed_age,
                    is_stale=feed_age > stale["lims_feed"],
                    comment="сера сырья гидроочистки")
        if self.lims_t95_known is not None:
            t95_val, t95_age = self._last_lims(self.lims_t95_known, ts)
            if t95_val is not None:
                quality["lims_t95_c"] = Measurement(
                    value=t95_val, unit="°C", source=Source.LIMS, age_hours=t95_age,
                    is_stale=t95_age > stale["lims"],
                    comment="Т95 продукта гидроочистки")
        if len(self.q21_sulfur):
            q21_val, q21_age = self._last(self.q21_sulfur, ts)
            q21_is_frozen = bool(self.q21_frozen.loc[:ts].iloc[-1]) if len(
                self.q21_frozen.loc[:ts]) else False
            if q21_val is not None:
                quality["q21_sulfur_ppm"] = Measurement(
                    value=q21_val, unit="мг/кг", source=Source.PAK, age_hours=q21_age,
                    is_stale=q21_age > stale["pak"], is_frozen=q21_is_frozen,
                    comment="поточный анализатор Q21 (телеметрия 24-2000)")
        if pak_val is not None:
            quality["pak_sulfur_ppm"] = Measurement(
                value=pak_val, unit="мг/кг", source=Source.PAK, age_hours=pak_age,
                is_stale=pak_age > stale["pak"], is_frozen=pak_is_frozen,
                comment="сигнал не менялся дольше порога" if pak_is_frozen else "")

        row = pd.concat([avt_row, ht_row])
        missing_share = float(row.isna().mean()) if len(row) else 1.0

        # причины брака по каждому тегу на этот момент — это и есть объяснение,
        # почему часть данных не используется
        avt_flags, ht_flags = self.avt_validity.flags_at(ts), self.ht_validity.flags_at(ts)
        frozen_tags = flagged_tags(avt_flags, ht_flags, "полка")
        sentinel_tags = flagged_tags(avt_flags, ht_flags, "заглушка")

        # Оперативный прибор — тот, по которому решает система (analyzer_source).
        # До 21.09 и примечание, и правило пригодности смотрели только на ряд из
        # файла ПАК: после перехода на Q21 неисправный Q21 не упоминался вовсе, а
        # зависший файловый ряд объявлял непригодным срез, который решался по Q21.
        # С детектором полки (flat_mask) расхождение двух правил сжалось с 9.7 %
        # часов теста до 0.4 %, и пригодность переведена на оперативный прибор;
        # правило приёмки на пересобранных прогонах — docs/PLAN.md, до счёта.
        operational_q21 = str(self.cfg["quality"].get("analyzer_source", "pak")) == "q21"
        q21_measure = quality.get("q21_sulfur_ppm")
        q21_frozen_now = bool(q21_measure is not None and q21_measure.is_frozen)
        operational_frozen = q21_frozen_now if (operational_q21 and q21_measure) else pak_is_frozen
        backup_frozen = pak_is_frozen if (operational_q21 and q21_measure) else False

        notes = []
        if operational_frozen:
            name = "Q21" if (operational_q21 and q21_measure) else "ПАК"
            notes.append(f"Поточный анализатор серы заморожен ({name}, оперативный) — "
                         f"значение не является фактом.")
        if backup_frozen:
            notes.append("Второй поточный анализатор (ряд ПАК из файла) заморожен.")
        if lims_age is not None and lims_age > stale["lims"]:
            notes.append(f"Последний анализ ЛИМС старше {stale['lims']} ч ({lims_age:.0f} ч).")
        if sentinel_tags:
            notes.append(f"Значения-заглушки в тегах: {tag_listing(sentinel_tags)}.")
        if frozen_tags:
            notes.append(f"Сигнал не меняется в тегах: {tag_listing(frozen_tags)}.")

        dq = DataQuality(
            missing_share=missing_share,
            frozen_tags=((["pak_sulfur"] if pak_is_frozen else [])
                         + (["q21_sulfur"] if q21_frozen_now else []) + sorted(frozen_tags)),
            sentinel_tags=sorted(sentinel_tags),
            stale_sources=[k for k, m in quality.items() if m.is_stale],
            usable=is_state_usable(missing_share, operational_frozen, lims_age, self.cfg),
            notes=notes,
        )

        return ProcessState(
            ts=ts,
            telemetry_avt={k: (None if pd.isna(v) else float(v)) for k, v in avt_row.items()},
            telemetry_ht={k: (None if pd.isna(v) else float(v)) for k, v in ht_row.items()},
            quality=quality,
            data_quality=dq,
        )

    # ------------------------------------------------------------------ #
    def model_bounds(self, tags: list[str], unit: str = "ht",
                     train_only: bool = True) -> dict[str, tuple[float, float]]:
        """Модельные диапазоны уставок = квантили обучающего периода.

        ЭТО ДОПУЩЕНИЕ, а не промышленный предел (правило границ из ТЗ).
        """
        df = self.ht if unit == "ht" else self.avt
        if train_only:
            lo, hi = self.cfg["split"]["train"]
            df = df.loc[lo:hi]
        q_lo, q_hi = self.cfg["limits"]["quantiles"]
        out = {}
        for tag in tags:
            if tag in df.columns:
                out[tag] = (float(df[tag].quantile(q_lo)), float(df[tag].quantile(q_hi)))
        return out
