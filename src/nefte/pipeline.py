"""Сборка среза состояния процесса (ProcessState) на произвольный момент времени.

Здесь живёт единственное место, где источники соединяются друг с другом.
Соединение — только по времени и только «назад», возраст каждого значения
сохраняется. Никакой агент не обращается к сырым файлам напрямую.
"""
from __future__ import annotations

import pandas as pd

from nefte.agents.schemas import DataQuality, Measurement, ProcessState, Source
from nefte.config import load_config
from nefte.data.cleaning import clean_lims_sulfur, frozen_mask
from nefte.data.loaders import lims_series, load_lims, load_pak, load_telemetry
from nefte.data.validity import SignalValidity


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

        lims = load_lims()
        self.lims_sulfur = clean_lims_sulfur(
            lims_series(self.cfg["quality"]["target"]["lims_source"], lims))
        # сера сырья гидроочистки: нужна кинетическому суррогату как «вход» реакции
        try:
            feed = lims_series("Гидроочистка|1|Mass.Sulfur", lims)
            self.lims_feed_sulfur = feed[feed > 0] * 10_000        # % масс. → мг/кг
        except KeyError:
            self.lims_feed_sulfur = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _last(series: pd.Series, ts: pd.Timestamp) -> tuple[float | None, float | None]:
        """Последнее значение до ``ts`` и его возраст в часах."""
        sub = series.loc[:ts]
        if sub.empty:
            return None, None
        age_h = (ts - sub.index[-1]).total_seconds() / 3600.0
        return float(sub.iloc[-1]), float(age_h)

    def build(self, ts: str | pd.Timestamp) -> ProcessState:
        ts = pd.Timestamp(ts)
        stale = self.cfg["quality"]["staleness_hours"]

        avt_row = self.avt.loc[:ts].iloc[-1] if len(self.avt.loc[:ts]) else pd.Series(dtype=float)
        ht_row = self.ht.loc[:ts].iloc[-1] if len(self.ht.loc[:ts]) else pd.Series(dtype=float)

        lims_val, lims_age = self._last(self.lims_sulfur, ts)
        pak_val, pak_age = self._last(self.pak_sulfur, ts)
        pak_is_frozen = bool(self.pak_frozen.loc[:ts].iloc[-1]) if len(
            self.pak_frozen.loc[:ts]) else False

        quality = {}
        if lims_val is not None:
            quality["lims_sulfur_mgkg"] = Measurement(
                value=lims_val, unit="мг/кг", source=Source.LIMS, age_hours=lims_age,
                is_stale=lims_age > stale["lims"])
        if self.lims_feed_sulfur is not None:
            feed_val, feed_age = self._last(self.lims_feed_sulfur, ts)
            if feed_val is not None:
                quality["lims_feed_sulfur_mgkg"] = Measurement(
                    value=feed_val, unit="мг/кг", source=Source.LIMS, age_hours=feed_age,
                    is_stale=feed_age > stale["lims"] * 7,
                    comment="сера сырья гидроочистки")
        if pak_val is not None:
            quality["pak_sulfur_ppm"] = Measurement(
                value=pak_val, unit="мг/кг", source=Source.PAK, age_hours=pak_age,
                is_stale=pak_age > stale["pak"], is_frozen=pak_is_frozen,
                comment="сигнал не менялся дольше порога" if pak_is_frozen else "")

        row = pd.concat([avt_row, ht_row])
        missing_share = float(row.isna().mean()) if len(row) else 1.0

        # причины брака по каждому тегу на этот момент — это и есть объяснение,
        # почему часть данных не используется
        flags = {**self.avt_validity.flags_at(ts), **self.ht_validity.flags_at(ts)}
        frozen_tags = [t for t, r in flags.items() if "полка" in r]
        sentinel_tags = [t for t, r in flags.items() if "заглушка" in r]

        notes = []
        if pak_is_frozen:
            notes.append("Поточный анализатор серы заморожен — значение не является фактом.")
        if lims_age is not None and lims_age > stale["lims"]:
            notes.append(f"Последний анализ ЛИМС старше {stale['lims']} ч ({lims_age:.0f} ч).")
        if sentinel_tags:
            notes.append(f"Значения-заглушки в тегах: {', '.join(sorted(sentinel_tags)[:5])}.")
        if frozen_tags:
            notes.append(f"Сигнал не меняется в тегах: {', '.join(sorted(frozen_tags)[:5])}.")

        dq = DataQuality(
            missing_share=missing_share,
            frozen_tags=(["pak_sulfur"] if pak_is_frozen else []) + sorted(frozen_tags),
            sentinel_tags=sorted(sentinel_tags),
            stale_sources=[k for k, m in quality.items() if m.is_stale],
            usable=missing_share < 0.2 and not (pak_is_frozen and lims_age is not None
                                                and lims_age > stale["lims"] * 3),
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
