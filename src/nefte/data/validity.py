"""Достоверность сигналов: детекторы как код, а не как разовые находки.

В телеметрии формально нет пропусков (0 % NaN), но брак закодирован значениями-
заглушками, «полками» и физически невозможными величинами. Этот модуль считает
маски брака один раз для всей истории, а затем мгновенно отвечает на вопрос
«что было недостоверно в момент t» — именно этот ответ попадает в объяснение
оператору.

Про отрицательные расходы отдельно. Нельзя просто взять теги на букву F: для
установки 24-2000 буква имени не соответствует смыслу. Но нельзя и слепо верить
описанию: у части тегов описание говорит «расход», а значения ведут себя как
температура. Поэтому тег считается принципиально неотрицательным, только если
описание говорит о расходе И доля отрицательных значений в истории мала. Теги с
описанием «расход» и массовой отрицательностью (F5 — 93 %, F26 — 62 %) не
исправляются, а помечаются как подозрительные: это сигнал другой природы либо
датчик с обратным знаком, и решать это должен технолог.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nefte.config import load_config
from nefte.data.cleaning import SENTINELS, frozen_mask
from nefte.data.loaders import load_tag_dictionary

# Доля отрицательных значений, выше которой тег не «чиним», а помечаем как
# подозрительный: это уже не единичный сбой, а систематика.
SUSPICIOUS_NEGATIVE_SHARE = 0.05


@dataclass
class SignalValidity:
    """Маски недостоверности по каждому тегу и времени + очищенные данные."""

    unit: str
    clean: pd.DataFrame
    sentinel: pd.DataFrame
    frozen: pd.DataFrame
    negative: pd.DataFrame
    dead_tags: list[str] = field(default_factory=list)
    suspicious_tags: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, raw: pd.DataFrame, unit: str = "avt",
              cfg: dict | None = None) -> "SignalValidity":
        cfg = cfg or load_config()
        tel = cfg["telemetry"]
        sentinels = tuple(tel.get("sentinel_values", SENTINELS))
        min_frozen = int(tel.get("frozen_min_samples", 18))

        df = raw.copy()

        # 1. мёртвые теги: сигнала нет вовсе (D10 — 307 в 189 207 из 189 217 точек)
        dead = [c for c in df.columns
                if df[c].nunique(dropna=True) <= 2
                and df[c].isin(sentinels).mean() > 0.5]
        for c in cfg["telemetry"].get("dead_tags", []):
            if c in df.columns and c not in dead:
                dead.append(c)

        # 2. значения-заглушки системы сбора
        sentinel = pd.DataFrame(False, index=df.index, columns=df.columns)
        for v in sentinels:
            sentinel |= df.eq(v)
        df = df.mask(sentinel)

        # 3. «полки»: значение не менялось дольше порога
        frozen = pd.DataFrame(
            {c: frozen_mask(df[c], min_frozen) for c in df.columns}, index=df.index)
        df = df.mask(frozen)

        # 4. физически невозможные отрицательные расходы. Решение принимается по
        #    ОБУЧАЮЩЕМУ периоду: «этот тег бывает отрицательным» — такое же
        #    правило, выведенное из данных, как нормировка severity, и выводить
        #    его по всей истории значит смотреть в тестовый период.
        train = tuple(cfg["split"]["train"]) if "split" in cfg else None
        nonneg, suspicious = cls._nonnegative_tags(df, unit, train)
        negative = pd.DataFrame(False, index=df.index, columns=df.columns)
        if nonneg:
            negative[nonneg] = df[nonneg] < 0
            df = df.mask(negative)

        return cls(unit=unit, clean=df, sentinel=sentinel, frozen=frozen,
                   negative=negative, dead_tags=dead, suspicious_tags=suspicious)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _nonnegative_tags(df: pd.DataFrame, unit: str,
                          train: tuple[str, str] | None = None
                          ) -> tuple[list[str], dict[str, float]]:
        """Расходные теги, у которых отрицательное значение — это брак.

        ``train`` ограничивает период, по которому считается доля отрицательных.
        """
        try:
            tags = load_tag_dictionary()
        except (OSError, KeyError, ValueError):
            # справочника нет или он в другом формате — не гадаем, а признаём,
            # что физического смысла тегов не знаем. Ловим именно эти ошибки:
            # широкий except прятал бы и наши собственные опечатки.
            return [], {}

        flows = {row.code for row in tags[tags["unit"] == unit].itertuples()
                 if "асход" in str(row.description)}
        scope = df.loc[train[0]:train[1]] if train else df
        if not len(scope):
            scope = df
        nonneg, suspicious = [], {}
        for tag in flows & set(df.columns):
            share = float((scope[tag] < 0).mean())
            if share <= SUSPICIOUS_NEGATIVE_SHARE:
                nonneg.append(tag)
            else:
                suspicious[tag] = round(share, 3)
        return sorted(nonneg), dict(sorted(suspicious.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------ #
    def flags_at(self, ts: pd.Timestamp) -> dict[str, list[str]]:
        """Что именно недостоверно в момент ``ts``: тег → причины."""
        ts = pd.Timestamp(ts)
        out: dict[str, list[str]] = {}
        for name, mask in (("заглушка", self.sentinel), ("полка", self.frozen),
                           ("отрицательный расход", self.negative)):
            sub = mask.loc[:ts]
            if sub.empty:
                continue
            row = sub.iloc[-1]
            for tag in row[row].index:
                out.setdefault(tag, []).append(name)
        for tag in self.dead_tags:
            out.setdefault(tag, []).append("датчик не даёт сигнала")
        return out

    def summary(self) -> pd.DataFrame:
        """Доля забракованных отсчётов по тегам, % — для отчёта и документации."""
        n = max(len(self.clean), 1)
        out = pd.DataFrame({
            "заглушка_%": self.sentinel.sum() / n * 100,
            "полка_%": self.frozen.sum() / n * 100,
            "отриц_%": self.negative.sum() / n * 100,
        }).round(2)
        out["итого_%"] = out.sum(axis=1).round(2)
        out["мёртвый"] = out.index.isin(self.dead_tags)
        return out.sort_values("итого_%", ascending=False)


def analyzer_health(series: pd.Series, cfg: dict | None = None) -> pd.DataFrame:
    """Периоды отказа поточного анализатора: ``start, end, value, hours``.

    Отдельная функция, потому что зависание ПАК — не «шум в теге», а событие,
    которое обязано попасть в объяснение оператору: 2026-04-15…27 прибор
    показывал 18.4 мг/кг при лабораторных 3.4.
    """
    from nefte.data.cleaning import frozen_intervals

    cfg = cfg or load_config()
    return frozen_intervals(series, int(cfg["telemetry"]["frozen_min_samples"]))
