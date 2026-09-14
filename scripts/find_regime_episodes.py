"""Эпизоды «режим менялся не в ответ на серу» — для оценки отклика (участник 2). CPU.

    python scripts/find_regime_episodes.py

Отклик серы на режим — главное допущение оптимизатора, и по истории его в лоб не
выделить: оператор поднимает температуру, УВИДЕВ рост серы, и корреляция смешивает
отклик процесса с обратной связью. Нужны изменения режима, причиной которых сера
не была. Здесь они собираются в список — модуль и критерий в
``src/nefte/data/episodes.py``, разбор в ``docs/REGIME_EPISODES.md``.

Три типа:

1. смены катализатора — по шагу нормированной температуры (``models/catalyst.py``);
2. выходы на режим после остановов — первые трое суток после пуска;
3. ступеньки температуры, нагрузки и давления вне пусков — с меткой, насколько
   вероятно, что это НЕ реакция на серу, и флагом «чистый» для тех, где
   одновременно не менялось ничего другого и ПАК после ступеньки работал.

Результат: reports/regime_episodes.json и сводка в консоли.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import clean_lims_sulfur, frozen_mask  # noqa: E402
from nefte.data.episodes import (  # noqa: E402
    CLEAN_LABELS,
    CONCURRENT_SHARE,
    NAMES,
    THRESHOLDS,
    hourly_regime,
    outage_intervals,
    regime_step_episodes,
    startup_episodes,
)
from nefte.data.loaders import lims_series, load_pak, load_telemetry  # noqa: E402
from nefte.models.catalyst import (  # noqa: E402
    build_observations,
    long_outages,
    outage_steps,
)
from nefte.models.regime import FEED  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402


def jsonable(value):
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    spec = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    delay = pd.Timedelta(hours=float(cfg["quality"]["lims_publication_delay_hours"]))

    ht = load_telemetry("ht")
    hourly = hourly_regime(ht)

    pak = load_pak()["sulfur_ppm"]
    pak = pak.where(~frozen_mask(pak, int(cfg["telemetry"]["frozen_min_samples"])))
    pak_hourly = pak.resample("1h").mean().reindex(hourly.index)

    lab = clean_lims_sulfur(lims_series(cfg["quality"]["target"]["lims_source"]))
    # Оператор видит анализ, когда он опубликован, а не когда отобрана проба.
    published = pd.Series(lab.values, index=lab.index + delay).sort_index()

    # --- 1. смены катализатора ------------------------------------------ #
    feed_sulfur = lims_series("Гидроочистка|1|Mass.Sulfur") * 1e4
    observations = build_observations(ht, lab, feed_sulfur)
    steps = outage_steps(observations, long_outages(ht[FEED]))
    changes = [row for row in steps if row["смена катализатора"]]
    change_marks = [pd.Timestamp(row["конец"]) for row in changes]
    print("\n[1] Смены катализатора (шаг нормированной температуры через останов)\n")
    for row in changes:
        print(f"  пуск {row['конец']}: NWABT {row['NWABT до, °C']} → "
              f"{row['NWABT после, °C']} °C (шаг {row['шаг, °C']:+.1f})")

    # --- 2. выходы на режим ------------------------------------------- #
    outages = outage_intervals(ht[FEED], 6.0)
    startups = startup_episodes(hourly, outages)
    print(f"\n[2] Выходы на режим после остановов длиннее 6 ч: {len(startups)}\n")
    for _, row in startups.iterrows():
        first, later = row["первые 6 ч"], row["через 3 сут"]
        tail = ("  (установка снова встала в окне)" if row["повторный останов в окне"]
                else "")
        print(f"  {str(row['момент'])[:16]}  простой {row['простой, ч']:.0f} ч; "
              f"WABT {first.get('wabt')} → {later.get('wabt')} °C, "
              f"нагрузка {first.get('feed')} → {later.get('feed')} м3/ч{tail}")

    # --- 3. ступеньки режима -------------------------------------------- #
    episodes = regime_step_episodes(hourly, pak_hourly, published, spec,
                                    [end for _, end in outages], change_marks)
    band = episodes.attrs["noise_band"]
    print(f"\n[3] Ступеньки режима вне пусков и смен катализатора: {len(episodes)}")
    print(f"    шумовая полоса наклона ПАК за 12 ч: «ровно» ≤ {band.flat:.2f}, "
          f"«явно» > {band.clear:.2f} мг/кг\n")
    table = (episodes.groupby(["переменная", "метка"]).size().unstack(fill_value=0)
             .rename(index=NAMES))
    print(table.to_string())

    clean = episodes[episodes["чистый"]]
    print(f"\n  Чистых эпизодов: {len(clean)} — метка «{CLEAN_LABELS[0]}» или "
          f"«{CLEAN_LABELS[1]}», ничто другое не менялось больше чем на "
          f"{CONCURRENT_SHARE:.0%} своего порога, ПАК после ступеньки работал.")
    by = clean.groupby(["переменная", "метка"]).size().unstack(fill_value=0).rename(index=NAMES)
    print(by.to_string())

    # Проверка критерия: если ступеньки температуры в сумме — реакция на серу, а
    # нагрузки и давления — нет, перевес «по направлению» у температуры обязан быть
    # больше, чем у остальных. Это и есть основание, на котором критерий стоит.
    print("\n  Перевес «похоже на реакцию» над «против тренда серы» по переменным:")
    balance = {}
    for variable in THRESHOLDS:
        part = episodes[episodes["переменная"] == variable]
        reaction = int((part["метка"] == "похоже на реакцию").sum())
        against = int((part["метка"] == "против тренда серы").sum())
        balance[variable] = {"похоже на реакцию": reaction, "против тренда серы": against}
        ratio = reaction / against if against else float("inf")
        print(f"    {NAMES[variable]:32s} {reaction:4d} против {against:4d} "
              f"(в {ratio:.1f} раза)")

    report = {
        "критерий": {
            "окно ступеньки, ч": 6, "разнесение, ч": 24, "пороги": THRESHOLDS,
            "окно пуска, ч": 72, "отступ от смены катализатора, сут": 7,
            "сопутствующее изменение, доля порога": CONCURRENT_SHARE,
            "наклон серы, ч": 12,
            "шумовая полоса": {"ровно": round(band.flat, 3), "явно": round(band.clear, 3)},
            "чистые метки": list(CLEAN_LABELS),
        },
        "смены_катализатора": changes,
        "выходы_на_режим": jsonable(startups.to_dict("records")),
        "ступеньки": {
            "всего": int(len(episodes)),
            "по_меткам": {NAMES[v]: episodes[episodes["переменная"] == v]["метка"]
                          .value_counts().to_dict() for v in THRESHOLDS},
            "перевес_реакции": balance,
            "чистых": int(len(clean)),
            "чистые_по_переменным": clean["переменная"].map(NAMES).value_counts().to_dict(),
            "чистые_по_разбиению": {
                name: int(((clean["момент"] >= pd.Timestamp(bounds[0]))
                           & (clean["момент"] < pd.Timestamp(bounds[1])
                              + pd.Timedelta("1D"))).sum())
                for name, bounds in cfg["split"].items()
                if isinstance(bounds, (list, tuple))},
        },
        "эпизоды": jsonable(episodes.to_dict("records")),
    }
    out = ROOT / "reports" / "regime_episodes.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
