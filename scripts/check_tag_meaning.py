"""Что на самом деле измеряют теги установки 24-2000.

    python scripts/check_tag_meaning.py

Организаторы подтвердили, что короткие имена колонок в `242000_tags.csv`
соответствуют тегам на листе «КИП», и назвали управляющие переменные так:
температура ГСС на входе Р-202 — `P8`, расход сырья массовый — `T11`, давление на
входе Р-202 — `F19`.

**Два кода из трёх на выданных данных не сходятся**, и молчать об этом нельзя:
если строить управление на неверном опознании величин, физика рекомендаций теряет
смысл. Скрипт собирает доказательства по трём независимым признакам:

1. **порядок величины** — температура реактора гидроочистки живёт в районе
   300–400 °C, давление 3–5 МПа, расход сотни единиц. Значение 0.17 не может быть
   температурой ни в каких единицах;
2. **корреляция с лабораторией** — тег, описанный как поточный анализатор серы,
   обязан коррелировать с лабораторным анализом серы. Если корреляции нет, это не
   анализатор;
3. **отклик на останов** — реакторная температура на остановленной установке
   падает до десятков градусов, расход уходит в ноль. Величины ведут себя
   по-разному, и по этому их тоже можно различить.

**Поправка к первой версии этого скрипта, и она существенная.** Сначала мы
заключили по признаку 2, что теги опознаны неверно вообще: ни один не коррелирует
с лабораторной серой сильнее 0.14. Вывод был слишком сильным. Лаборатория меряет
раз в сутки, и на такой сетке отклик режима не виден в принципе — отсутствие
корреляции с ЛИМС не улика. Проверка по ПОТОЧНОМУ анализатору
(`scripts/find_delays.py`, шаг 10 минут) дала обратное: у всех семи управляющих
тегов знак реакции совпадает с физикой, |corr| до 0.49 на первых разностях. То
есть реакторные температуры и расходы опознаны ВЕРНО, а несходящимися остались
конкретные коды `P8` и `F19` из ответа организаторов.

Признак 2 в таблице остаётся — он по-прежнему показывает, что теги, описанные как
поточные анализаторы серы, ими не являются. Но общий вывод про опознание величин
теперь опирается на отклик, а не на лабораторную корреляцию.

Результат: reports/tag_meaning.json и таблица с вердиктом по каждому тегу.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import clean_lims_sulfur  # noqa: E402
from nefte.data.loaders import (  # noqa: E402
    lims_series,
    load_pak,
    load_tag_dictionary,
    load_telemetry,
)
from nefte.models.regime import FEED, outage_mask  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

# Теги, которые организаторы назвали управляющими, плюс те, что мы используем
# сейчас, плюс те, чьи описания говорят «анализатор серы».
UNDER_QUESTION = ["P8", "T11", "F19", "T5", "T6", "P13", "W7", "W10", "F26",
                  "F15", "P24", "F2", "T12", "T23"]

# Физические диапазоны, по которым величину можно узнать по порядку.
PLAUSIBLE = {
    "температура реактора, °C": (250.0, 420.0),
    "давление реактора, МПа": (2.0, 6.0),
    "перепад давления, кгс/см2": (0.3, 6.0),
    "сера в ДТ, мг/кг": (1.0, 40.0),
    "расход, сотни ед.": (50.0, 700.0),
}


def guess_by_magnitude(median: float) -> str:
    """Что это может быть по порядку величины."""
    hits = [name for name, (lo, hi) in PLAUSIBLE.items() if lo <= median <= hi]
    return ", ".join(hits) if hits else "ни одна физическая величина не подходит"


def main() -> int:
    use_utf8_console()
    cfg = load_config()

    ht = load_telemetry("ht")
    tags = load_tag_dictionary()
    described = dict(zip(tags[tags["unit"] == "ht"]["code"],
                         tags[tags["unit"] == "ht"]["description"]))

    lab = clean_lims_sulfur(lims_series(cfg["quality"]["target"]["lims_source"]))
    pak = load_pak()["sulfur_ppm"].reindex(ht.index)
    # ретроспективный разбор: нужен весь эпизод останова, а не то, что было
    # известно в моменте
    down = (outage_mask(ht[FEED], retrospective=True) if FEED in ht.columns
            else pd.Series(False, index=ht.index))

    # лабораторный анализ сопоставляем с последним предшествующим отсчётом
    positions = ht.index.searchsorted(lab.index, side="right") - 1
    valid = positions >= 0
    lab_values = lab.to_numpy()[valid]

    rows = []
    for code in UNDER_QUESTION:
        if code not in ht.columns:
            continue
        series = ht[code]
        median = float(series.median())

        matched = series.to_numpy()[positions[valid]]
        mask = ~np.isnan(matched)
        corr_lab = (float(np.corrcoef(matched[mask], lab_values[mask])[0, 1])
                    if mask.sum() > 50 else float("nan"))

        both = series.notna() & pak.notna()
        corr_pak = (float(np.corrcoef(series[both], pak[both])[0, 1])
                    if both.sum() > 1000 else float("nan"))

        working, stopped = series[~down], series[down]
        drop = (float(stopped.median() / working.median())
                if len(stopped) and working.median() else float("nan"))

        description = str(described.get(code, ""))
        says_analyzer = "анализатор" in description.lower()
        verdict = guess_by_magnitude(median)
        if says_analyzer and abs(corr_lab) < 0.3:
            verdict += "; описан как анализатор серы, но с лабораторией не связан"

        rows.append({
            "тег": code,
            "медиана": round(median, 2),
            "corr с ЛИМС": round(corr_lab, 3),
            "corr с ПАК": round(corr_pak, 3),
            "доля на останове": None if drop != drop else round(drop, 2),
            "описание КИП": description[:46],
            "что это по данным": verdict,
        })

    frame = pd.DataFrame(rows)
    print("\nЧто измеряют теги 24-2000 — по данным, а не по описанию\n")
    print(frame[["тег", "медиана", "corr с ЛИМС", "доля на останове",
                 "описание КИП"]].to_string(index=False))

    print("\nВердикт по каждому:")
    for row in rows:
        print(f"  {row['тег']:4s} {row['что это по данным']}")

    named = {"P8": "температура ГСС на входе Р-202", "T11": "расход сырья массовый",
             "F19": "давление на входе Р-202"}
    print("\nУправляющие переменные, названные организаторами:")
    for code, meaning in named.items():
        row = next((r for r in rows if r["тег"] == code), None)
        if row is None:
            continue
        fits = meaning.split()[0][:4].lower()
        print(f"  {code}: заявлено «{meaning}», медиана {row['медиана']} — "
              f"{'сходится' if fits in row['что это по данным'].lower() else 'НЕ СХОДИТСЯ'}")

    print("\nВывод, и читать его надо вместе с scripts/find_delays.py.")
    print("  1. Слабая связь с ЛАБОРАТОРИЕЙ сама по себе ничего не доказывает. "
          "Лаборатория меряет серу примерно раз в сутки, а отклик на изменение "
          "режима держится часами — на такой сетке его не видно в принципе. "
          "Корреляция ниже 0.14 с ЛИМС — не улика против тегов.")
    print("  2. На ПОТОЧНОМ анализаторе (шаг 10 минут) связь есть, и она "
          "физически правильная: у всех семи управляющих тегов знак реакции "
          "совпал с ожидаемым — рост температуры снижает серу, рост расхода "
          "сырья повышает, |corr| до 0.49 на первых разностях. Значит, "
          "реакторные температуры и расходы опознаны ВЕРНО.")
    print("  3. Несходящимися остались конкретные коды. Медиана P8 равна 0.17: "
          "температурой ГСС такое значение не является ни в каких единицах. "
          "Три тега описаны на листе КИП как поточные анализаторы серы и с "
          "лабораторией не связаны.")
    print("  Отсюда вопрос организаторам: управляющими считать ФИЗИЧЕСКИЕ "
          "величины (температура реактора, расход сырья, давление) или именно "
          "коды P8/T11/F19? Мы работаем по величинам, опознанным по значениям и "
          "по знаку отклика, и это стоит подтвердить.")

    out = ROOT / "reports" / "tag_meaning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tags": rows, "named_by_organizers": named},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
