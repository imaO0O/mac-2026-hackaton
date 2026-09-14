"""Схемы ЭЛОУ-АВТ-6 с кодами тегов против справочника КИП и данных. Только CPU.

    python scripts/check_avt_schemes.py

14.09 пришли три листа тех же схем, что `АВТ_схемы.pdf`, но с вписанными от руки
кодами тегов: колонна К-1 с печью П-1/1, атмосферная К-2 со стриппингами К-6, К-7,
К-9 и вакуумная К-10 с печью П-3. Схемы гидроочистки 24-2000 среди них нет.

Почерк — не данные, поэтому каждое прочтение проверяется двумя независимыми
способами:

1. **справочник КИП**: описание тега обязано говорить о том же аппарате и потоке,
   где код стоит на схеме (ключевое слово из описания записано рядом с прочтением);
2. **топология**: материальный баланс колонны по кодам с листа обязан сходиться, а
   два прибора на одном трубопроводе — идти вместе.

Отдельно — теги, чьи коды совпадают с кодами 24-2000, но означают другое.

Результат: таблицы в консоли и reports/avt_schemes.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.data.loaders import load_tag_dictionary, load_telemetry  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "avt_schemes.json"
STUBS = [307.0, 313.0, 240.0]

# Выделение на схеме. В легенде исходного `АВТ_схемы.pdf` синие и зелёные кружки —
# «управляемые переменные»; жёлтые и оранжевые в легенде не объяснены.
MANAGED, YELLOW, ORANGE, PLAIN = "управляемая", "жёлтый", "оранжевый", ""

# (тег, лист, где на схеме, прибор, выделение, слово из описания КИП)
SCHEME = [
    # --- К-1, печь П-1/1 ---
    ("P2", "К-1", "верх К-1", "PIRC", MANAGED, "верха К1"),
    ("T1", "К-1", "верх К-1", "TI", PLAIN, "верха К1"),
    ("F3", "К-1", "орошение К-1 из Е-1", "FC", PLAIN, "орошение К1"),
    ("F8", "К-1", "обессоленная нефть, 1-й ход", "FI", PLAIN, "1-й ход"),
    ("F9", "К-1", "обессоленная нефть, 2-й ход", "FI", PLAIN, "2-й ход"),
    ("F7", "К-1", "обессоленная нефть, 3-й ход", "FI", PLAIN, "3-й ход"),
    ("D10", "К-1", "нефть после ЭЛОУ", "D", PLAIN, "Плотность"),
    ("F5", "К-1", "пар в низ К-1", "FC", PLAIN, "пара в К1"),
    ("P4", "К-1", "низ К-1", "PC", PLAIN, "низа К1"),
    ("T6", "К-1", "низ К-1, горячая струя от П-1/1", "TC", MANAGED, "низ К1"),
    # --- К-2, стриппинги К-6, К-7, К-9, печи П-1/2, П-1/3 ---
    ("P22", "К-2", "верх К-2", "PIC", MANAGED, "верха К2"),
    ("P67", "К-2", "верх К-2", "PI", PLAIN, "верха К2"),
    ("P21", "К-2", "верх К-2", "PI", PLAIN, "верха К2"),
    ("P23", "К-2", "верх К-2", "PI", PLAIN, "верха К-2"),
    ("T20", "К-2", "верх К-2", "TI", PLAIN, "верха К2"),
    ("F19", "К-2", "острое орошение К-2", "FC", MANAGED, "орошения К2"),
    ("F16", "К-2", "нафта на блок стабилизации", "FC", PLAIN, "бензина от Н4"),
    ("T13", "К-2", "1-е ЦО", "TI", YELLOW, "1-е ЦО"),
    ("F14", "К-2", "1-е ЦО", "FC", MANAGED, "1 ЦО"),
    ("T18", "К-2", "1-е ЦО из К-2", "TI", PLAIN, "1-е ЦО"),
    ("F12", "К-2", "2-е ЦО", "FC", MANAGED, "2 ЦО"),
    ("T17", "К-2", "2-е ЦО из К-2", "TI", PLAIN, "2-е ЦО"),
    ("T11", "К-2", "3-е ЦО", "TC", PLAIN, "3-е ЦО"),
    ("F64", "К-2", "3-е ЦО", "FC", MANAGED, "3-го ЦО"),
    ("T15", "К-2", "3-е ЦО из К-2", "TI", PLAIN, "ЦО К-2"),
    ("F65", "К-2", "сырьё К-2 от печей П-1/2, П-1/3", "FI", MANAGED, "Производительность К-2"),
    ("T33", "К-2", "низ К-2", "TI", MANAGED, "низа К-2"),
    ("T24", "К-2", "переток в К-6", "TI", ORANGE, "в К-6"),
    ("F26", "К-2", "пар в К-6", "FC", PLAIN, "пара в К6"),
    ("F25", "К-2", "фр. 120-180 °C (бензин) из К-6", "FC", MANAGED, "бензина после Т26"),
    ("F27", "К-2", "пар в К-7", "FC", PLAIN, "пара в К7"),
    ("F34", "К-2", "фр. 150-250 °C (керосин) из К-7", "FC", MANAGED, "фр.150-250"),
    ("T66", "К-2", "переток в К-9", "TI", ORANGE, "в К-9"),
    ("F28", "К-2", "пар в К-9", "FC", PLAIN, "пара в К9"),
    ("F32", "К-2", "фр. 240-350 °C из К-9", "FC", MANAGED, "фр.240-290"),
    ("F30", "К-2", "ДТ, нижний отбор", "FC", MANAGED, "фр.290-350"),
    ("W70", "К-2", "ДТ, нижний отбор", "W", PLAIN, "фр.290-350"),
    ("T71", "К-2", "нижний отбор из К-2", "TI", ORANGE, "Фр.290-350"),
    ("F29", "К-2", "пар в низ К-2", "FC", PLAIN, "пара в К2"),
    ("F31", "К-2", "мазут из низа К-2 в П-3", "FI", PLAIN, "в П3"),
    # --- К-10, печь П-3 ---
    ("F56", "К-10", "фр. до 350 °C с установки", "FC", MANAGED, "до 350"),
    ("F57", "К-10", "фр. до 350 °C с установки", "FC", MANAGED, "до 350"),
    ("P50", "К-10", "вакуум верха К-10", "PI", YELLOW, "верха К10"),
    ("P51", "К-10", "вакуум верха К-10", "PI", YELLOW, "верха К-10"),
    ("F35", "К-10", "ВЦО в К-10", "FC", PLAIN, "ВЦО в К-10"),
    ("T49", "К-10", "верх К-10", "TC", MANAGED, "верха К10"),
    ("T37", "К-10", "ВЦО в К-10", "TC", MANAGED, "ВЦО в К-10"),
    ("F36", "К-10", "СЦО в К-10", "FC", PLAIN, "СЦО"),
    ("T58", "К-10", "фр. до 350 °C после Т-37", "TC", PLAIN, "до 350"),
    ("F41", "К-10", "доп. ВЦО (на листе написано «T4»)", "FC", PLAIN, "доп. ВЦО"),
    ("T39", "К-10", "К-10 над насадкой", "TC", PLAIN, "в К10"),
    ("T40", "К-10", "ВЦО из К-10", "TI", ORANGE, "ВЦО из К10"),
    ("T38", "К-10", "доп. ВЦО", "TI", PLAIN, "доп. ВЦО"),
    ("T61", "К-10", "НЦО в К-10", "TI", PLAIN, "НЦО"),
    ("F62", "К-10", "НЦО", "FC", PLAIN, "НЦО"),
    ("F46", "К-10", "НЦО", "FC", PLAIN, "НЦО"),
    ("T47", "К-10", "НЦО после холодильника", "TC", PLAIN, "350-560"),
    ("T42", "К-10", "отбор НЦО к Н-25", "TI", ORANGE, "420-500"),
    ("F59", "К-10", "фр. 350-560 °C с установки", "FC", MANAGED, "420-500"),
    ("F60", "К-10", "фр. 350-560 °C с установки", "FC", MANAGED, "350-560"),
    ("F53", "К-10", "НЦО от Н-25", "FC", PLAIN, "НЦО"),
    ("P52", "К-10", "перепад на насадке", "PD", PLAIN, "Перепад давления"),
    ("F54", "К-10", "циркуляция через теплообменники", "FC", PLAIN, "Н-34"),
    ("T55", "К-10", "выход печи П-3 (мазут от К-2)", "TC", PLAIN, "печи П3"),
    ("F45", "К-10", "линия Т-7/2 в низ К-10", "FC", PLAIN, "гудрона в К10"),
    ("P44", "К-10", "низ К-10", "PI", PLAIN, "низа К10"),
    ("T48", "К-10", "низ К-10", "TI", ORANGE, "низа К10"),
    ("L43", "К-10", "уровень в К-10", "LC", PLAIN, "Уровень в К10"),
    ("F63", "К-10", "гудрон на УПБ", "FI", PLAIN, "битумную"),
    ("F68", "К-10", "гудрон с установки", "FI", PLAIN, "гудрона с установки"),
    ("F69", "К-10", "гудрон с установки", "FI", PLAIN, "гудрона с установки"),
]

# Кандидаты в управляющие АВТ, записанные в configs/config.yaml → controls.avt до схем.
CONFIG_CANDIDATES = ["T55", "T1", "T20", "P22", "F29", "F30", "F32"]


def _norm(text: str) -> str:
    return re.sub(r"[\s\-–]+", "", str(text)).lower()


def _ratio(run: pd.DataFrame, top: list[str], bottom: list[str]) -> dict:
    num = run[top].sum(axis=1, min_count=len(top))
    den = run[bottom].sum(axis=1, min_count=len(bottom))
    ok = num.notna() & den.notna() & (den > 0)
    r = (num[ok] / den[ok])
    return {"числитель": "+".join(top), "знаменатель": "+".join(bottom),
            "медиана": round(float(r.median()), 3),
            "p05": round(float(r.quantile(0.05)), 3), "p95": round(float(r.quantile(0.95)), 3),
            "corr": round(float(num[ok].corr(den[ok])), 3), "отсчётов": int(ok.sum())}


def main() -> int:
    use_utf8_console()
    tags = load_tag_dictionary()
    kip_avt = tags[tags["unit"] == "avt"].set_index("code")["description"].astype(str)
    kip_ht = tags[tags["unit"] == "ht"].set_index("code")["description"].astype(str)
    avt = load_telemetry("avt")
    ht_columns = set(load_telemetry("ht").columns)
    avt = avt.mask(avt.isin(STUBS))
    # работающая установка: есть сырьё К-2 и горячая печь П-3
    run = avt[(avt["F65"] > 500) & (avt["T55"] > 300)]

    rows = []
    for tag, sheet, place, instrument, mark, word in SCHEME:
        description = kip_avt.get(tag, "")
        in_data = tag in avt.columns
        series = run[tag].dropna() if in_data else pd.Series(dtype=float)
        rows.append({
            "тег": tag, "лист": sheet, "на схеме": place, "прибор": instrument,
            "выделение": mark, "описание КИП": description,
            "описание сходится": bool(description) and _norm(word) in _norm(description),
            "в данных": in_data,
            "медиана": None if series.empty else round(float(series.median()), 2),
        })
    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 48)
    print("\nКоды со схем против справочника КИП (медиана — на работающей установке)\n")
    print(frame[["тег", "лист", "на схеме", "прибор", "выделение", "описание сходится",
                 "медиана"]].to_string(index=False))
    mismatched = frame[~frame["описание сходится"]]["тег"].tolist()
    print(f"\nПрочтений: {len(frame)}; описание КИП не сходится: {mismatched or 'ни у одного'}")

    balances = {
        "К-2: отборы / сырьё": _ratio(run, ["F16", "F25", "F34", "F32", "F30", "F31"], ["F65"]),
        "К-10: отборы / мазут в П-3": _ratio(run, ["F57", "F59", "F63", "F68", "F69"], ["F31"]),
        "К-1: сырьё К-2 / три хода нефти": _ratio(run, ["F65"], ["F7", "F8", "F9"]),
        "дубль: F56 / F57 (фр. до 350)": _ratio(run, ["F56"], ["F57"]),
        "дубль: F59 / F60 (фр. 350-560)": _ratio(run, ["F59"], ["F60"]),
        "массовый / объёмный: W70 / F30": _ratio(run, ["W70"], ["F30"]),
    }
    print("\nТопология по данным:")
    for name, b in balances.items():
        print(f"  {name:36s} медиана {b['медиана']:.3f} (p05 {b['p05']:.3f} … p95 {b['p95']:.3f}), "
              f"corr {b['corr']:.3f}")

    namesakes = [{"тег": t, "АВТ": kip_avt.get(t, ""), "24-2000": kip_ht.get(t, "")}
                 for t in sorted(set(avt.columns) & ht_columns)]
    print("\nОдин код — две разные величины (АВТ и 24-2000):")
    for n in namesakes:
        print(f"  {n['тег']:4s} АВТ: {n['АВТ'][:45]:45s} | 24-2000: {n['24-2000'][:50]}")

    managed = frame[frame["выделение"] == MANAGED]["тег"].tolist()
    controllers = {t: kip_avt.get(t, "") for t in kip_avt.index if "Выход на FIRC" in kip_avt.get(t, "")}
    print(f"\nВыделены на схемах как управляемые: {', '.join(managed)}")
    print("Температуры с каскадом на расход («Выход на FIRC…» в КИП): "
          + ", ".join(controllers))
    print("Кандидаты из configs/config.yaml → controls.avt:")
    for t in CONFIG_CANDIDATES:
        status = ("выделен на схеме" if t in managed else
                  "регулятор температуры по КИП" if t in controllers else "не выделен")
        print(f"  {t}: {status}")

    REPORT.write_text(json.dumps({
        "источник": "три листа схем ЭЛОУ-АВТ-6 с кодами тегов, 14.09",
        "прочтения": rows, "не сходится с КИП": mismatched,
        "топология": balances, "совпадающие_коды": namesakes,
        "выделены_как_управляемые": managed,
        "регуляторы_температуры": controllers,
        "кандидаты_конфига": {t: (t in managed, t in controllers) for t in CONFIG_CANDIDATES},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
