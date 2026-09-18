"""Блок надёжности для дашборда: содержание и числа (участник 2, план 14.09, п. 5).

Вёрстку делает участник 1. Здесь — ЧТО показывать и с какими оговорками: одна
функция без Streamlit, чтобы содержание проверялось тестами, а не глазами.

Два раздела.

* **Тяжесть режима** — индекс, класс, что класс запрещает, и шесть факторов с
  вкладом каждого. Вклад — вес × значение / сумма весов доступных факторов: так
  вклады складываются ровно в индекс (до обрезки в [0, 1]), и оператор видит не
  «severity 0.62», а из чего она сложилась.
* **Катализатор** — сутки цикла, запас до уровня вывода, остаточный ресурс и
  контрольная точка, числа из ``reports/catalyst_life.json``. Линейная оценка
  ресурса показывается только в первой половине цикла: бэктест на цикле с
  известным исходом (docs/CATALYST_LIFE.md §5) меряет её ошибку +11 % на 108-х
  сутках и +74 % на 300-х, и всегда в оптимистичную сторону.

Чего блок не делает: не показывает оценку ресурса для прошлых циклов — она
опиралась бы на данные после выбранного момента. Для текущего цикла оценка
считается по срезу данных, и если выбранный момент раньше среза, это сказано
флагом, а не спрятано.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from nefte.agents.reliability import ReliabilityAgent
from nefte.agents.schemas import ReliabilityAssessment
from nefte.models.catalyst import CHECKPOINT_LEVEL_MARGIN_C, CHECKPOINT_MAX_RATE_C_PER_MONTH

# ключ фактора → (название, что это) — для подписи и подсказки на карточке
FACTOR_LABELS = {
    "wabt": ("WABT реакторного блока",
             "средняя температура T5, T6, T11 в долях диапазона обучающего периода"),
    "dp_r202": ("перепад давления на Р-202",
                "в долях диапазона обучающего периода"),
    "anomaly": ("нетипичность режима",
                "расстояние до обычных режимов в долях порога детектора аномалий"),
    "ramp": ("скорость изменения режима",
             ("худший из двух каналов за час — реакторные температуры и расход сырья; "
              "выше 0.8 подъём температур запрещён до стабилизации")),
    "catalyst": ("износ катализатора", ""),
    "furnace": ("температура на выходе печи П-3",
                "в долях диапазона обучающего периода"),
}
CATALYST_BASIS = {
    "age": "наработка с последнего сброса в долях p95 обучающего периода — возраст, "
           "а не состояние",
    "activity": "уровень WABT, приведённой к нагрузке, за 30 суток в долях p05…p95 "
                "обучающего периода",
}
# Перепад Р-202 по режиму агента (reliability.dp_factor): подпись фактора и почему
# он может быть выключен. Выключенный — не «нет данных»: его не потеряли, его не
# считают намеренно, и оператор должен это видеть.
DP_MEANING = {
    "level": "уровень P8 в долях диапазона обучающего периода: меряет гидравлику и "
             "плотность загрузки, а не износ (docs/DP_PROXY.md)",
    "growth": "прирост P8 от начала цикла при той же нагрузке, в долях p95 обучающего "
              "периода",
}
DISABLED_WHY = {"dp_r202": "выключен: на выданных данных перепад не растёт с наработкой, "
                           "а уровнем переводит свежую загрузку в тяжёлый режим "
                           "(docs/DP_PROXY.md)"}
CLASS_TEXT = {
    "low": "мягкий режим: дополнительных ограничений нет",
    "medium": "средний режим: шаг вверх по реакторным температурам не больше 1 °C",
    "high": "тяжёлый режим: повышать реакторные температуры нельзя",
}
EOR_SOURCE = ("уровень вывода взят из истории установки (два завершённых цикла), "
              "норматива в пакете нет")


def load_catalyst_report(root: Path) -> dict | None:
    path = Path(root) / "reports" / "catalyst_life.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def catalyst_basis(agent: ReliabilityAgent) -> str:
    """Каким рядом агент мерит износ — от этого зависит подпись фактора."""
    return "activity" if getattr(agent, "catalyst_series", None) is not None else "age"


def dp_mode(agent: ReliabilityAgent) -> str:
    """Как агент учитывает перепад Р-202: off, level или growth."""
    return getattr(agent, "dp_factor", "level")


def _finite(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


# --------------------------------------------------------------------------- #
# тяжесть режима
# --------------------------------------------------------------------------- #

def severity_section(assessment: ReliabilityAssessment, basis: str = "age",
                     weights: dict[str, float] | None = None, dp: str = "off") -> dict:
    weights = weights or ReliabilityAgent.WEIGHTS
    disabled = ["dp_r202"] if dp == "off" else []
    present = {k: float(v) for k, v in assessment.factors.items()
               if k in weights and k not in disabled and _finite(v)}
    total = sum(weights[k] for k in present)
    rows = []
    for key, value in present.items():
        label, meaning = FACTOR_LABELS.get(key, (key, ""))
        if key == "catalyst":
            meaning = CATALYST_BASIS.get(basis, "")
        if key == "dp_r202":
            meaning = DP_MEANING.get(dp, meaning)
        rows.append({"фактор": key, "название": label, "что это": meaning,
                     "значение": round(value, 3), "вес": weights[key],
                     "вклад": round(weights[key] * value / total, 3) if total else None})
    rows.sort(key=lambda row: -(row["вклад"] or 0.0))
    return {
        "индекс": round(float(assessment.severity_index), 3),
        "класс": assessment.risk_class,
        "что значит класс": CLASS_TEXT[assessment.risk_class],
        "главный фактор": rows[0]["название"] if rows else None,
        "факторы": rows,
        "нет данных": [FACTOR_LABELS.get(k, (k, ""))[0] for k in weights
                       if k not in present and k not in disabled],
        "выключены": [{"фактор": FACTOR_LABELS.get(k, (k, ""))[0], "почему": DISABLED_WHY[k]}
                      for k in disabled],
        "сужение границ": {tag: [round(lo, 2), round(hi, 2)]
                           for tag, (lo, hi) in assessment.constraints.items()},
        "заметки": list(assessment.notes),
    }


# --------------------------------------------------------------------------- #
# катализатор
# --------------------------------------------------------------------------- #

def _reference_cycle(report: dict) -> dict | None:
    """Эталон — завершённый цикл, который наблюдается с пуска."""
    return next((c for c in report.get("циклы", [])
                 if c.get("завершён") and c.get("наблюдается с начала")), None)


def _life_estimate(report: dict, day: float, half: float | None) -> dict:
    life = report["остаточный_ресурс_мес"]
    second_half = half is not None and day > half
    methods = []
    if not second_half:
        methods.append({"способ": "(а) по средней скорости", "мес": life["по средней скорости"],
                        "ДИ": life.get("по средней скорости, ДИ"),
                        "на чём держится": "линейность и доверительный интервал скорости"})
    for analogy in life.get("по аналогии", []):
        methods.append({"способ": f"(б) по аналогии с циклом {analogy['цикл']}",
                        "мес": analogy["оставалось, мес"],
                        "нижняя граница": analogy["нижняя граница"],
                        "на чём держится": "сколько прожил предшественник после того же уровня"})
    lag = life.get("по отставанию от эталона") or {}
    if lag:
        methods.append({"способ": "(г) по отставанию от эталона", "мес": lag["остаток, мес"],
                        "на чём держится": ("эталон прожил ещё "
                                            f"{lag['эталон прожил ещё, мес']} мес, "
                                            f"отставание {lag['отставание, °C']} °C = "
                                            f"{lag['отставание, мес наработки']} мес")})
    methods.append({"способ": "(в) по текущей скорости", "мес": life["по текущей скорости"],
                    "на чём держится": "не прогноз: нижняя граница, если разгон ранней "
                                      "фазы не кончится"})

    core = [m["мес"] for m in methods
            if m["способ"].startswith(("(а)", "(г)"))
            or (m["способ"].startswith("(б)") and not m.get("нижняя граница"))]
    backtest = report.get("бэктест_метода") or []
    near = [row for row in backtest if row["сутки"] <= day] or backtest[:1]
    error = near[-1] if near else None
    return {
        "половина цикла": "вторая" if second_half else "первая",
        "диапазон, мес": [min(core), max(core)] if core else None,
        # «вероятнее» опирается на схождение (г) и поправленной (а) — во второй
        # половине цикла (а) не показывается, и схождения больше нет
        "вероятнее, мес": None if second_half or not lag else lag["остаток, мес"],
        "способы": methods,
        "ошибка линейной оценки на эталоне": None if error is None else {
            "сутки": error["сутки"], "ошибка, %": error["ошибка (а), %"]},
    }


def catalyst_section(report: dict | None, ts) -> dict:
    if not report:
        return {"доступно": False,
                "почему": "нет reports/catalyst_life.json — scripts/check_catalyst_life.py"}
    ts = pd.Timestamp(ts)
    cut = pd.Timestamp(report["срез_данных"])
    current = report["текущий_цикл"]
    current_start = pd.Timestamp(current["пуск"])
    changes = sorted(pd.Timestamp(o["конец"]) for o in report.get("остановы", [])
                     if o.get("смена катализатора"))
    started = [c for c in changes if c <= ts]
    section: dict = {"доступно": True, "срез данных": str(cut.date()), "оговорки": [EOR_SOURCE]}

    if not started:
        section.update({"пуск цикла": None, "сутки цикла": None, "ресурс": None})
        section["оговорки"].append("цикл наблюдается не с пуска: в начале данных катализатор "
                                   "уже был какого-то возраста, сутки цикла неизвестны")
        return section
    start = started[-1]
    section["пуск цикла"] = str(start.date())
    section["сутки цикла"] = int((ts - start).days)
    if start != current_start:
        section["ресурс"] = None
        section["оговорки"].append("оценка ресурса есть только для текущего цикла: для "
                                   "прошлого момента она опиралась бы на данные после него")
        return section

    day = float(current["наработка, сут"])
    reference = _reference_cycle(report)
    half = reference["длительность, сут"] / 2 if reference else None
    rate = report.get("скорость_дезактивации") or {}
    section.update({
        "оценка на сутки": int(day),
        "по данным после момента": bool(ts < cut),
        "NWABT, °C": current["NWABT сейчас, °C"],
        "уровень вывода, °C": report["уровень_вывода_°C"],
        "запас до уровня вывода, °C": current["запас до уровня вывода, °C"],
        "скорость в этом цикле, °C/мес": current["скорость в этом цикле, °C/мес"],
        "средняя скорость, °C/мес": rate.get("°C/мес"),
        "средняя скорость, ДИ": rate.get("95% ДИ"),
        "ресурс": _life_estimate(report, day, half),
    })
    if ts < cut:
        section["оговорки"].append(f"оценка посчитана по данным до {cut.date()} — позже "
                                   "выбранного момента")
    if half is not None:
        tail = [row for row in report.get("бэктест_метода") or [] if row["сутки"] > half]
        text = (f"после {half:.0f}-х суток цикла линейная оценка не показывается: на эталоне "
                "она всегда оптимистична")
        if tail:
            text += ", ошибка " + ", ".join(f"+{row['ошибка (а), %']:.0f} % на {row['сутки']}-х "
                                            "сутках" for row in tail)
        section["оговорки"].append(text)

    checkpoint = report.get("контрольная_точка") or {}
    if checkpoint.get("уровень эталона, °C") is not None:
        lag = checkpoint.get("отставание") or []
        last = lag[-1] if lag else None
        section["контрольная точка"] = {
            "сутки": checkpoint["сутки"],
            "дата": str((current_start + pd.Timedelta(days=checkpoint["сутки"])).date()),
            "достигнута": bool(checkpoint.get("достигнута")),
            "пересмотреть вниз, если NWABT выше, °C": round(
                checkpoint["уровень эталона, °C"] + CHECKPOINT_LEVEL_MARGIN_C, 1),
            "или скорость на сутках 60…N выше, °C/мес": CHECKPOINT_MAX_RATE_C_PER_MONTH,
            "отставание от эталона": None if last is None else {
                "сутки": last["сутки"], "°C": last["отставание, °C"], "значимо": last["значимо"]},
        }
    return section


def reliability_panel(assessment: ReliabilityAssessment, catalyst_report: dict | None,
                      ts: datetime | pd.Timestamp | None = None, basis: str = "age",
                      dp: str = "off") -> dict:
    """Всё содержание блока надёжности для выбранного момента."""
    moment = ts if ts is not None else assessment.ts
    return {"тяжесть": severity_section(assessment, basis, dp=dp),
            "катализатор": catalyst_section(catalyst_report, moment)}
