"""Что будет дальше: следующий анализ, тонны под риском, уже идущее движение.

Карточка отвечала на вопрос «что сделать сейчас». Три вещи, которых ей не хватало,
оператор спрашивает сразу после этого, и все три измерены, а не придуманы.

**Когда придёт анализ и что он покажет.** Ритм лаборатории ровно суточный, момент
следующего анализа предсказывается точно (медиана ошибки 0.0 ч). С диапазоном
сложнее: рабочий интервал 80 % покрывает фактический СЛЕДУЮЩИЙ анализ лишь в 66 %
случаев на валидации — до него ещё часы, и за эти часы сера успевает уехать.
Поэтому диапазон расширяется множителем, подобранным на ОБУЧАЮЩЕМ периоде
(1.52 к σ): покрытие 76.9 % на валидации и 80.4 % на тесте, то есть обещание
«восемь раз из десяти» честное. `scripts/check_next_lab.py`.

**Сколько тонн за этим стоит.** Эффект в мг/кг и процентах выпуска — не та мера, в
которой считает смена. Тонны под риском = расход товарного потока (тег, не
допущение) × часы до анализа: медиана 3054 т на валидации. Это не убыток, а объём,
который будет сделан до появления контрольного факта. Названную заказчиком цену
ошибки (50–100× запаса по качеству) на него НЕ умножаем: чем кончается прямая
подстановка этой цены, уже померено — `scripts/check_economic_ranking.py`.

**Не едет ли режим уже.** Отклик серы запаздывает на 4.6 ч, и чужое действие
(оператор, регулятор) за это время не видно ни в анализаторе, ни в признаках.
Система посоветует добавить, и два воздействия сложатся. Запрет частых действий
знает только о СВОИХ. Померено: в 28.7 % моментов валидации режим уже прошёл целый
шаг цикла (2 °C) за время отклика, среди вмешательств — 32.2 %.
`scripts/check_already_moving.py`.
"""
from __future__ import annotations

import pandas as pd

# Постоянная времени канала серы, ч (reports/delays.json): через неё считается и
# «проверить эффект», и окно, в котором ищется чужое движение.
TAU_H = 4.6
# Ритм лабораторных анализов, ч. Измерен на ОБУЧАЮЩЕМ периоде (reports/next_lab.json).
LAB_CADENCE_H = 24.0
# Множитель к сигме для обещания «на следующий анализ». Подобран на обучении до
# покрытия 80 % и проверен на валидации (76.9 %) и тесте (80.4 %).
NEXT_LAB_SIGMA = 1.52
# Из интервала контракта (mean ± 1.96σ) восстанавливаем σ.
Z95 = 1.96


def next_lab(state, q) -> dict | None:
    """Когда придёт следующий анализ и в каком диапазоне его ждать."""
    lims = state.quality.get("lims_sulfur_mgkg")
    mean = q.predictions.get("product_sulfur_mgkg")
    span = q.intervals.get("product_sulfur_mgkg")
    if lims is None or lims.age_hours is None or mean is None or span is None:
        return None
    sigma = (float(span[1]) - float(span[0])) / (2 * Z95)
    if not sigma > 0:
        return None
    due_in = LAB_CADENCE_H - float(lims.age_hours)
    half = NEXT_LAB_SIGMA * sigma
    return {
        "через, ч": round(float(max(due_in, 0.0)), 1),
        "ожидается": str(pd.Timestamp(state.ts) + pd.Timedelta(
            hours=float(max(due_in, 0.0)))),
        "задерживается": bool(due_in < 0),
        "диапазон, мг/кг": [round(float(mean - half), 1), round(float(mean + half), 1)],
        "доля попаданий на истории": 0.8,
    }


def tonnes_at_risk(hours: float | None, throughput_tph: float | None) -> float | None:
    """Сколько продукта будет сделано до контрольного факта."""
    if hours is None or throughput_tph is None or throughput_tph <= 0:
        return None
    return round(float(hours) * float(throughput_tph), -1)


def already_moving(model, ts, step_c: float, tau_h: float = TAU_H) -> dict | None:
    """Прошёл ли режим целый шаг цикла за время отклика — без нашего участия."""
    matrix = getattr(model, "feature_matrix", None)
    if matrix is None or "reg_wabt" not in getattr(matrix, "columns", []):
        return None
    index = matrix.index
    position = index.searchsorted(pd.Timestamp(ts), side="right") - 1
    if position < 0:
        return None
    earlier = index.searchsorted(index[position] - pd.Timedelta(hours=tau_h),
                                 side="right") - 1
    if earlier < 0 or earlier == position:
        return None
    change = float(matrix["reg_wabt"].iloc[position] - matrix["reg_wabt"].iloc[earlier])
    if change != change or abs(change) < float(step_c):
        return None
    return {"ход, °C": round(change, 2), "окно, ч": tau_h,
            "куда": "вверх" if change > 0 else "вниз"}


def sentence(block: dict) -> str:
    """Строка карточки «Что дальше» из собранного блока."""
    parts: list[str] = []
    lab = block.get("следующий анализ")
    if lab:
        low, high = lab["диапазон, мг/кг"]
        delay = " (анализ уже задерживается)" if lab["задерживается"] else ""
        parts.append(f"следующий анализ примерно через {lab['через, ч']:g} ч{delay}, "
                     f"ждём {low:g}–{high:g} мг/кг — так попадали 8 раз из 10")
    tonnes = block.get("тонн под риском")
    if tonnes:
        parts.append(f"до него будет сделано около {tonnes:g} т продукта")
    parts.append(f"эффект правки проявится за ~{TAU_H:g} ч — тогда и проверять")
    moving = block.get("режим уже едет")
    if moving:
        parts.append(f"ВНИМАНИЕ: режим уже идёт {moving['куда']} на "
                     f"{abs(moving['ход, °C']):g} °C за последние {moving['окно, ч']:g} ч, "
                     f"часть эффекта ещё в пути — не складывайте воздействия")
    return "; ".join(parts) + "."


# Насколько должен уйти режим за время отклика, чтобы считать это ПЕРЕХОДОМ, а не
# дрейфом: пять шагов цикла. Пуск и останов выглядят именно так, и советовать в
# такой момент «вернуть уставку за 45 циклов» — значит мешать технологу.
# Проверено по каталогу эпизодов (участник 2, docs/REGIME_EPISODES.md): ловит 12 из
# 12 фаз разгона и остывания и 9 из 9 остановов вместе с признаком «установка
# остановлена», помечает 1.6 % стационарных моментов при допустимых 5 %. Соседние
# пороги хуже: 5 °C помечает 7.6 % стационара, 15 и 20 °C теряют события.
TRANSITION_C = 10.0
# Единицы измерения по первой букве тега: карточка без них читается как отладка.
UNITS = {"T": "°C", "P": "МПа", "F": "т/ч", "W": "", "Q": "мг/кг"}


def cycles(count: int) -> str:
    """«1 цикл», «2 цикла», «5 циклов» — иначе карточку неприятно читать."""
    tail = count % 100
    if 11 <= tail <= 14:
        return f"{count} циклов"
    last = count % 10
    if last == 1:
        return f"{count} цикл"
    if 2 <= last <= 4:
        return f"{count} цикла"
    return f"{count} циклов"


def catalyst_age_months(moment, changes) -> float | None:
    """Возраст катализатора в месяцах по журналу замен; до первой замены — неизвестен."""
    moment = pd.Timestamp(moment)
    starts = [pd.Timestamp(c) for c in (changes or []) if pd.Timestamp(c) <= moment]
    if not starts:
        return None
    return (moment - max(starts)).total_seconds() / (86400 * 30.44)


def aging_drift(ts, train_end, rate_c_per_month: float | None,
                changes=(), cap_c: float | None = None) -> dict | None:
    """Насколько катализатор постарел относительно того, с которого снят диапазон, °C.

    Скорость — измерение участника 2 (`reports/catalyst_life.json`, 0.85 °C/мес),
    журнал замен — `reliability.catalyst_changes`. Мера — разница ВОЗРАСТА
    катализатора сейчас и на конец обучения, а не календарь: катализатор меняли
    посреди теста (23.04.2026), и свежему нужна меньшая температура, а не большая.
    Если катализатор сейчас моложе того, с которого снят диапазон, старением выход
    вверх не объясняется — ``None``.
    """
    if rate_c_per_month is None or train_end is None:
        return None
    now = catalyst_age_months(ts, changes)
    then = catalyst_age_months(train_end, changes)
    if now is None or then is None:
        return None
    older = now - then
    # только вверх: нужда измерена вверх, а сужать диапазон на свежем катализаторе
    # значило бы ввести непроверенное ограничение (участник 2, 21.09)
    if older <= 0:
        return None
    drift = float(rate_c_per_month) * older
    # и не больше размаха цикла: устаревшая опора не должна уводить границу без предела
    if cap_c is not None:
        drift = min(drift, float(cap_c))
    return {"°C": round(drift, 2),
            "месяцев": round(older, 1),
            "возраст катализатора, мес": round(now, 1),
            "скорость, °C/мес": float(rate_c_per_month)}


def out_of_band(state, bounds: dict, steps: dict,
                transition: dict | None = None,
                aging: dict | None = None) -> dict | None:
    """Уставки, вышедшие за рабочий диапазон, и сколько циклов до возврата.

    Диапазон — ДОПУЩЕНИЕ (квантили обучающего периода): паспортных ограничений
    оборудования в пакете нет, и эксперт 11.09 сказал, что не будет — они
    конфиденциальны. Поэтому карточка обязана называть и выход, и то, что граница
    наша, а не заводская.
    """
    out = []
    for tag, (low, high) in (bounds or {}).items():  # noqa: PLR1702
        current = state.telemetry_ht.get(tag, state.telemetry_avt.get(tag))
        if current is None or current != current:
            continue
        step = (steps["temperature_c"] if tag.startswith("T") else
                steps["pressure_mpa"] if tag.startswith("P") else
                abs(float(current)) * steps["flow_rel"])
        if not step > 0:
            continue
        gap = (float(current) - float(high) if current > high else
               float(low) - float(current) if current < low else 0.0)
        # Молчим, пока выход меньше шага цикла: такой тег оптимизатор возвращает
        # сам, ничего из рассмотрения не выпадает, и говорить оператору не о чем.
        if gap <= step:
            continue
        # температура ВЫШЕ диапазона в пределах старения катализатора — не
        # нарушение: диапазон взят с молодого катализатора, и «вернуть вниз»
        # значило бы поднять серу
        aged = bool(aging and tag.startswith("T") and current > high
                    and gap <= float(aging["°C"]))
        out.append({
            "тег": tag,
            "значение": round(float(current), 2),
            "диапазон": [round(float(low), 2), round(float(high), 2)],
            "выход": round(float(gap), 2),
            "куда возвращать": "вниз" if current > high else "вверх",
            "циклов до возврата": int(-(-gap // step)),
            "шаг за цикл": round(float(step), 2),
            "единица": UNITS.get(tag[0], ""),
            "старение катализатора": aged,
        })
    if not out:
        return None
    out.sort(key=lambda row: -row["циклов до возврата"])
    parts = [f"{row['тег']} {row['значение']:g} при диапазоне "
             f"{row['диапазон'][0]:g}–{row['диапазон'][1]:g} "
             f"(на {row['выход']:g} {row['единица']} "
             f"{'выше' if row['куда возвращать'] == 'вниз' else 'ниже'})"
             for row in out]
    worst = out[0]
    assumption = ("Диапазон — ДОПУЩЕНИЕ (квантили обучающего периода): "
                  "паспортных ограничений оборудования в пакете нет")
    if transition:
        # Режим уходит целыми десятками градусов — это пуск или останов. Выход за
        # диапазон в такой момент нормален, и «возврат за N циклов» был бы советом
        # мешать технологу выводить установку.
        return {
            "теги": out, "переход": transition,
            "строка": (f"режим в переходе: за {transition['окно, ч']:g} ч ушёл "
                       f"{transition['куда']} на {abs(transition['ход, °C']):g} °C. "
                       + "Вне рабочего диапазона: " + "; ".join(parts)
                       + ". Для пуска и останова это нормально — рекомендаций по "
                       + "возврату уставок система не даёт, режим ведёт технолог. "
                       + assumption),
        }
    to_return = [row for row in out if not row["старение катализатора"]]
    aged_rows = [row for row in out if row["старение катализатора"]]
    text = ""
    if aged_rows:
        names = ", ".join(f"{row['тег']} на {row['выход']:g} {row['единица']}"
                          for row in aged_rows)
        text += (f"выше диапазона обучения: {names} — это в пределах старения "
                 f"катализатора: он на {aging['месяцев']:g} мес старше того, с которого "
                 f"снят диапазон, при {aging['скорость, °C/мес']:g} °C/мес это "
                 f"≈ {aging['°C']:g} °C. Снижать температуру НЕ нужно — это подняло "
                 f"бы серу. ")
    if to_return:
        worst = max(to_return, key=lambda row: row["циклов до возврата"])
        parts_back = [part for part, row in zip(parts, out)
                      if not row["старение катализатора"]]
        text += ("режим вне рабочего диапазона: " + "; ".join(parts_back)
                 + f". Возврат — по {worst['шаг за цикл']:g} {worst['единица']} "
                 + f"за цикл, это {cycles(worst['циклов до возврата'])} "
                 + f"по {worst['тег']}. ")
    return {"теги": out, "строка": (text + assumption).strip()}
