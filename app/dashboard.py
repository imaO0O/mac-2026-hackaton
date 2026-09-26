"""Дашборд оператора: один цикл принятия решения целиком.

    streamlit run app/dashboard.py

Работает локально на CPU, без интернета и внешних сервисов. Показывает то, что
требует п.5 ТЗ: время и состояние, проблему, предлагаемое действие, ожидаемый
эффект, проверенные ограничения, уверенность и объяснение — либо мотивированный
отказ.

Тяжёлые объекты (телеметрия, модель) кэшируются: пересчёт при смене момента
времени занимает доли секунды.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nefte.agents.schemas import effect_text  # noqa: E402
from nefte.config import load_config  # noqa: E402
from nefte.data.cleaning import clean_lims_sulfur  # noqa: E402
from nefte.config import ROOT  # noqa: E402
from nefte.data.loaders import lims_series  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.reliability_panel import (  # noqa: E402
    FACTOR_LABELS,
    catalyst_basis,
    dp_mode,
    load_catalyst_report,
    reliability_panel,
)

st.set_page_config(page_title="МАС «Нефтекод»", layout="wide")

# Классы тяжести в интерфейсе — по-русски: оператор и эксперты читают карточку на
# русском, а «low / high» в ней выглядело как недоделка.
CLASS_RU = {"low": "мягкий", "medium": "средний", "high": "тяжёлый"}

# Боковая панель шире: названия сцен защиты длинные и обрезались на середине.
st.markdown("<style>[data-testid='stSidebar']{min-width:430px !important;"
            "width:430px !important}</style>", unsafe_allow_html=True)

# Шапка и плашка решения. Главный вопрос оператора — «что делать сейчас», и ответ на
# него должен читаться с другого конца пульта, раньше любых таблиц.
st.markdown("""<style>
.nk-top{display:flex;align-items:baseline;gap:14px;margin:-8px 0 10px}
.nk-brand{font-size:1.9rem;font-weight:800;color:#1F2A44}
.nk-line{font-size:1rem;color:#6B7280}
.nk-dec{border-radius:12px;padding:16px 22px;margin:6px 0 16px;color:#fff}
.nk-dec .k{font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;opacity:.85}
.nk-dec .h{font-size:1.75rem;font-weight:800;line-height:1.25;margin:4px 0 6px}
.nk-dec .s{font-size:1rem;opacity:.95}
.nk-dec .c{margin-top:10px;display:flex;flex-wrap:wrap;gap:8px}
.nk-dec .c span{background:rgba(255,255,255,.18);border-radius:999px;padding:3px 12px;
 font-size:.9rem}
.nk-act{background:#1F4E8C}.nk-hold{background:#2E7D4F}.nk-watch{background:#C0701F}
.nk-stop{background:#B03A2E}.nk-down{background:#5B6472}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Загрузка данных и модели…")
def load_system():
    from run_cycle import build_system

    cfg = load_config()
    sb = StateBuilder(cfg)
    return cfg, sb, build_system(sb, cfg)


@st.cache_data(show_spinner=False)
def load_lab(_series_key: str) -> pd.Series:
    return clean_lims_sulfur(lims_series(_series_key))


def quality_chart(sb: StateBuilder, lab: pd.Series, ts: pd.Timestamp,
                  limit: float, days: int = 14) -> go.Figure:
    """Сера: поточный анализатор, лабораторные точки, предел и текущий момент."""
    lo, hi = ts - pd.Timedelta(days=days), ts + pd.Timedelta(days=1)
    pak = sb.pak_sulfur.loc[lo:hi]
    frozen = sb.pak_frozen.loc[lo:hi]

    fig = go.Figure()
    # Оба поточных анализатора: оперативный — Q21, и именно их расхождение
    # карточка называет недостоверностью (строка «Достоверность»). Раньше на
    # графике был только файловый ряд — прибор, по которому система не решает.
    q21 = sb.q21_sulfur.loc[lo:hi] if len(sb.q21_sulfur) else pd.Series(dtype=float)
    if len(q21):
        fig.add_trace(go.Scatter(x=q21.index, y=q21.values, name="Q21 (оперативный)",
                                 line=dict(color="#F58518", width=1)))
        # полка неисправности Q21 (24.88 ± 0.04) — детектор с допуском
        q21_frozen = sb.q21_frozen.loc[lo:hi] if len(sb.q21_frozen) else pd.Series(dtype=bool)
        if q21_frozen.any():
            shelf = q21[q21_frozen.reindex(q21.index, fill_value=False)]
            fig.add_trace(go.Scatter(x=shelf.index, y=shelf.values, name="Q21 завис",
                                     mode="markers", marker=dict(color="#B279A2", size=3)))
    fig.add_trace(go.Scatter(x=pak.index, y=pak.values, name="ПАК (файл)",
                             line=dict(color="#4C78A8", width=1)))
    if frozen.any():
        stuck = pak[frozen.reindex(pak.index, fill_value=False)]
        fig.add_trace(go.Scatter(x=stuck.index, y=stuck.values, name="ПАК завис",
                                 mode="markers", marker=dict(color="#E45756", size=3)))
    lab_w = lab.loc[lo:hi]
    fig.add_trace(go.Scatter(x=lab_w.index, y=lab_w.values, name="ЛИМС",
                             mode="markers", marker=dict(color="#1F2A44", size=10,
                                                         symbol="diamond",
                                                         line=dict(color="#FFFFFF", width=1.5))))
    # Подпись предела — слева на подложке: справа её перекрывали свежие точки.
    fig.add_hline(y=limit, line_dash="dash", line_color="#E45756",
                  annotation_text=f"предел {limit:g} мг/кг", annotation_position="top left",
                  annotation_bgcolor="rgba(255,255,255,0.85)")
    # Подпись «сейчас» отдельной аннотацией: у add_vline на оси дат она падает.
    fig.add_vline(x=ts, line_color="#54A24B")
    fig.add_annotation(x=ts, y=1, yref="paper", text="сейчас", showarrow=False,
                       yanchor="bottom", font=dict(color="#2E7D4F"))
    # Даты по-русски числами: plotly по умолчанию пишет «Feb 20».
    fig.update_xaxes(tickformat="%d.%m", hoverformat="%d.%m.%Y %H:%M")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="сера, мг/кг", legend=dict(orientation="h"))
    return fig


def pareto_chart(candidates, chosen=None, hold=None, limit: float | None = None) -> go.Figure:
    """Все допустимые варианты: сера против выпуска, фронт Парето, выбор и бездействие.

    Раньше здесь были только три альтернативы и выбранный вариант — четыре точки,
    по которым размен «качество ↔ выпуск» не виден. Теперь видно всё поле, из
    которого оптимизатор выбирал, и где на нём оказалось «ничего не делать».
    """
    def point(c):
        return c.predicted_quality.get("product_sulfur_mgkg"), c.throughput

    fig = go.Figure()
    front = [c for c in candidates if c.pareto_rank == 0]
    rest = [c for c in candidates if c.pareto_rank != 0]
    for group, name, color, size in ((rest, "допустимые варианты", "#B8BEC8", 7),
                                     (front, "фронт Парето", "#1F4E8C", 10)):
        if not group:
            continue
        xs, ys = zip(*(point(c) for c in group))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=name, marker=dict(color=color, size=size),
            text=[describe_moves(c) for c in group],
            hovertemplate="%{text}<br>сера %{x:.2f}<br>выпуск %{y:.1f}<extra></extra>"))
    for cand, name, color, symbol in ((hold, "ничего не делать", "#111827", "x"),
                                      (chosen, "выбрано", "#D9822B", "star")):
        if cand is None or point(cand)[0] is None:
            continue
        x, y = point(cand)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=name,
                                 marker=dict(color=color, size=18, symbol=symbol,
                                             line=dict(color="#FFFFFF", width=1.5))))
    if limit is not None:
        fig.add_vline(x=limit, line_dash="dash", line_color="#B03A2E",
                      annotation_text=f"предел {limit:g}", annotation_position="bottom right")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="прогноз серы, мг/кг", yaxis_title="выпуск, т/ч",
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def scene_moments() -> list[tuple[str, pd.Timestamp, str]]:
    """Моменты сцен защиты — те же, что в консольном `scripts/demo.py`.

    Дашборд и консольное демо обязаны показывать одно и то же: сцены подобраны по
    журналам прогонов, и две независимые копии списка разошлись бы при первой
    правке.
    """
    from demo import SCENES

    out = []
    for scene in SCENES:
        stamps = scene["ts"] if isinstance(scene["ts"], list) else [scene["ts"]]
        for ts in stamps:
            stamp = pd.Timestamp(ts)
            out.append((f"{scene['title']} — {stamp:%d.%m.%Y %H:%M}", stamp, scene["point"]))
    return out


@st.cache_data(show_spinner=False)
def load_cetane() -> pd.Series:
    return lims_series("Гидроочистка|2|CetaneNumber").sort_index()


def readable(text: str) -> str:
    """Текст агентов — словами оператора, без служебных идентификаторов.

    Агенты пишут для журнала и аудита: «mix_098», «(medium)», «catalyst (1.50)»,
    «docs/DP_PROXY.md». Технологу на экране это ничего не говорит, поэтому здесь —
    только перевод; сами числа и выводы не трогаем.
    """
    text = text.replace("лучший — hold", "лучший — ничего не менять")
    text = re.sub(r"\bmix_0*(\d+)", r"вариант №\1", text)
    text = re.sub(r"\blocal_0*(\d+)", r"уточнённый вариант №\1", text)
    for eng, ru in CLASS_RU.items():
        text = text.replace(f"({eng})", f"({ru})")
    for key, (label, _) in FACTOR_LABELS.items():
        text = re.sub(rf"\b{key} \(", f"{label} (", text)
    text = re.sub(r"\s*\(docs/[^)]*\)", "", text)
    return text.replace(" не вошёл в смесь: ", " — в смесь не входит: ")


def tag_names(cfg: dict) -> dict[str, tuple[str, str]]:
    """Тег → (название, единица) из справочника рычагов в конфиге."""
    out = {}
    for unit_tags in (cfg.get("controls") or {}).values():
        for item in unit_tags or []:
            out[item["tag"]] = (item.get("name", item["tag"]),
                                str(item.get("unit", "")).replace("м3", "м³"))
    return out


def describe_moves(c) -> str:
    """Вариант словами: какие уставки и насколько он двигает."""
    parts = [f"{t} {d:+.2f}" for t, d in (c.deltas or {}).items() if abs(d) > 1e-6]
    return ", ".join(parts) if parts else "ничего не менять"


def flow_strip(rec) -> None:
    """Путь решения одной полосой: пять шагов агентов цветными блоками.

    Подробная трасса ниже отвечает на «что именно сказал каждый», а инженеру ЦУП
    сначала нужно увидеть одним взглядом, КТО повлиял на решение: где данные
    отбракованы, кто ограничил, чем кончилось. Цвет — по смыслу шага: зелёный —
    всё в порядке, жёлтый — ограничение, красный — стоп.
    """
    green, amber, red, blue, gray = "#2E7D4F", "#D9822B", "#B03A2E", "#1F4E8C", "#5B6472"
    outcome = rec.outcome()
    # Держим режим, но показатель уже вне спецификации (сцена «Т95 за пределом») —
    # жёлтым, как и плашка над карточкой: зелёный здесь читался бы «всё хорошо».
    off_spec = "ЗА ПРЕДЕЛОМ" in rec.problem or "ВНЕ СПЕЦИФИКАЦИИ" in rec.problem
    boxes = []
    for step in rec.trace:
        text = readable(step.summary)
        name = step.agent
        if name == "срез состояния":
            color = red if "непригоден" in text else green
        elif name == "агент надёжности":
            color = (red if "НЕДОПУСТИМ" in text else gray if "установка стоит" in text else
                     amber if "тяжёлый режим" in text or "ограничил" in text else green)
        elif name == "оркестратор":
            color = (red if outcome == "отказ" else blue if outcome == "меняем уставки" else
                     amber if off_spec else green)
        else:
            color = gray
        short = text if len(text) <= 150 else text[:147].rsplit(" ", 1)[0] + "…"
        boxes.append(f"<div style='flex:1 1 0;min-width:0;border-radius:8px;"
                     f"padding:8px 10px;background:{color};color:#fff'>"
                     f"<div style='font-weight:700;font-size:0.95rem'>{name}</div>"
                     f"<div style='font-size:0.8rem;line-height:1.3;opacity:.95'>{short}</div>"
                     f"</div>")
    arrow = "<div style='align-self:center;font-size:1.4rem;color:#8a93a3'>→</div>"
    st.markdown("<div style='display:flex;gap:6px;align-items:stretch;margin:4px 0 12px'>"
                + arrow.join(boxes) + "</div>", unsafe_allow_html=True)


def short_name(name: str) -> str:
    """«Р-201: температура ГСС на выходе» → «Температура ГСС на выходе Р-201»."""
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if ":" in name:
        where, what = name.split(":", 1)
        name = f"{what.strip()} {where.strip()}"
    return name[:1].upper() + name[1:]


def decision_banner(rec, *, ts, cfg, system, state, q, unit_down: bool) -> None:
    """Ответ системы одной плашкой: что делать, чего ждать и насколько верить.

    Цвет — тот же язык, что в полосе решения: синий — действие, зелёный — держим,
    янтарный — держим, но показатель вне спецификации, красный — совета нет, серый —
    установка стоит. Числа берутся из той же рекомендации, что и карточка ниже.
    """
    moves = {t: d for t, d in (rec.action.deltas if rec.action else {}).items()
             if abs(d) > 1e-6}
    outlook = getattr(rec, "outlook", None) or {}
    lab = outlook.get("следующий анализ") or {}
    when = f"{pd.Timestamp(ts):%d.%m.%Y %H:%M}"
    chips: list[str] = []
    if lab.get("задерживается"):
        chips.append("анализ просрочен")
    elif lab.get("через, ч") is not None:
        chips.append(f"анализ через {lab['через, ч']:g} ч")

    if rec.abstained:
        kind = "down" if unit_down else "stop"
        head = ("Совета нет: установка остановлена" if unit_down else
                "Совета нет: данным сейчас нельзя верить"
                if (not state.data_quality.usable or q.confidence < system.min_confidence)
                else "Совета нет: ни один вариант не проходит ограничения")
        sentences = re.split(r"(?<=[.!?])\s+", readable(rec.abstain_reason or ""))
        # Причины, а не служебная арифметика уверенности: её видно в карточке ниже.
        sub = " ".join([x for x in sentences if not x.startswith("Уверенность")][:2])
    elif moves:
        kind = "act"
        names = tag_names(cfg)
        step = cfg["limits"]["max_step_per_cycle"]

        def share(tag: str) -> float:
            before = rec.action.moves[tag] - moves[tag]
            allowed = (step["temperature_c"] if tag.startswith("T") else
                       step["pressure_mpa"] if tag.startswith("P") else
                       abs(before) * step["flow_rel"])
            return abs(moves[tag]) / allowed if allowed else 0.0

        main_tag = max(moves, key=share)
        name, unit = names.get(main_tag, (main_tag, ""))
        d = moves[main_tag]
        before, after = rec.action.moves[main_tag] - d, rec.action.moves[main_tag]
        digits = 3 if abs(d) < 0.01 else 2
        head = (f"{short_name(name)}: {before:.{digits}f} → {after:.{digits}f} {unit}"
                .strip())
        rest = [f"{t} {moves[t]:+.2f}" for t in sorted(moves, key=share, reverse=True)
                if t != main_tag]
        sub = (f"Главный рычаг — {main_tag} ({d:+.{digits}f} {unit}, "
               f"{share(main_tag):.0%} допустимого шага).")
        if rest:
            sub += " Остальные правки мелкие: " + ", ".join(rest) + "."
        effect = rec.expected_effect or {}
        now = q.predictions.get("product_sulfur_mgkg")
        new = effect.get("product_sulfur_mgkg") or effect.get("сера, мг/кг")
        if isinstance(now, (int, float)) and isinstance(new, (int, float)):
            chips.insert(0, f"сера {now:.2f} → {new:.2f} мг/кг")
        out = effect.get("throughput_change_pct") or effect.get("выпуск, %")
        if isinstance(out, (int, float)):
            chips.insert(1, f"выпуск {out:+.2f} %")
        conf = f"уверенность {rec.confidence:.2f}"
        if not rec.action.guaranteed:
            conf += " — нужен контрольный анализ"
        chips.append(conf)
    else:
        off_spec = "ЗА ПРЕДЕЛОМ" in rec.problem or "ВНЕ СПЕЦИФИКАЦИИ" in rec.problem
        kind = "watch" if off_spec else "hold"
        head = ("Режим по сере не менять — Т95 выше предела, решение за технологом"
                if off_spec and "Т95" in rec.problem else
                "Режим по сере не менять — но показатель вне спецификации" if off_spec
                else "Режим не менять")
        sub = rec.problem
        chips.append(f"уверенность {rec.confidence:.2f}")
    kicker = {"act": "совет оператору", "hold": "решение системы", "watch": "внимание технологу",
              "stop": "отказ с причиной", "down": "отказ с причиной"}[kind]
    chip_html = "".join(f"<span>{c}</span>" for c in chips)
    st.markdown(f"<div class='nk-dec nk-{kind}'><div class='k'>{when} · {kicker}</div>"
                f"<div class='h'>{head}</div><div class='s'>{sub}</div>"
                f"<div class='c'>{chip_html}</div></div>", unsafe_allow_html=True)


def trace_rows(rec) -> None:
    """Цикл решения по шагам: какой агент что вернул и какое правило сработало.

    Полоса решения наверху говорит то же коротко; подробный журнал нужен при
    разборе, поэтому он живёт на отдельной вкладке.
    """
    for number, step in enumerate(rec.trace, start=1):
        left, right = st.columns([1, 4])
        left.markdown(f"**{number}. {step.agent}**")
        right.markdown(readable(step.summary))
        if step.details:
            with right.expander("подробности"):
                st.json(step.details, expanded=False)


def ask_block(cfg: dict, rec) -> None:
    """Вопрос оператора локальной LLM — только если слой включён в конфиге."""
    from nefte.llm_explain import LocalLLMExplainer

    explainer = LocalLLMExplainer.from_config(cfg)
    if explainer is None:
        return
    st.subheader("Спросить систему")
    question = st.text_input("Вопрос по этому решению",
                             placeholder="Почему расход, а не температура?")
    if question:
        answer = explainer.ask(rec, question)
        (st.success if answer.from_llm else st.warning)(answer.text)
        if answer.note:
            st.caption(answer.note)


def severity_block(section: dict) -> None:
    """Тяжесть режима: не одно число, а из чего оно сложилось.

    Раньше здесь была голая столбиковая диаграмма факторов. Она не отвечала на два
    вопроса, которые оператор задаёт первыми: что именно класс запрещает и почему
    фактора нет в списке — его не измерили или он намеренно выключен. Содержание
    собирает `nefte.reliability_panel`, проверенный тестами; здесь только вёрстка.
    """
    st.subheader("Тяжесть режима: из чего сложилась")
    left, right = st.columns([2, 3])
    with left:
        st.metric("Индекс", f"{section['индекс']:.2f}",
                  CLASS_RU.get(section["класс"], section["класс"]),
                  delta_color="off", delta_arrow="off")
        st.caption(section["что значит класс"])
        if section["выключены"]:
            for item in section["выключены"]:
                st.caption(readable(f"⏻ {item['фактор']} — {item['почему']}"))
        if section["нет данных"]:
            st.caption("нет измерений: " + ", ".join(section["нет данных"])
                       + " (вклады остальных пересчитаны на их веса)")
    with right:
        rows = section["факторы"]
        if rows:
            frame = pd.DataFrame(rows).set_index("название")
            # Горизонтальные столбцы: у вертикальных названия факторов обрезались
            # («износ катализат…») и читались только наклонив голову.
            order = frame["вклад"].sort_values()
            fig = go.Figure(go.Bar(x=order.values, y=order.index, orientation="h",
                                   marker_color="#1F4E8C", text=[f"{v:.2f}" for v in order],
                                   textposition="outside", cliponaxis=False))
            fig.update_layout(height=60 + 42 * len(order), margin=dict(l=10, r=40, t=10, b=10),
                              xaxis_title="вклад в индекс тяжести")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(frame[["значение", "вес", "вклад"]], width="stretch")
    if section["сужение границ"]:
        limits = ", ".join(f"{tag} ∈ [{lo:g}, {hi:g}]"
                           for tag, (lo, hi) in section["сужение границ"].items())
        st.caption(f"Агент надёжности сузил границы оптимизатору: {limits}")
    for note in section["заметки"]:
        st.caption(readable(note))


def method_line(method: dict) -> str:
    """Способ оценки ресурса одной строкой. Раньше сюда печатался словарь как есть —
    с «нижняя граница: True» и «ДИ: [16.9, 23.4]»."""
    head = str(method.get("способ", ""))
    months = method.get("мес")
    if months is not None:
        head += f" — {months:g} мес"
    extra = []
    if method.get("ДИ"):
        low, high = method["ДИ"]
        extra.append(f"ДИ {low:g}–{high:g}")
    if method.get("нижняя граница"):
        extra.append("нижняя граница")
    if extra:
        head += f" ({', '.join(extra)})"
    known = {"способ", "мес", "ДИ", "нижняя граница", "на чём держится"}
    rest = [f"{k}: {v}" for k, v in method.items() if k not in known]
    basis = method.get("на чём держится")
    return "; ".join([head] + ([f"на чём держится: {basis}"] if basis else []) + rest)


def catalyst_block(section: dict) -> None:
    """Ресурс катализатора — с оговорками видимым текстом, а не в подсказке."""
    if not section.get("доступно"):
        st.caption(f"Ресурс катализатора не посчитан: {section.get('почему', '')}")
        return
    st.subheader("Катализатор: где цикл и сколько осталось")
    cols = st.columns(4)
    cols[0].metric("Пуск цикла", section.get("пуск цикла") or "—")
    cols[1].metric("Сутки цикла", section.get("сутки цикла") if section.get("сутки цикла")
                   is not None else "—")
    resource = section.get("ресурс") or {}
    span = resource.get("диапазон, мес")
    cols[2].metric("Ресурс, мес", f"{span[0]:g}–{span[1]:g}" if span else "—",
                   None if resource.get("вероятнее, мес") is None
                   else f"вероятнее {resource['вероятнее, мес']:g}", delta_color="off",
                   delta_arrow="off")
    margin = section.get("запас до уровня вывода, °C")
    cols[3].metric("Запас до вывода, °C", "—" if margin is None else f"{margin:.1f}")
    for method in (resource.get("способы") or []):
        st.caption("· " + method_line(method))
    for caveat in section.get("оговорки", []):
        st.warning(caveat)


def main() -> None:
    cfg, sb, system = load_system()
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]
    lab = load_lab(cfg["quality"]["target"]["lims_source"])

    st.markdown("<div class='nk-top'><div class='nk-brand'>МАС «Нефтекод»</div>"
                "<div class='nk-line'>советчик оператора · АВТ → гидроочистка 24-2000 → "
                "смешение ДТ</div></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Момент времени")
        # Прямая ссылка на сцену: ?scene=3 — на защите сцену открывают одной
        # ссылкой, не листая меню, и по ней же снимаются скриншоты для слайдов.
        try:
            start = int(st.query_params.get("scene", 0))
        except (TypeError, ValueError):
            start = 0
        mode = st.radio("Что смотреть", ["Сцены защиты", "Любой момент"])
        point = None
        if mode == "Сцены защиты":
            scenes = scene_moments()
            index = st.selectbox("Сцена", range(len(scenes)),
                                 index=min(max(start, 0), len(scenes) - 1),
                                 format_func=lambda i: scenes[i][0])
            _, ts, point = scenes[index]
        else:
            day = st.date_input("Дата", value=pd.Timestamp("2026-03-05").date(),
                                min_value=sb.ht.index[0].date(),
                                max_value=sb.ht.index[-1].date())
            hour = st.slider("Час", 0, 23, 0)
            ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
        if point:
            st.info(f"**Что показывает сцена.** {readable(point)}")
        st.caption("Данные и модель кэшируются, пересчёт занимает доли секунды.")

    state = sb.build(pd.Timestamp(ts))
    # Лимит частоты воздействий — состояние оркестратора, и осмыслен он только в
    # хронологическом прогоне. На дашборде оператор двигает ползунок как хочет, в
    # том числе назад: без сброса карточка зависела бы от истории просмотра, а не
    # от выбранного момента.
    system._last_action_ts = None
    rec = system.run(state)
    q = system.quality.assess(state)
    r = system.reliability.assess(state)

    # ---------- решение одной плашкой ------------------------------------ #
    decision_banner(rec, ts=ts, cfg=cfg, system=system, state=state, q=q,
                    unit_down=system.reliability.is_unit_down(state))

    # ---------- строка состояния -------------------------------------- #
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    # Показываем ОПЕРАТИВНЫЙ анализатор — тот, по которому решает система
    # (`quality.analyzer_source`). До 21.09 здесь стоял ряд из файла ПАК и после
    # перехода на Q21 дашборд показывал оператору «18.45, завис», когда система
    # работала по живому Q21 с 24.9: подпись и число были от другого прибора.
    source = str(cfg["quality"].get("analyzer_source", "pak"))
    key, label = (("q21_sulfur_ppm", "Q21") if source == "q21"
                  else ("pak_sulfur_ppm", "ПАК"))
    pak = state.quality.get(key) or state.quality.get("pak_sulfur_ppm")
    lims = state.quality.get("lims_sulfur_mgkg")
    # На остановленной установке риск по сере и класс тяжести смысла не имеют:
    # «риск 36 %» и «мягкий режим» рядом с «установка стоит» читались как
    # противоречие. Решение системы от этого не меняется — только подписи.
    unit_down = system.reliability.is_unit_down(state)
    risk = q.spec_risk.get("product_sulfur_mgkg", 0) if q.spec_risk else None
    c1.metric(f"Прибор {label}, мг/кг",
              f"{pak.value:.2f}" if pak and pak.value else "—",
              "завис" if pak and pak.is_frozen else
              (f"возраст {pak.age_hours:.0f} ч" if pak and pak.age_hours is not None
               else None),
              delta_color="inverse" if pak and pak.is_frozen else "off",
              delta_arrow="off")
    c2.metric("ЛИМС, мг/кг", f"{lims.value:.2f}" if lims and lims.value else "—",
              f"возраст {lims.age_hours:.0f} ч" if lims and lims.age_hours else None,
              delta_color="off", delta_arrow="off")
    # На остановленной установке продукта нет — число прогноза читалось бы как
    # качество того, чего не выпускают.
    c3.metric("Прогноз, мг/кг",
              f"{q.predictions.get('product_sulfur_mgkg', float('nan')):.2f}"
              if q.predictions and not unit_down else "—",
              "установка стоит" if unit_down else
              "данным не верим" if not state.data_quality.usable else
              (f"риск {risk:.0%}" if risk is not None else None),
              delta_color="off" if unit_down or not state.data_quality.usable or risk is None
              or risk < system.act_risk_threshold else "inverse", delta_arrow="off")
    c4.metric("Тяжесть режима", "—" if unit_down else f"{r.severity_index:.2f}",
              "установка стоит" if unit_down else CLASS_RU.get(r.risk_class, r.risk_class),
              delta_color="off", delta_arrow="off")
    # Т95 и цетановое число — второй и третий обязательные показатели. Без них в
    # строке состояния оператор видел одну серу и не знал, что другой показатель
    # уже за пределом.
    t95 = q.predictions.get("product_t95_c")
    t95_limit = cfg["spec"]["t95_c"]["max"]
    c5.metric("Т95 по анализу, °C", "—" if t95 is None else f"{t95:.1f}",
              None if t95 is None else (f"выше {t95_limit:.0f}" if t95 > t95_limit
                                        else f"предел {t95_limit:.0f}"),
              delta_color="inverse" if t95 is not None and t95 > t95_limit else "off",
              delta_arrow="off")
    cetane = load_cetane().loc[:pd.Timestamp(ts)]
    # норматив ЦЧ — у товарной марки, у самого ГО ДТ его нет (ответ 15.09)
    grades = cfg["spec"].get("grades") or {}
    blend_grade = grades.get(cfg["spec"].get("blend_grade"), {})
    cetane_min = blend_grade.get("cetane_number_min",
                                 cfg["spec"].get("cetane_number", {}).get("min", 51.0))
    if len(cetane):
        value, when = float(cetane.iloc[-1]), cetane.index[-1]
        c6.metric("ЦЧ ГО ДТ", f"{value:.1f}",
                  f"{blend_grade.get('name', 'товарное')} ≥ {cetane_min:g}, анализ {when:%d.%m}",
                  delta_color="inverse" if value < cetane_min else "off", delta_arrow="off")
    else:
        c6.metric("ЦЧ ГО ДТ", "—")

    # ---------- путь решения одной полосой ----------------------------- #
    st.subheader("Как система пришла к решению")
    st.caption("Зелёный — всё в порядке · жёлтый — агент ограничил · красный — стоп · "
               "синий — предлагаем действие. Решение принимает оператор.")
    flow_strip(rec)

    # Детали — по вкладкам: на одной ленте разделы шли друг за другом одинаковыми
    # блоками, и до графика или тяжести режима приходилось долго листать.
    tabs = st.tabs(["Совет и проверки", "Варианты", "Качество и данные",
                    "Режим и катализатор", "Смешение", "Журнал агентов"])
    with tabs[0]:
        # ---------- рекомендация ------------------------------------------ #
        st.subheader("Рекомендация оператору")
        if rec.abstained:
            st.error(f"**Рекомендации нет.** {readable(rec.abstain_reason)}")
        else:
            moves = {t: d for t, d in (rec.action.deltas if rec.action else {}).items()
                     if abs(d) > 1e-6}
            # Зелёная плашка — только когда всё в норме: «Т95 ЗА ПРЕДЕЛОМ» или «ФАКТ ВНЕ
            # СПЕЦИФИКАЦИИ» при удержании режима читались в зелёном как «всё хорошо».
            off_spec = "ЗА ПРЕДЕЛОМ" in rec.problem or "ВНЕ СПЕЦИФИКАЦИИ" in rec.problem
            box = (st.warning if off_spec else st.success) if not moves else (
                st.warning if rec.action and not rec.action.guaranteed else st.info)
            box(f"**{rec.problem}**")

            # Действие — на всю ширину: в левой колонке 2/5 числа таблицы уходили за
            # край экрана ноутбука, оставались только названия параметров.
            st.markdown("**Действие**")
            if moves:
                # точность — та, при которой ход виден: давление двигается на
                # тысячные МПа, и при двух знаках «Δ 0» читалось как «не трогать»
                def digits(delta: float) -> int:
                    return 3 if abs(delta) < 0.01 else 2
                # Оператору нужно не «T5», а что это и в чём, и какой из шагов
                # главный: +0.12 °C рядом с −6.63 м³/ч — не равные по смыслу правки.
                # Размер шага — в долях разрешённого за цикл (limits.max_step_per_cycle).
                names = tag_names(cfg)
                step = cfg["limits"]["max_step_per_cycle"]

                def share(tag: str, delta: float, before: float) -> str:
                    allowed = (step["temperature_c"] if tag.startswith("T") else
                               step["pressure_mpa"] if tag.startswith("P") else
                               abs(before) * step["flow_rel"])
                    return f"{abs(delta) / allowed:.0%}" if allowed else "—"

                rows = []
                for t, d in sorted(moves.items(), key=lambda kv: -abs(kv[1]) / max(
                        1e-9, (step["temperature_c"] if kv[0].startswith("T") else
                               step["pressure_mpa"] if kv[0].startswith("P") else
                               abs(rec.action.moves[kv[0]] - kv[1]) * step["flow_rel"]))):
                    name, unit = names.get(t, (t, ""))
                    before = rec.action.moves[t] - d
                    rows.append({"параметр": f"{t} — {name}" + (f", {unit}" if unit else ""),
                                 "сейчас": f"{before:.{digits(d)}f}",
                                 "рекомендуется": f"{rec.action.moves[t]:.{digits(d)}f}",
                                 "Δ": f"{d:+.{digits(d)}f}",
                                 "от допустимого шага": share(t, d, before)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Сверху — главный рычаг совета. «От допустимого шага» — доля "
                           "изменения, разрешённого за один цикл.")
            else:
                st.write("Изменение уставок не требуется.")
            left, right = st.columns([2, 3])
            with left:
                if moves:
                    st.markdown("**Ожидаемый эффект**")
                    st.write(effect_text(rec.expected_effect))
                else:
                    # Без действия «+0.00 к бездействию, выпуск +0.00 %» — шум: оператору
                    # нужно, что будет при текущем режиме.
                    st.markdown("**Если ничего не менять**")
                    st.write(re.sub(r",\s*(выпуск|энергия) [+-]0\.00 %", "", re.sub(
                        r"\s*\([+-]0\.00[^)]*\)", "", effect_text(rec.expected_effect))))
                st.markdown(f"**Уверенность:** {rec.confidence:.2f}")
            with right:
                st.markdown("**Проверенные ограничения**")
                for item in rec.checked_constraints:
                    st.markdown(f"- {item}")
                st.markdown(f"**Почему:** {readable(rec.explanation)}")
                cause = getattr(rec, "cause", None)
                if cause:
                    st.markdown(f"**Разбор причины:** {cause.get('строка', '')}")
                    st.caption("Вклады по группам каналов, мг/кг: " + ", ".join(
                        f"{name} {value:+.2f}"
                        for name, value in (cause.get("вклады по группам") or {}).items()))

        band = getattr(rec, "out_of_band", None)
        if band:
            # при пуске и останове это не нарушение, а работа технолога
            (st.info if band.get("переход") else st.warning)(
                f"**Возврат в норму.** {band.get('строка', '')}")
            st.dataframe(pd.DataFrame(band.get("теги") or []),
                         hide_index=True, width="stretch")

        outlook = getattr(rec, "outlook", None)
        if outlook:
            st.subheader("Что дальше")
            lab_block = outlook.get("следующий анализ") or {}
            cols = st.columns(3)
            if lab_block.get("задерживается"):
                # «через 0 ч — задерживается» читалось как ошибка: анализ уже должен был
                # прийти, и оператору нужно именно это.
                cols[0].metric("Следующий анализ", "просрочен", "ждём контрольный анализ",
                               delta_color="inverse", delta_arrow="off")
            elif lab_block:
                cols[0].metric("Следующий анализ через", f"{lab_block['через, ч']:g} ч")
            if lab_block:
                # На непригодном срезе карточка диапазон не обещает: он опирался бы на
                # данные, которым система только что отказалась верить. Дашборд
                # падал здесь с KeyError — ровно в сцене «недостоверные данные».
                if "диапазон, мг/кг" in lab_block:
                    low, high = lab_block["диапазон, мг/кг"]
                    cols[1].metric("Ожидаемый диапазон, мг/кг", f"{low:g}–{high:g}",
                                   help="попадание 8 раз из 10 — измерено на обучении, "
                                        "проверено на валидации и тесте")
                else:
                    cols[1].metric("Ожидаемый диапазон, мг/кг", "не обещаем",
                                   help="данным сейчас верить нельзя — диапазон опирался "
                                        "бы на них; ждём контрольный анализ")
            tonnes = outlook.get("тонн под риском")
            if tonnes:
                cols[2].metric("Продукта до анализа, т", f"{tonnes:g}",
                               help="расход товарного потока × часы до анализа: объём, "
                                    "который будет сделан до контрольного факта")
            moving = outlook.get("режим уже едет")
            if moving:
                st.warning(f"Режим уже идёт {moving['куда']} на "
                           f"{abs(moving['ход, °C']):g} °C за последние "
                           f"{moving['окно, ч']:g} ч — часть эффекта ещё в пути, "
                           f"не складывайте воздействия.")

    with tabs[1]:
        # ---------- альтернативы ------------------------------------------- #
        if rec.alternatives:
            st.subheader("Альтернативы и фронт Парето")
            # Таблица и поле выбора друг под другом: с описанием вариантов словами в
            # половине ширины не помещались выпуск, энергия и Т95.
            st.caption("Три различающиеся альтернативы — оператор видит, чем платит "
                       "за каждую. Риск варианта — после изменения режима; риск в шапке "
                       "карточки — у текущего режима.")
            st.dataframe(pd.DataFrame([
                {"что меняем": describe_moves(c),
                 "сера": round(c.predicted_quality.get("product_sulfur_mgkg", float("nan")), 2),
                 # риск и тяжесть режима у каждого варианта СВОИ — именно между
                 # ними и выбирает технолог
                 # «риск варианта», а не просто «риск»: в шапке рекомендации риск
                 # текущего режима считает калиброванный классификатор, путать их нельзя
                 "риск варианта": f"{c.spec_risk.get('product_sulfur_mgkg', float('nan')):.0%}",
                 "тяжесть": round(c.severity_index or 0, 3),
                 # Т95 — второй обязательный показатель. Без неё оператор видит
                 # только половину размена: вариант с лучшей серой может быть
                 # хуже по разгонке, и выбирать вслепую он не должен.
                 "Т95": (None if c.predicted_quality.get("product_t95_c") is None
                         else round(c.predicted_quality["product_t95_c"], 1)),
                 "выпуск": round(c.throughput or 0, 1),
                 "энергия": round(c.energy_proxy or 0, 1),
                 "с запасом": "да" if c.guaranteed else "нет"}
                for c in rec.alternatives]), hide_index=True, width="stretch")
            # Все допустимые варианты пересчитываются тем же оптимизатором (доли
            # секунды): решение не меняется, это только картинка поля выбора.
            try:
                field = system.optimizer.propose(state, q, r)
            except Exception:                               # noqa: BLE001
                field = rec.alternatives
            hold = system.optimizer.last_hold()
            st.plotly_chart(pareto_chart(field or rec.alternatives, chosen=rec.action,
                                         hold=hold, limit=limit), width="stretch")
        else:
            st.info("Сравнивать нечего: в этот момент система не предлагает изменений "
                    "режима.")

    with tabs[2]:
        # ---------- данные и достоверность --------------------------------- #
        st.subheader("Качество продукта и достоверность данных")
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(quality_chart(sb, lab, pd.Timestamp(ts), limit),
                            width="stretch")
        with right:
            dq = state.data_quality
            st.metric("Пропусков в срезе", f"{dq.missing_share:.1%}")
            gap = getattr(rec, "analyzer_gap", None)
            if gap:
                st.warning(f"Расхождение анализаторов {gap['расхождение, мг/кг']:g} мг/кг "
                           f"(ПАК {gap['пак, мг/кг']:g}, Q21 {gap['q21, мг/кг']:g}): при "
                           f"таком расхождении оперативное значение на истории ошибалось "
                           f"втрое сильнее обычного.")
            if dq.notes:
                for note in dq.notes:
                    st.markdown(f"- {note}")
            else:
                st.markdown("Замечаний к данным нет.")

    with tabs[3]:
        # ---------- надёжность: из чего сложилась тяжесть режима ------------ #
        panel = reliability_panel(r, load_catalyst_report(ROOT), ts,
                                  basis=catalyst_basis(system.reliability),
                                  dp=dp_mode(system.reliability))
        severity_block(panel["тяжесть"])
        catalyst_block(panel["катализатор"])

    with tabs[4]:
        # ---------- партия с превышением: гашение в резервуаре --------------- #
        if rec.tank_rescue:
            st.subheader("Партия с превышением: что примет резервуар")
            rescue = rec.tank_rescue
            cols = st.columns(3)
            cols[0].metric("Сера партии, мг/кг", f"{rescue['сера партии, мг/кг']:g}")
            cols[1].metric("Доля партии", f"{rescue['доля партии']:.0%}"
                           if rescue["возможно"] else "—")
            cols[2].metric("Тонн резервуара на тонну партии",
                           f"{rescue.get('на тонну партии нужно тонн резервуара', '—')}")
            st.caption(
                f"Практика установки: партию гасят топливом с запасом по сере "
                f"(резервуар {rescue['сера резервуара, мг/кг']:g} мг/кг — ДОПУЩЕНИЕ, "
                f"данных по паркам в пакете нет). Цель смеси "
                f"{rescue['целевая сера смеси, мг/кг']:g} мг/кг. Это операция товарного "
                "парка, а не замена правке режима.")
            if not rescue["возможно"]:
                st.warning(str(rescue.get("почему", "")))

        # ---------- смешение ----------------------------------------------- #
        if rec.blend is not None:
            st.subheader("Смешение товарного ДТ")
            left, right = st.columns([2, 3])
            with left:
                st.dataframe(pd.DataFrame([
                    {"компонент": name, "доля, %": round(share * 100, 2)}
                    for name, share in rec.blend.fractions.items()]),
                    hide_index=True, width="stretch")
                st.markdown(f"**Сумма долей:** {rec.blend.fractions_sum() * 100:.1f} % "
                            f"(жёсткое требование ТЗ)")
                st.markdown(f"**Выпуск смеси:** {rec.blend.throughput_tph:.1f} т/ч")
                if rec.blend.cetane_improver_pct > 0:
                    # Присадка дороже топлива в 100 раз, поэтому её доза и её цена —
                    # не деталь рецептуры, а отдельное экономическое решение.
                    st.markdown(
                        f"**Цетаноповышающая присадка:** "
                        f"{rec.blend.cetane_improver_pct:.3f} % массы — минимальная доза "
                        f"под норматив. Стоит {rec.blend.improver_cost_share * 100:.1f} % "
                        f"цены тонны, поэтому чистая ценность "
                        f"**{rec.blend.net_value_tph:.1f} т/ч** при выпуске "
                        f"{rec.blend.throughput_tph:.1f}.")
                cetane = rec.blend.properties.get("cetane_number")
                if cetane is not None:
                    st.markdown(f"**Цетановое число смеси:** {cetane:.1f}")
            with right:
                basis = rec.blend.basis_sulfur_mgkg
                if basis is not None:
                    st.markdown(f"Рецептура посчитана на **прогнозной** сере "
                                f"{basis:.2f} мг/кг — то есть для того режима, который "
                                f"рекомендован, а не для прошедшего.")
                if rec.blend.feasible:
                    st.success("Рецептура проходит жёсткие проверки.")
                else:
                    st.error("Допустимой рецептуры нет: " + "; ".join(rec.blend.violations))
                if rec.blend.uncertified:
                    # «Не посчитали» — это не «годно». Показываем отдельно от нарушений:
                    # там система знает, что плохо, здесь — что не знает вовсе.
                    st.warning("Не подтверждено по обязательным показателям: "
                               + ", ".join(rec.blend.uncertified)
                               + ". Анализов у компонентов нет — годной рецептуру "
                                 "называть нельзя, пока их не сделают.")
                for note in rec.blend.notes:
                    st.markdown(f"- {readable(note)}")
        if rec.blend is None:
            st.info("Смешение не считалось: совета в этот момент нет.")

    with tabs[5]:
        st.subheader("Цикл решения: кто что сказал")
        trace_rows(rec)
        ask_block(cfg, rec)
        with st.expander("Полный ответ агентов (JSON)"):
            st.json({"quality": q.model_dump(mode="json"),
                     "reliability": r.model_dump(mode="json"),
                     "recommendation": rec.model_dump(mode="json")})


if __name__ == "__main__":
    main()
