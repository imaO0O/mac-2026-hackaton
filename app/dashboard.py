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
from nefte.data.loaders import lims_series  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402

st.set_page_config(page_title="МАС «Нефтекод»", layout="wide")


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
    fig.add_trace(go.Scatter(x=pak.index, y=pak.values, name="ПАК",
                             line=dict(color="#4C78A8", width=1)))
    if frozen.any():
        stuck = pak[frozen.reindex(pak.index, fill_value=False)]
        fig.add_trace(go.Scatter(x=stuck.index, y=stuck.values, name="ПАК завис",
                                 mode="markers", marker=dict(color="#E45756", size=3)))
    lab_w = lab.loc[lo:hi]
    fig.add_trace(go.Scatter(x=lab_w.index, y=lab_w.values, name="ЛИМС",
                             mode="markers", marker=dict(color="#111", size=9,
                                                         symbol="diamond")))
    fig.add_hline(y=limit, line_dash="dash", line_color="#E45756",
                  annotation_text=f"предел {limit} мг/кг")
    fig.add_vline(x=ts, line_color="#54A24B")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="сера, мг/кг", legend=dict(orientation="h"))
    return fig


def pareto_chart(candidates) -> go.Figure:
    """Фронт Парето: запас по качеству против тяжести режима."""
    fig = go.Figure()
    front = [c for c in candidates if c.pareto_rank == 0]
    rest = [c for c in candidates if c.pareto_rank != 0]
    for group, name, color, size in ((rest, "варианты", "#B0B0B0", 6),
                                     (front, "фронт Парето", "#4C78A8", 10)):
        if not group:
            continue
        fig.add_trace(go.Scatter(
            x=[c.predicted_quality.get("product_sulfur_mgkg") for c in group],
            y=[c.throughput for c in group], mode="markers", name=name,
            marker=dict(color=color, size=size),
            text=[c.id for c in group],
            hovertemplate="%{text}<br>сера %{x:.2f}<br>выпуск %{y:.1f}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="прогноз серы, мг/кг", yaxis_title="выпуск, т/ч",
                      legend=dict(orientation="h"))
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


def trace_block(rec) -> None:
    """Цикл решения по шагам: какой агент что вернул и какое правило сработало."""
    st.subheader("Цикл решения: кто что сказал")
    for number, step in enumerate(rec.trace, start=1):
        left, right = st.columns([1, 4])
        left.markdown(f"**{number}. {step.agent}**")
        right.markdown(step.summary)
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


def main() -> None:
    cfg, sb, system = load_system()
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]
    lab = load_lab(cfg["quality"]["target"]["lims_source"])

    st.title("МАС «Нефтекод»: АВТ → гидроочистка → блендинг")

    with st.sidebar:
        st.header("Момент времени")
        mode = st.radio("Что смотреть", ["Сцены защиты", "Любой момент"])
        point = None
        if mode == "Сцены защиты":
            scenes = scene_moments()
            index = st.selectbox("Сцена", range(len(scenes)),
                                 format_func=lambda i: scenes[i][0])
            _, ts, point = scenes[index]
        else:
            day = st.date_input("Дата", value=pd.Timestamp("2026-02-18").date(),
                                min_value=sb.ht.index[0].date(),
                                max_value=sb.ht.index[-1].date())
            hour = st.slider("Час", 0, 23, 0)
            ts = pd.Timestamp(day) + pd.Timedelta(hours=hour)
        st.caption("Данные и модель кэшируются, пересчёт занимает доли секунды.")

    if point:
        st.info(f"**Что показывает сцена.** {point}")

    state = sb.build(pd.Timestamp(ts))
    # Лимит частоты воздействий — состояние оркестратора, и осмыслен он только в
    # хронологическом прогоне. На дашборде оператор двигает ползунок как хочет, в
    # том числе назад: без сброса карточка зависела бы от истории просмотра, а не
    # от выбранного момента.
    system._last_action_ts = None
    rec = system.run(state)
    q = system.quality.assess(state)
    r = system.reliability.assess(state)

    # ---------- строка состояния -------------------------------------- #
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    pak = state.quality.get("pak_sulfur_ppm")
    lims = state.quality.get("lims_sulfur_mgkg")
    c1.metric("ПАК, мг/кг", f"{pak.value:.2f}" if pak and pak.value else "—",
              "завис" if pak and pak.is_frozen else None, delta_color="inverse")
    c2.metric("ЛИМС, мг/кг", f"{lims.value:.2f}" if lims and lims.value else "—",
              f"возраст {lims.age_hours:.0f} ч" if lims and lims.age_hours else None,
              delta_color="off")
    c3.metric("Прогноз, мг/кг",
              f"{q.predictions.get('product_sulfur_mgkg', float('nan')):.2f}"
              if q.predictions else "—",
              f"риск {q.spec_risk.get('product_sulfur_mgkg', 0):.0%}" if q.spec_risk else None,
              delta_color="inverse")
    c4.metric("Тяжесть режима", f"{r.severity_index:.2f}", r.risk_class, delta_color="off")
    # Т95 и цетановое число — второй и третий обязательные показатели. Без них в
    # строке состояния оператор видел одну серу и не знал, что другой показатель
    # уже за пределом.
    t95 = q.predictions.get("product_t95_c")
    t95_limit = cfg["spec"]["t95_c"]["max"]
    c5.metric("Т95 по анализу, °C", "—" if t95 is None else f"{t95:.1f}",
              None if t95 is None else (f"выше {t95_limit:.0f}" if t95 > t95_limit
                                        else f"предел {t95_limit:.0f}"),
              delta_color="inverse" if t95 is not None and t95 > t95_limit else "off")
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
                  delta_color="inverse" if value < cetane_min else "off")
    else:
        c6.metric("ЦЧ ГО ДТ", "—")

    # ---------- рекомендация ------------------------------------------ #
    st.subheader("Рекомендация оператору")
    if rec.abstained:
        st.error(f"**Рекомендации нет.** {rec.abstain_reason}")
    else:
        moves = {t: d for t, d in (rec.action.deltas if rec.action else {}).items()
                 if abs(d) > 1e-6}
        box = st.success if not moves else (
            st.warning if rec.action and not rec.action.guaranteed else st.info)
        box(f"**{rec.problem}**")

        left, right = st.columns([2, 3])
        with left:
            st.markdown("**Действие**")
            if moves:
                # точность — та, при которой ход виден: давление двигается на
                # тысячные МПа, и при двух знаках «Δ 0» читалось как «не трогать»
                def digits(delta: float) -> int:
                    return 3 if abs(delta) < 0.01 else 2
                st.dataframe(pd.DataFrame([
                    {"тег": t, "сейчас": f"{rec.action.moves[t] - d:.{digits(d)}f}",
                     "рекомендуется": f"{rec.action.moves[t]:.{digits(d)}f}",
                     "Δ": f"{d:+.{digits(d)}f}"}
                    for t, d in moves.items()]), hide_index=True, use_container_width=True)
            else:
                st.write("Изменение уставок не требуется.")
            st.markdown("**Ожидаемый эффект**")
            st.write(effect_text(rec.expected_effect))
        with right:
            st.markdown("**Проверенные ограничения**")
            for item in rec.checked_constraints:
                st.markdown(f"- {item}")
            st.markdown(f"**Уверенность:** {rec.confidence:.2f}")
            st.markdown(f"**Почему:** {rec.explanation}")

    trace_block(rec)
    ask_block(cfg, rec)

    # ---------- данные и достоверность --------------------------------- #
    st.subheader("Качество продукта и достоверность данных")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(quality_chart(sb, lab, pd.Timestamp(ts), limit),
                        use_container_width=True)
    with right:
        dq = state.data_quality
        st.metric("Пропусков в срезе", f"{dq.missing_share:.1%}")
        if dq.notes:
            for note in dq.notes:
                st.markdown(f"- {note}")
        else:
            st.markdown("Замечаний к данным нет.")
        if r.factors:
            st.markdown("**Факторы тяжести режима**")
            st.bar_chart(pd.Series(r.factors, name="вклад"))

    # ---------- смешение ----------------------------------------------- #
    if rec.blend is not None:
        st.subheader("Смешение товарного ДТ")
        left, right = st.columns([2, 3])
        with left:
            st.dataframe(pd.DataFrame([
                {"компонент": name, "доля, %": round(share * 100, 2)}
                for name, share in rec.blend.fractions.items()]),
                hide_index=True, use_container_width=True)
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
                st.markdown(f"- {note}")

    # ---------- альтернативы ------------------------------------------- #
    if rec.alternatives:
        st.subheader("Альтернативы и фронт Парето")
        left, right = st.columns([2, 3])
        with left:
            st.dataframe(pd.DataFrame([
                {"вариант": c.id,
                 "сера": round(c.predicted_quality.get("product_sulfur_mgkg", float("nan")), 2),
                 # риск и тяжесть режима у каждого варианта СВОИ — именно между
                 # ними и выбирает технолог
                 # «по суррогату» в заголовке не педантизм: в шапке рекомендации
                 # риск считает калиброванный классификатор, и путать их нельзя
                 "риск (суррогат)": round(c.spec_risk.get("product_sulfur_mgkg",
                                                          float("nan")), 3),
                 "тяжесть": round(c.severity_index or 0, 3),
                 # Т95 — второй обязательный показатель. Без неё оператор видит
                 # только половину размена: вариант с лучшей серой может быть
                 # хуже по разгонке, и выбирать вслепую он не должен.
                 "Т95": (None if c.predicted_quality.get("product_t95_c") is None
                         else round(c.predicted_quality["product_t95_c"], 1)),
                 "выпуск": round(c.throughput or 0, 1),
                 "энергия": round(c.energy_proxy or 0, 1),
                 "с запасом": "да" if c.guaranteed else "нет"}
                for c in rec.alternatives]), hide_index=True, use_container_width=True)
        with right:
            st.plotly_chart(pareto_chart(rec.alternatives + ([rec.action] if rec.action else [])),
                            use_container_width=True)

    with st.expander("Полный ответ агентов (JSON)"):
        st.json({"quality": q.model_dump(mode="json"),
                 "reliability": r.model_dump(mode="json"),
                 "recommendation": rec.model_dump(mode="json")})


if __name__ == "__main__":
    main()
