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


def main() -> None:
    cfg, sb, system = load_system()
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]
    lab = load_lab(cfg["quality"]["target"]["lims_source"])

    st.title("МАС «Нефтекод»: АВТ → гидроочистка → блендинг")

    with st.sidebar:
        st.header("Момент времени")
        windows = cfg["demo_windows"]
        titles = {
            "stable": "устойчивый режим",
            "quality_risk": "риск по качеству",
            "bad_data_frozen_pak": "завис поточный анализатор",
            "bad_data_lims_outlier": "выброс в лаборатории",
        }
        choice = st.selectbox("Демо-сценарий", list(windows),
                              format_func=lambda k: f"{titles.get(k, k)}")
        lo, hi = pd.Timestamp(windows[choice][0]), pd.Timestamp(windows[choice][1])
        ts = st.slider("Время", min_value=lo.to_pydatetime(), max_value=hi.to_pydatetime(),
                       value=lo.to_pydatetime(), step=pd.Timedelta(hours=6).to_pytimedelta(),
                       format="DD.MM.YYYY HH:mm")
        st.caption("Данные и модель кэшируются, пересчёт занимает доли секунды.")

    state = sb.build(pd.Timestamp(ts))
    rec = system.run(state)
    q = system.quality.assess(state)
    r = system.reliability.assess(state)

    # ---------- строка состояния -------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
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
                st.dataframe(pd.DataFrame([
                    {"тег": t, "сейчас": round(rec.action.moves[t] - d, 2),
                     "рекомендуется": round(rec.action.moves[t], 2), "Δ": round(d, 2)}
                    for t, d in moves.items()]), hide_index=True, use_container_width=True)
            else:
                st.write("Изменение уставок не требуется.")
            st.markdown("**Ожидаемый эффект**")
            st.json(rec.expected_effect, expanded=True)
        with right:
            st.markdown("**Проверенные ограничения**")
            for item in rec.checked_constraints:
                st.markdown(f"- {item}")
            st.markdown(f"**Уверенность:** {rec.confidence:.2f}")
            st.markdown(f"**Почему:** {rec.explanation}")

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
