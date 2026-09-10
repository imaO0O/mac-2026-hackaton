"""Бэктест агента надёжности и детекторов достоверности (участник 2). Только CPU.

    python scripts/backtest_reliability.py

Проверяет две вещи, которые обычно принимают на веру:

1. **Достоверность данных.** Сколько отсчётов забраковано по каждому тегу и по
   каким причинам, какие теги мертвы, какие подозрительны, когда отказывал
   поточный анализатор. Результат — таблица в reports/ и раздел документации.

2. **Осмысленность severity.** Прокси-индекс тяжести режима не имеет разметки,
   поэтому его нельзя «проверить по метке». Но можно спросить: связан ли он с тем,
   что произойдёт дальше? Считаем ROC-AUC severity как предиктора превышения
   спецификации в следующем лабораторном анализе. Если AUC около 0.5 — индекс
   бесполезен, и честнее это знать до защиты, а не после вопроса жюри.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.reliability import ReliabilityAgent, regime_anomaly_frame
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import clean_lims_sulfur  # noqa: E402
from nefte.data.loaders import (  # noqa: E402
    lims_series,
    load_pak,
    load_telemetry,  # noqa: E402
)
from nefte.data.validity import analyzer_health  # noqa: E402
from nefte.models.regime import FEED, RECYCLE_GAS  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402


def severity_series(agent: ReliabilityAgent, avt: pd.DataFrame, ht: pd.DataFrame,
                    freq: str = "1h") -> pd.DataFrame:
    """Severity по всей истории — расчётом САМОГО агента, а не копией.

    Здесь была вторая реализация, считавшая четыре фактора из шести: без наработки
    катализатора и без нетипичности режима, то есть без четверти веса. Отчёт при
    этом утверждал, что расчёт тот же. Теперь вызывается метод агента, и разойтись
    им больше негде.
    """
    temps = [t for t in agent.REACTOR_TEMPS if t in ht.columns]
    grid = ht.resample(freq, label="right", closed="right").last()
    grid_avt = avt.resample(freq, label="right", closed="right").last()

    frame = pd.DataFrame(index=grid.index)
    if temps:
        frame["wabt"] = grid[temps].mean(axis=1)
    if agent.DP_TAG in grid.columns:
        frame[agent.DP_TAG] = grid[agent.DP_TAG]
    if agent.FURNACE_TAG in grid_avt.columns:
        frame[agent.FURNACE_TAG] = grid_avt[agent.FURNACE_TAG]

    anomaly_frame = (regime_anomaly_frame(avt, ht)
                     .resample(freq, label="right", closed="right").last())
    severity, factors = agent.severity_series(frame, anomaly_frame, with_factors=True)
    factors = factors.copy()
    factors["severity"] = severity
    return factors


def explain_health_by_downtime(agent, health: pd.DataFrame) -> dict:
    """Какая часть «отказов анализатора» объясняется остановом установки.

    Зависший поточный анализатор — не всегда неисправность: если через него
    ничего не течёт, он держит последнее значение. Разделять эти случаи важно,
    иначе исправный прибор попадёт в список отказавших.
    """
    if agent.down_series is None or health.empty:
        return {}
    rows = []
    for _, ep in health.iterrows():
        window = agent.down_series.loc[ep["start"]:ep["end"]]
        share = float(window.mean()) if len(window) else 0.0
        rows.append({"start": str(ep["start"])[:16], "hours": round(float(ep["hours"]), 1),
                     "value": round(float(ep["value"]), 2),
                     "downtime_share_%": round(share * 100, 0)})
    frame = pd.DataFrame(rows)
    total = float(frame["hours"].sum())
    explained = float((frame["hours"] * frame["downtime_share_%"] / 100).sum())
    return {
        "total_hours": round(total, 0),
        "explained_by_downtime_hours": round(explained, 0),
        "explained_%": round(explained / total * 100, 0) if total else 0,
        "episodes": frame.sort_values("hours", ascending=False).head(5).to_dict("records"),
    }


def describe_campaigns(agent) -> dict:
    """Сколько циклов между остановами удалось восстановить и какой они длины."""
    if agent.run_hours is None:
        return {}
    run = agent.run_hours.dropna()
    # новый цикл начинается там, где счётчик наработки сбрасывается
    resets = run.diff() < 0
    lengths = []
    current_start = run.index[0]
    for ts in run.index[resets.to_numpy()]:
        lengths.append((ts - current_start).total_seconds() / 86400)
        current_start = ts
    lengths.append((run.index[-1] - current_start).total_seconds() / 86400)
    lengths = [x for x in lengths if x > 0]
    if not lengths:
        return {}
    series = pd.Series(lengths)
    return {"n_campaigns": len(lengths),
            "median_days": round(float(series.median()), 1),
            "max_days": round(float(series.max()), 1),
            "scale_hours_p95": round(float(agent.run_hours_scale or 0), 1)}


def describe_anomalies(agent, avt: pd.DataFrame, ht: pd.DataFrame, cfg: dict) -> dict:
    """Как часто многомерный детектор считает режим нетипичным."""
    if not agent.detector.fitted:
        return {}
    frame = pd.DataFrame(index=ht.index)
    temps = [t for t in agent.REACTOR_TEMPS if t in ht.columns]
    if temps:
        frame["wabt"] = ht[temps].mean(axis=1)
    if agent.DP_TAG in ht.columns:
        frame[agent.DP_TAG] = ht[agent.DP_TAG]
    if agent.FURNACE_TAG in avt.columns:
        frame[agent.FURNACE_TAG] = avt[agent.FURNACE_TAG]
    if FEED in ht.columns:
        frame["feed"] = ht[FEED]
        if RECYCLE_GAS in ht.columns:
            denom = ht[FEED].where(ht[FEED] > ht[FEED].median() * 0.1)
            frame["h2_oil"] = ht[RECYCLE_GAS] / denom
    if "P13" in ht.columns:
        frame["pressure"] = ht["P13"]

    flags = agent.detector.flags(frame.dropna())
    tr_lo, tr_hi = cfg["split"]["train"]
    te_lo, te_hi = cfg["split"]["test"]
    train_part = flags.loc[tr_lo:tr_hi]
    test_part = flags.loc[te_lo:te_hi]
    return {
        "threshold": round(agent.detector.threshold, 2),
        "columns": agent.detector.columns,
        "train_%": round(float(train_part.mean() * 100), 2) if len(train_part) else None,
        "test_%": round(float(test_part.mean() * 100), 2) if len(test_part) else None,
    }


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = cfg["spec"]["product_sulfur_mgkg"]["max"]
    print("[1/3] загрузка и очистка…")
    sb = StateBuilder(cfg)

    # ---------- 1. отчёт по достоверности ---------------------------------
    report: dict = {"validity": {}, "severity": {}}
    for unit, validity in (("avt", sb.avt_validity), ("ht", sb.ht_validity)):
        summary = validity.summary()
        bad = summary[summary["итого_%"] > 0]
        report["validity"][unit] = {
            "n_tags": int(len(summary)),
            "dead_tags": validity.dead_tags,
            "suspicious_negative": validity.suspicious_tags,
            "worst": {k: float(v) for k, v in bad["итого_%"].head(8).items()},
            "mean_bad_share_pct": float(summary["итого_%"].mean().round(3)),
        }
        print(f"      {unit}: тегов {len(summary)}, мёртвых {len(validity.dead_tags)}, "
              f"подозрительных по знаку {len(validity.suspicious_tags)}, "
              f"средний брак {summary['итого_%'].mean():.2f} %")

    health = analyzer_health(load_pak()["sulfur_ppm"], cfg)
    report["analyzer_failures"] = {
        "n_episodes": int(len(health)),
        "total_hours": float(health["hours"].sum()) if len(health) else 0.0,
        "longest": [
            {"start": str(r.start), "end": str(r.end), "value": float(r.value),
             "hours": float(r.hours)}
            for r in health.head(5).itertuples()
        ],
    }
    print(f"      отказы ПАК: {len(health)} эпизодов, "
          f"{health['hours'].sum():.0f} ч суммарно")

    # ---------- 2. severity ------------------------------------------------
    print("[2/3] severity по истории…")
    agent = ReliabilityAgent.from_history(sb.avt, sb.ht, cfg,
                                         raw_ht=load_telemetry("ht"))
    factors = severity_series(agent, sb.avt, sb.ht)
    sev = factors["severity"].dropna()

    lab = clean_lims_sulfur(lims_series(cfg["quality"]["target"]["lims_source"]))
    test_lo, test_hi = cfg["split"]["test"]

    # severity в момент, доступный ДО анализа (за час до отбора пробы)
    paired = pd.merge_asof(
        lab.rename("sulfur").reset_index().rename(columns={"index": "ts", "ts": "ts"}),
        sev.rename("severity").reset_index().rename(columns={"date": "ts", "index": "ts"}),
        on="ts", direction="backward", tolerance=pd.Timedelta("6h"),
        allow_exact_matches=False).dropna()

    # ---------- 2б. новые факторы: наработка и аномалии --------------------
    campaigns = describe_campaigns(agent)
    if campaigns:
        print(f"      циклов между остановами: {campaigns['n_campaigns']}, "
              f"медиана {campaigns['median_days']} сут, "
              f"самый длинный {campaigns['max_days']} сут")
    downtime_share = float(agent.down_series.mean() * 100) if agent.down_series is not None else 0
    print(f"      установка не в работе: {downtime_share:.1f} % времени")
    explained = explain_health_by_downtime(agent, health)
    if explained:
        print(f"      из {explained['total_hours']:.0f} ч зависаний ПАК на остановы "
              f"приходится {explained['explained_by_downtime_hours']:.0f} ч "
              f"({explained['explained_%']:.0f} %) — прибор держал последнее значение, "
              f"потому что через него ничего не шло")
    anomalies = describe_anomalies(agent, sb.avt, sb.ht, cfg)
    if anomalies:
        print(f"      детектор аномалий: порог {anomalies['threshold']}, "
              f"нетипично train {anomalies['train_%']} %, test {anomalies['test_%']} %")
    report["downtime"] = {"share_%": round(downtime_share, 2), **explained}
    report["campaigns"] = campaigns
    report["anomaly_detector"] = anomalies
    report["risk_thresholds"] = [round(agent.thresholds[0], 3), round(agent.thresholds[1], 3)]
    print(f"      пороги risk_class по train: "
          f"{agent.thresholds[0]:.2f} / {agent.thresholds[1]:.2f}")

    print("[3/3] оценка…")
    from sklearn.metrics import roc_auc_score

    for split, (lo, hi) in (("train", cfg["split"]["train"]),
                            ("test", (test_lo, test_hi))):
        part = paired[(paired["ts"] >= lo) & (paired["ts"] <= pd.Timestamp(hi) + pd.Timedelta(days=1))]
        over = (part["sulfur"] > limit).astype(int)
        if over.nunique() < 2:
            continue
        auc = float(roc_auc_score(over, part["severity"]))
        report["severity"][split] = {
            "n": int(len(part)),
            "n_over_limit": int(over.sum()),
            "roc_auc": round(auc, 3),
            "mean_severity_ok": round(float(part.loc[over == 0, "severity"].mean()), 3),
            "mean_severity_over": round(float(part.loc[over == 1, "severity"].mean()), 3),
        }
        print(f"      {split}: n={len(part)}, превышений={over.sum()}, "
              f"ROC-AUC severity={auc:.3f}, "
              f"severity в норме {part.loc[over == 0, 'severity'].mean():.3f} / "
              f"при превышении {part.loc[over == 1, 'severity'].mean():.3f}")

    # ---------- 3. проверка конфликта целей --------------------------------
    # Жёсткость режима и качество тянут в разные стороны: выше температура —
    # ниже сера, но тяжелее режим для катализатора. Если это так, severity будет
    # предсказывать превышения ХУЖЕ случайного, а обратный индекс — лучше.
    tradeoff = {}
    for split, (lo, hi) in (("train", cfg["split"]["train"]), ("test", (test_lo, test_hi))):
        block = report["severity"].get(split)
        if block:
            tradeoff[split] = {
                "auc_severity": block["roc_auc"],
                "auc_inverse": round(1 - block["roc_auc"], 3),
            }
    report["tradeoff_quality_vs_severity"] = tradeoff

    # Дезактивация катализатора: перепад давления на Р-202 должен расти тем
    # быстрее, чем тяжелее режим. Проверяем связь недельных средних.
    if ReliabilityAgent.DP_TAG in sb.ht.columns:
        weekly_sev = sev.resample("7D").mean()
        weekly_dp = sb.ht[ReliabilityAgent.DP_TAG].resample("7D").mean().diff()
        pair = pd.concat([weekly_sev.rename("sev"), weekly_dp.rename("dp")], axis=1).dropna()
        if len(pair) > 10:
            rho = float(pair["sev"].corr(pair["dp"], method="spearman"))
            report["severity"]["dp_growth_spearman"] = round(rho, 3)
            print(f"      связь severity с ростом ΔP Р-202 (недельные средние): "
                  f"Spearman {rho:+.3f}")

    report["severity"]["distribution"] = {
        "low_%": round(float((sev < 0.5).mean() * 100), 1),
        "medium_%": round(float(((sev >= 0.5) & (sev < 0.8)).mean() * 100), 1),
        "high_%": round(float((sev >= 0.8).mean() * 100), 1),
    }
    print(f"      доля времени: low {report['severity']['distribution']['low_%']} %, "
          f"medium {report['severity']['distribution']['medium_%']} %, "
          f"high {report['severity']['distribution']['high_%']} %")

    out = ROOT / "reports" / "reliability_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
