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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.agents.reliability import ReliabilityAgent  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import clean_lims_sulfur  # noqa: E402
from nefte.data.loaders import lims_series, load_pak  # noqa: E402
from nefte.data.validity import analyzer_health  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402


def severity_series(agent: ReliabilityAgent, avt: pd.DataFrame, ht: pd.DataFrame,
                    freq: str = "1h") -> pd.DataFrame:
    """Векторный расчёт severity по всей истории — тот же расчёт, что в агенте."""
    temps = [t for t in agent.REACTOR_TEMPS if t in ht.columns]
    grid = ht.resample(freq, label="right", closed="right").last()
    grid_avt = avt.resample(freq, label="right", closed="right").last()

    factors = pd.DataFrame(index=grid.index)
    if temps:
        factors["wabt"] = grid[temps].mean(axis=1).map(
            lambda v: agent.norms.normalize("wabt", None if pd.isna(v) else float(v)))
    if agent.DP_TAG in grid.columns:
        factors["dp_r202"] = grid[agent.DP_TAG].map(
            lambda v: agent.norms.normalize(agent.DP_TAG, None if pd.isna(v) else float(v)))
    if agent.FURNACE_TAG in grid_avt.columns:
        factors["furnace"] = grid_avt[agent.FURNACE_TAG].map(
            lambda v: agent.norms.normalize(agent.FURNACE_TAG, None if pd.isna(v) else float(v)))
    if agent.ramp_series is not None:
        factors["ramp"] = agent.ramp_series.resample(
            freq, label="right", closed="right").max().clip(0, 1)

    weights = pd.Series({k: agent.WEIGHTS[k] for k in factors.columns})
    mask = factors.notna()
    total = mask.mul(weights, axis=1).sum(axis=1)
    severity = (factors.fillna(0).mul(weights, axis=1).sum(axis=1) / total.replace(0, np.nan))
    factors["severity"] = severity.clip(0, 1)
    return factors


def main() -> int:
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
    agent = ReliabilityAgent.from_history(sb.avt, sb.ht, cfg)
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
