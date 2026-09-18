"""Держится ли P8 как перепад давления на Р-202 в severity (участник 2). Только CPU.

    python scripts/check_dp_proxy.py

Зачем. Новая таблица тегов 24-2000 (15.09) показала, что фактор «перепад Р-202» в
тяжести режима считался по W10 — а это массовый расход бензина (W10/F1 = 0.745,
плотность). Участник 1 заменил его на P8. Здесь проверяется, что P8 даёт как
признак ТЯЖЕСТИ: перепад на слое катализатора должен расти с закоксовыванием внутри
цикла и падать после замены на свежий, а от нагрузки зависеть гидравлически
(∝ расход²).

Три варианта фактора, одинаково нормированные на обучающий период:

* **W10** — как было (бензин);
* **P8** — как в новой таблице;
* **P8 за вычетом нагрузки** — остаток P8 после a + b·F26², подогнанного на
  обучающем периоде: сопротивление слоя сверх гидравлики.

Результат: reports/dp_proxy.json и таблицы в консоли.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nefte.agents.reliability import ReliabilityAgent  # noqa: E402
from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.loaders import load_telemetry  # noqa: E402
from nefte.pipeline import StateBuilder  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402
from scripts.backtest_reliability import severity_series  # noqa: E402

EVENTS = {"замена 2024-04-17": ("2024-04-17", True), "замена 2026-04-23": ("2026-04-23", True),
          "ремонт 2026-06-30": ("2026-06-30", False)}
CYCLE_1 = ("2024-05-01", "2026-04-10")
LOAD_COLUMN = "P8_без_нагрузки"
REPORT = ROOT / "reports" / "dp_proxy.json"


def daily_working(raw: pd.DataFrame, tag: str) -> pd.DataFrame:
    work = raw["F26"] > 0.5 * raw["F26"].median()
    return pd.DataFrame({"dp": raw[tag], "f": raw["F26"]})[work].resample("1D").median().dropna()


def hydraulic_fit(daily: pd.DataFrame, train: tuple[str, str]) -> tuple[float, float]:
    part = daily.loc[train[0]:train[1]]
    slope, intercept = np.polyfit(part["f"] ** 2, part["dp"], 1)
    return float(intercept), float(slope)


def events_table(daily: pd.DataFrame, fit: tuple[float, float]) -> list[dict]:
    resid = daily["dp"] - (fit[0] + fit[1] * daily["f"] ** 2)
    rows = []
    for name, (day, change) in EVENTS.items():
        t = pd.Timestamp(day)
        before = slice(t - pd.Timedelta(days=45), t - pd.Timedelta(days=10))
        after = slice(t + pd.Timedelta(days=10), t + pd.Timedelta(days=45))
        rows.append({"событие": name, "замена": change,
                     "до": round(float(daily["dp"].loc[before].median()), 3),
                     "после": round(float(daily["dp"].loc[after].median()), 3),
                     "нагрузка до": round(float(daily["f"].loc[before].median()), 0),
                     "нагрузка после": round(float(daily["f"].loc[after].median()), 0),
                     "остаток до": round(float(resid.loc[before].median()), 3),
                     "остаток после": round(float(resid.loc[after].median()), 3)})
    return rows


def cycle_trend(daily: pd.DataFrame) -> dict:
    d = daily.loc[CYCLE_1[0]:CYCLE_1[1]]
    months = np.arange(len(d)) / 30.44
    X = np.column_stack([np.ones(len(d)), d["f"] ** 2, months])
    coef, *_ = np.linalg.lstsq(X, d["dp"].to_numpy(), rcond=None)
    return {"рост за месяц при той же нагрузке": round(float(coef[2]), 5),
            "за цикл": round(float(coef[2] * months[-1]), 4),
            "уровень": round(float(d["dp"].median()), 3),
            "spearman_с_сутками_цикла": round(float(spearmanr(d["dp"], months)[0]), 3)}


def severity_variant(sb, cfg, raw, tag: str, ht: pd.DataFrame) -> dict:
    # Сравниваются варианты УРОВНЯ перепада, поэтому фактор включается здесь явно:
    # в конфиге он выключен (reliability.dp_factor: off), и без этого все три варианта
    # совпали бы — перепада в тяжести не было бы ни в одном.
    ReliabilityAgent.DP_TAG = tag
    local = {**cfg, "reliability": {**(cfg.get("reliability") or {}), "dp_factor": "level"}}
    agent = ReliabilityAgent.from_history(sb.avt, ht, local, raw_ht=raw)
    factors = severity_series(agent, sb.avt, ht)
    sev = factors["severity"].dropna()
    weekly = pd.concat([sev.resample("7D").mean().rename("sev"),
                        ht[tag].resample("7D").mean().diff().rename("dp")], axis=1).dropna()
    medium, high = agent.thresholds
    out = {"настройки": {**(local.get("reliability") or {}), "тег": tag},
           "пороги": [round(medium, 3), round(high, 3)],
           "spearman_severity_с_ростом_перепада": round(float(
               weekly["sev"].corr(weekly["dp"], method="spearman")), 3)}
    for split in ("val", "test"):
        lo, hi = cfg["split"][split]
        part = factors.loc[lo:hi]
        s = part["severity"].dropna()
        top = part.drop(columns="severity").idxmax(axis=1).value_counts(normalize=True)
        out[split] = {"low": round(float((s < medium).mean()), 3),
                      "medium": round(float(((s >= medium) & (s < high)).mean()), 3),
                      "high": round(float((s >= high).mean()), 3),
                      "фактор_перепада_медиана": round(float(part["dp_r202"].median()), 3),
                      "перепад_главный_фактор": round(float(top.get("dp_r202", 0.0)), 3)}
    return out


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    train = tuple(cfg["split"]["train"])
    sb = StateBuilder(cfg)
    raw = load_telemetry("ht")

    report: dict = {"теги": {}}
    for tag in ("W10", "P8"):
        daily = daily_working(raw, tag)
        fit = hydraulic_fit(daily, train)
        report["теги"][tag] = {
            "corr_с_расходом": round(float(daily["dp"].corr(daily["f"])), 3),
            "гидравлика_по_обучению": {"a": round(fit[0], 4), "b_при_F26²": fit[1]},
            "события": events_table(daily, fit), "цикл_1": cycle_trend(daily)}

    daily = daily_working(raw, "P8")
    a, b = hydraulic_fit(daily, train)
    ht = sb.ht.copy()
    ht[LOAD_COLUMN] = ht["P8"] - (a + b * ht["F26"] ** 2)
    original = ReliabilityAgent.DP_TAG
    try:
        report["severity"] = {name: severity_variant(sb, cfg, raw, tag, ht)
                              for name, tag in (("W10", "W10"), ("P8", "P8"),
                                                ("P8 за вычетом нагрузки", LOAD_COLUMN))}
    finally:
        ReliabilityAgent.DP_TAG = original

    for tag, block in report["теги"].items():
        print(f"\n=== {tag}: corr с F26 {block['corr_с_расходом']}, цикл 1: {block['цикл_1']}")
        print(pd.DataFrame(block["события"]).to_string(index=False))
    print("\nВарианты фактора перепада в severity:")
    for name, block in report["severity"].items():
        test = block["test"]
        rho = block["spearman_severity_с_ростом_перепада"]
        print(f"  {name:24s} пороги {block['пороги']}, Spearman {rho:+.3f}; "
              f"тест low/medium/high {test['low']:.1%}/{test['medium']:.1%}/{test['high']:.1%}, "
              f"перепад главный в {test['перепад_главный_фактор']:.0%}")

    # reliability_settings в отчёт не пишется: это сравнение вариантов, у каждого свои
    # настройки (поле «настройки» внутри), и сверять его с конфигом нечего
    payload = {**report_provenance(cfg), **report}
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
