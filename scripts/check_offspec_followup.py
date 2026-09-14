"""Анализ уже выше предела, а система держит режим: права ли она. Только CPU.

    python scripts/check_offspec_followup.py

Зачем. На часовом прогоне теста в сотнях моментов свежий лабораторный анализ выше
10 мг/кг, а оркестратор держит режим: риск модели ниже порога. Карточка при этом
писала «ФАКТ ВНЕ СПЕЦИФИКАЦИИ» и «режим устойчив» одновременно. Прежде чем чинить
решение, надо проверить, ошибается ли оно. Три вопроса:

1. **Как часто это бывает** — по прогонам тестового периода.
2. **Повторяется ли превышение**: после пробы выше предела — насколько чаще
   следующая проба тоже выше, чем после нормальной (по периодам).
3. **Не занижает ли модель риск именно в этих моментах**: средняя вероятность
   модели против фактической частоты превышения в пробах, перед которыми
   последний опубликованный анализ был выше предела.

Если бы модель здесь занижала риск, правильным было бы правило «после свежего
превышения поднимать риск». Результат: reports/offspec_followup.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.cleaning import clean_lims_sulfur  # noqa: E402
from nefte.data.loaders import lims_series, load_lims  # noqa: E402
from nefte.models.dataset import build_feature_matrix, build_training_table  # noqa: E402
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "offspec_followup.json"
RUNS = ("reports/test_period.json", "reports/test_period_step1h.json")
# «свежий» анализ для вопроса 3: предыдущая проба не старше полутора суток
FRESH_HOURS = 36.0


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    limit = float(cfg["spec"]["product_sulfur_mgkg"]["max"])
    delay = float(cfg["quality"]["lims_publication_delay_hours"])
    stale = float(cfg["quality"]["staleness_hours"]["lims"])
    lab = clean_lims_sulfur(lims_series(cfg["quality"]["target"]["lims_source"],
                                        load_lims())).sort_index()
    known = pd.DataFrame({"known_ts": lab.index + pd.Timedelta(hours=delay),
                          "sample_ts": lab.index, "lab": lab.values}).sort_values("known_ts")

    # 1. как часто в прогонах теста
    runs = {}
    for name in RUNS:
        path = ROOT / name
        if not path.exists():
            continue
        rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
        rows["ts"] = pd.to_datetime(rows["ts"])
        j = pd.merge_asof(rows.sort_values("ts"), known, left_on="ts", right_on="known_ts",
                          direction="backward")
        age = (j["ts"] - j["sample_ts"]).dt.total_seconds() / 3600
        hot = j[(j["lab"] > limit) & (age <= stale)]
        runs[Path(name).name] = {
            "моментов": int(len(j)),
            "свежий анализ выше предела": int(len(hot)),
            "исходы": {k: int(v) for k, v in hot["исход"].value_counts().items()},
            "медианный риск при «держим режим»": round(float(
                hot.loc[hot["исход"] == "держим режим", "риск"].median()), 3),
        }
    print("\n1. Свежий анализ выше предела в прогонах теста:")
    for name, r in runs.items():
        print(f"   {name}: {r['свежий анализ выше предела']} из {r['моментов']}, исходы {r['исходы']}")

    # 2. повторяется ли превышение в следующей пробе
    repeat = {}
    for part in ("train", "val", "test"):
        lo, hi = cfg["split"][part]
        over = (lab.loc[lo:hi] > limit).astype(int)
        nxt = over.shift(-1).dropna()
        cur = over.loc[nxt.index] == 1
        repeat[part] = {
            "проб": int(len(over)),
            "после пробы выше предела": round(float(nxt[cur].mean()), 3),
            "таких пар": int(cur.sum()),
            "после нормальной": round(float(nxt[~cur].mean()), 3),
        }
    print("\n2. Следующая проба выше предела:")
    for part, r in repeat.items():
        print(f"   {part:5s} после превышения {r['после пробы выше предела']:.1%} "
              f"({r['таких пар']} пар), после нормальной {r['после нормальной']:.1%}")

    # 3. калибровка модели в пробах после свежего превышения
    model = SulfurModel.load(SulfurModel.default_path(0.0))
    X, y = build_training_table(0.0, features=build_feature_matrix(),
                                train_bounds=tuple(cfg["split"]["train"]))
    risk = model.predict_risk(X[model.features])
    frame = pd.DataFrame({"ts": X.index, "over": (y > limit).values, "risk": risk.values})
    frame = pd.merge_asof(frame.sort_values("ts"), known, left_on="ts", right_on="known_ts",
                          direction="backward", allow_exact_matches=False)
    frame["age"] = (frame["ts"] - frame["sample_ts"]).dt.total_seconds() / 3600
    calib = {}
    for part in ("train", "val", "test"):
        lo, hi = cfg["split"][part]
        f = frame[(frame["ts"] >= lo) & (frame["ts"] < pd.Timestamp(hi) + pd.Timedelta(days=1))
                  & (frame["age"] <= FRESH_HOURS)]
        calib[part] = {}
        for label, g in (("после превышения", f[f["lab"] > limit]),
                         ("после нормальной", f[f["lab"] <= limit])):
            calib[part][label] = {"проб": int(len(g)), "событий": int(g["over"].sum()),
                                  "средний риск модели": round(float(g["risk"].mean()), 3),
                                  "факт выше предела": round(float(g["over"].mean()), 3)}
    print("\n3. Вероятность модели против факта (пробы после свежего анализа):")
    for part, groups in calib.items():
        for label, g in groups.items():
            print(f"   {part:5s} {label:17s} проб {g['проб']:4d}, событий {g['событий']:3d}: "
                  f"риск {g['средний риск модели']:.3f}, факт {g['факт выше предела']:.3f}")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "предел": limit,
                                  "прогоны": runs, "повтор_превышения": repeat,
                                  "калибровка_после_превышения": calib},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
