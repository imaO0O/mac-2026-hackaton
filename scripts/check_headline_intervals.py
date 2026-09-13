"""Интервалы у главных чисел защиты. Только CPU.

    python scripts/check_headline_intervals.py

Главные числа проекта — MAE 1.29 и ROC-AUC 0.82 виртуального анализатора, «69 %
пропусков» на рабочей точке, ROC-AUC сети на двух часах — до сих пор назывались без
разброса. А он большой: превышений на тесте 36 среди анализов и 65 среди моментов
решения.

Интервалы — перцентильный блочный бутстрэп по НЕДЕЛЯМ, 90 %. Не формула для
независимых наблюдений: соседние анализы делят состояние установки, а соседние
моменты решения — одни и те же превышения (шаг 12 ч, окно факта 24 ч). Блок в
неделю сохраняет эту связь внутри выборки.

Перед счётом скрипт воспроизводит точечные значения из отчётов и без совпадения
ничего не пишет: интервал вокруг другого числа хуже, чем никакого.

Результат: таблица в консоли и reports/headline_intervals.json.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import roc_auc_score  # noqa: E402

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix, build_training_table  # noqa: E402
from nefte.models.quality_model import Z90, SulfurModel  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

BOOTSTRAP = 2000
REPORT = ROOT / "reports" / "headline_intervals.json"


def week_index(index: pd.DatetimeIndex) -> tuple[np.ndarray, int]:
    labels = pd.Index(index.to_period("W").astype(str))
    codes, uniques = pd.factorize(labels)
    return codes, len(uniques)


def block_bootstrap(codes: np.ndarray, n_weeks: int, stat, key: str) -> np.ndarray:
    """Статистика на выборках недель с возвращением; в каждой — все анализы недели.

    Сид у каждого числа свой и выводится из его имени. С общим генератором
    интервал зависел от того, какие числа посчитаны ДО него: добавление парной
    разницы с ПАК сдвинуло концы остальных интервалов на тысячные, и сверка
    документации с отчётом это поймала.
    """
    rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
    members = [np.flatnonzero(codes == w) for w in range(n_weeks)]
    out = np.empty(BOOTSTRAP)
    for b in range(BOOTSTRAP):
        pick = rng.integers(0, n_weeks, n_weeks)
        idx = np.concatenate([members[w] for w in pick])
        try:
            out[b] = stat(idx)
        except ValueError:          # в выборку не попало ни одного превышения
            out[b] = np.nan
    return out


def interval(values: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(values, [5, 95])
    return [round(float(lo), 3), round(float(hi), 3)]


def model_block(name: str, pred: pd.DataFrame, risk: pd.Series, y: pd.Series,
                limit: float, threshold: float, reference: dict) -> dict:
    over = (y > limit).to_numpy()
    err = (pred["q50"] - y).to_numpy()
    half = (Z90 * pred["sigma"]).to_numpy()
    inside = np.abs(err) <= half
    r = risk.to_numpy()

    def recall_at(t):
        return lambda i: float((r[i] > t)[over[i]].mean()) if over[i].any() else np.nan

    def precision_at(t):
        return lambda i: (float(over[i][r[i] > t].mean()) if (r[i] > t).any() else np.nan)

    stats = {
        "MAE": lambda i: float(np.abs(err[i]).mean()),
        "coverage_80": lambda i: float(inside[i].mean()),
        "roc_auc": lambda i: float(roc_auc_score(over[i], r[i])),
        "recall": recall_at(threshold),
        "precision": precision_at(threshold),
    }
    # Отчёты до исправления хранят «рабочие» precision/recall при пороге, округлённом
    # до двух знаков; новые — при точном и пишут его в spec_threshold. Сверяемся с
    # тем, что реально лежит в отчёте, а интервал считаем при точном пороге.
    reported_at = float(reference.get("spec_threshold", round(threshold, 2)))
    point = {k: f(np.arange(len(y))) for k, f in stats.items()}
    point_reported = {**point, "recall": recall_at(reported_at)(np.arange(len(y))),
                      "precision": precision_at(reported_at)(np.arange(len(y)))}
    # воспроизводим отчёт: иначе интервал был бы вокруг другого числа
    checks = {"MAE": "MAE", "coverage_80": "coverage_80", "roc_auc": "roc_auc",
              "recall": "spec_recall", "precision": "spec_precision"}
    for key, field in checks.items():
        if abs(point_reported[key] - reference[field]) > 1e-6:
            raise SystemExit(f"ОШИБКА: {name}: {key} = {point_reported[key]:.6f}, в отчёте "
                             f"{reference[field]:.6f}. Интервал считать не вокруг чего.")
    codes, n_weeks = week_index(y.index)
    block = {"анализов": int(len(y)), "превышений": int(over.sum()), "недель": n_weeks}
    for key, f in stats.items():
        block[key] = {"значение": round(point[key], 3),
                      "90% интервал": interval(block_bootstrap(codes, n_weeks, f, f"{name}/{key}"))}
    block["порог"] = round(threshold, 4)
    if abs(reported_at - threshold) > 1e-9:
        block["в отчёте при округлённом пороге"] = {
            "порог": reported_at, "recall": round(point_reported["recall"], 3),
            "precision": round(point_reported["precision"], 3)}
    return block


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    out: dict = {}

    feats = build_feature_matrix()
    for horizon in (0, 2):
        model = SulfurModel.load(SulfurModel.default_path(horizon))
        X, y = build_training_table(horizon_hours=horizon, features=feats,
                                    train_bounds=tuple(cfg["split"]["train"]))
        test = time_split(X.index, cfg)["test"].to_numpy()
        reference = json.loads((ROOT / "reports" / f"quality_metrics_h{horizon}.json")
                               .read_text(encoding="utf-8"))["splits"]["test"]["model"]
        name = f"бустинг h{horizon}"
        pred = model.predict_frame(X[test])
        out[name] = model_block(name, pred, model.predict_risk(X[test]),
                                y[test], model.limit, model.alarm_threshold, reference)
        # Разница с поточным анализатором — ПАРНО, на одних и тех же анализах: у
        # каждой MAE по отдельности интервал широк, а разница может быть точнее.
        yt, pak = y[test], X[test]["pak_sulfur"]
        ok = pak.notna().to_numpy()
        e_model = np.abs((pred["q50"] - yt).to_numpy())[ok]
        e_pak = np.abs((pak - yt).to_numpy())[ok]
        codes, n_weeks = week_index(yt.index[ok])
        diff = lambda i: float(e_model[i].mean() - e_pak[i].mean())  # noqa: E731
        out[name]["MAE минус MAE ПАК"] = {
            "значение": round(diff(np.arange(ok.sum())), 3),
            "90% интервал": interval(block_bootstrap(codes, n_weeks, diff, f"{name}/MAE-ПАК")),
            "анализов с ПАК": int(ok.sum())}

    try:
        from nefte.models.sequence import SulfurSequenceModel
        for tag in ("tcn48_pre_h2", "tcn48_pre_s100_h2", "tcn48_pre_s200_h2"):
            path = ROOT / "models" / f"sulfur_seq_{tag}"
            if not (path / "meta.json").exists():
                continue
            seq = SulfurSequenceModel.load(path).attach(feats)
            X, y = build_training_table(horizon_hours=2, features=feats,
                                        train_bounds=tuple(cfg["split"]["train"]))
            yt = y[time_split(X.index, cfg)["test"].to_numpy()]
            pred = seq.predict_index(pd.DatetimeIndex(yt.index))
            ok = pred["q50"].notna().to_numpy()
            yt, pred = yt[ok], pred[ok]
            from nefte.models.quality_model import interval_risk
            risk = interval_risk(pred, seq.limit)
            over = (yt > seq.limit).to_numpy()
            codes, n_weeks = week_index(yt.index)
            r = risk.to_numpy()
            auc = float(roc_auc_score(over, r))
            reference = json.loads((ROOT / "reports" / f"sequence_metrics_{tag}.json")
                                   .read_text(encoding="utf-8"))["splits"]["test"]["model"]
            if abs(auc - reference["roc_auc"]) > 1e-3:
                print(f"[сеть {tag}] ROC-AUC {auc:.4f} не совпадает с отчётом "
                      f"{reference['roc_auc']:.4f} — пропускаю")
                continue
            out[f"сеть {tag}"] = {
                "анализов": int(len(yt)), "превышений": int(over.sum()), "недель": n_weeks,
                "roc_auc": {"значение": round(auc, 3), "90% интервал": interval(
                    block_bootstrap(codes, n_weeks,
                                    lambda i: float(roc_auc_score(over[i], r[i])),
                                    f"сеть {tag}/roc_auc"))}}
    except ImportError:
        print("[сеть] torch не установлен — интервалы сети не считаются")

    # ---- уровень решений: рабочая точка прогона по тестовому периоду ----
    from scripts.run_test_period import MAIN_WINDOW
    from scripts.compare_decision_curves import decision_flags
    tp = json.loads((ROOT / "reports" / "test_period.json").read_text(encoding="utf-8"))
    act = tp["summary"]["порог вмешательства"]["рабочий"]
    working = next(r for r in tp["summary"]["порог вмешательства"]["перебор"] if r["рабочий"])
    flags = decision_flags(pd.DataFrame(tp["rows"]), act, MAIN_WINDOW, act).set_index("ts")
    over = flags["over"].to_numpy()
    missed = flags["missed"].to_numpy()
    fa = flags["fa"].to_numpy()
    codes, n_weeks = week_index(pd.DatetimeIndex(flags.index))
    miss_rate = lambda i: float(missed[i].sum() / over[i].sum()) if over[i].any() else np.nan  # noqa: E731
    fa_rate = lambda i: float(fa[i].sum() / (~over[i]).sum())  # noqa: E731
    if round(miss_rate(np.arange(len(over))), 3) != working["доля пропусков"]:
        raise SystemExit("ОШИБКА: доля пропусков не воспроизводит прогон по тесту")
    out["решения, рабочая точка бустинга"] = {
        "моментов": int(len(over)), "превышений": int(over.sum()), "недель": n_weeks,
        "доля пропусков": {"значение": working["доля пропусков"],
                           "90% интервал": interval(block_bootstrap(codes, n_weeks, miss_rate, "решения/пропуски"))},
        "доля ложных тревог": {"значение": working["доля ложных тревог"],
                               "90% интервал": interval(block_bootstrap(codes, n_weeks, fa_rate, "решения/ложные"))},
    }

    print("\nГлавные числа с 90 % интервалами (блочный бутстрэп по неделям)\n")
    for name, block in out.items():
        parts = [f"{k} {v['значение']:.3f} [{v['90% интервал'][0]:.3f}…{v['90% интервал'][1]:.3f}]"
                 for k, v in block.items() if isinstance(v, dict) and "значение" in v]
        size = block.get("анализов", block.get("моментов"))
        print(f"  {name:34s} n={size}, превышений {block['превышений']}, недель {block['недель']}")
        for part in parts:
            print(f"      {part}")
        if "в отчёте при округлённом пороге" in block:
            e = block["в отчёте при округлённом пороге"]
            print(f"      (в отчёте при округлённом пороге {e['порог']}: recall "
                  f"{e['recall']:.3f}, precision {e['precision']:.3f})")

    REPORT.write_text(json.dumps({**report_provenance(cfg), "бутстрэп": {
        "блок": "неделя", "выборок": BOOTSTRAP, "уровень": 0.9}, "числа": out},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
