"""Разбор прогноза по группам каналов: стоит ли показывать его оператору. Только CPU.

    python scripts/check_risk_attribution.py

Зачем. Карточка сейчас говорит, ЧТО делать и с каким запасом, но не говорит,
почему прогноз именно такой. Вклады считаются точно (SHAP для CatBoost — не
приближение, а разложение конкретного прогноза: сумма вкладов плюс база в
точности равна прогнозу), стоят доли миллисекунды на решение и ничего в решении
не меняют.

Но показывать разбор имеет смысл, только если он несёт информацию. Если в девяти
случаях из десяти «виновата» одна и та же группа, строка в карточке станет фоном,
который оператор перестанет читать.

**Правило записано ДО счёта.** Разбор уходит в карточку, если на ТЕСТЕ:

1. ведущая группа меняется: доля самой частой лидирующей группы не выше 80 %;
2. разложение сходится: |база + сумма вкладов − прогноз| < 0.01 мг/кг на всех
   строках (иначе разбор говорит не о том прогнозе, который показан).

Не выполняется — измеренный отказ, в карточку не идёт.

Вариантов разбора два, правило для обоих записано до счёта:

* **А. все группы** — как есть;
* **Б. без группы измерений серы** — оператор и так видит показание прибора в
  первой строке карточки, и «прогноз высокий, потому что сера высокая» ему ничего
  не добавляет; ведущей считается группа среди тех, на которые можно повлиять
  уставками или которые приходят с АВТ.

Третий вариант (**В**) следует из тех же чисел и отдельного счёта не требует:
показывать строку не всегда, а только когда причина НЕОБЫЧНА — ведущая группа не
та, что ведёт обычно. Его правило тоже записано до подстановки чисел: такая
строка полезна, если появляется не реже чем в 5 % решений (иначе её никто не
увидит) и не чаще чем в половине (иначе она не редкая).

Группы — не произвольные: это те же каналы, по которым у модели зашита физика
(`PHYSICS_MONOTONE`), плюс «сырьё с АВТ» — связка, ради которой в системе есть
отдельный агент АВТ, и «свежесть данных» — канал, из-за которого система умеет
отказываться.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT, load_config  # noqa: E402
from nefte.data.features import time_split  # noqa: E402
from nefte.models.dataset import build_feature_matrix  # noqa: E402
from nefte.models.quality_model import SulfurModel  # noqa: E402
from nefte.provenance import report_provenance  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

REPORT = ROOT / "reports" / "risk_attribution.json"
MAX_SHARE = 0.80
TOLERANCE = 0.01
MEASURED = "показания по сере"
USUAL = "сырьё с АВТ"
# Пороги по величине вклада: 0 — без порога, 0.3 — эффект одного градуса уставки
# (нижняя граница измеренного отклика, docs/SULFUR_RESPONSE.md).
FLOORS = (0.0, 0.1, 0.3, 0.5)

GROUPS: list[tuple[str, tuple[str, ...]]] = [
    (MEASURED, ("ht_Q21", "pak_sulfur", "lims_sulfur_prev", "ht_Q20")),
    (USUAL, ("avt_", "vak_", "lims_feed_sulfur")),
    ("режим реактора", ("reg_wabt", "reg_kinetic", "reg_drive", "reg_dt_react",
                        "ht_T5", "ht_T11", "ht_T16", "ht_T18")),
    ("водород", ("reg_h2", "reg_makeup", "ht_P13", "ht_F25")),
    ("нагрузка и квенч", ("ht_F26", "ht_F22", "ht_F14", "reg_quench")),
    ("свежесть данных", ("feature_age_h",)),
]


def group_of(name: str) -> str:
    for title, prefixes in GROUPS:
        if any(name.startswith(p) for p in prefixes):
            return title
    return "прочая телеметрия"


def main() -> int:
    use_utf8_console()
    cfg = load_config()
    model = SulfurModel.load(SulfurModel.default_path(0.0, "sulfur"))
    features = build_feature_matrix()
    masks = time_split(features.index, cfg)

    out: dict = {}
    for split in ("val", "test"):
        X = features[masks[split].to_numpy()][model.features].dropna()
        if X.empty:
            continue
        from catboost import Pool
        start = time.time()
        shap = model.models["q50"].get_feature_importance(Pool(X), type="ShapValues")
        seconds = (time.time() - start) / len(X)

        contrib = pd.DataFrame(shap[:, :-1], columns=model.features, index=X.index)
        base = float(shap[0, -1]) + model.y_offset
        grouped = contrib.T.groupby(group_of).sum().T
        pred = model.predict_frame(X)["q50"]
        error = float(np.abs(base + grouped.sum(axis=1) - pred).max())

        share = grouped.abs().idxmax(axis=1).value_counts(normalize=True).round(3)
        rest = grouped.drop(columns=[MEASURED], errors="ignore")
        lead_b = rest.abs().idxmax(axis=1)
        share_b = lead_b.value_counts(normalize=True).round(3)
        # Порог по ВЕЛИЧИНЕ вклада: показывать строку, только если канал тянет
        # прогноз не меньше, чем даёт градус уставки (0.3 мг/кг — нижняя граница
        # измеренного отклика). Проверяем, что от строки после этого останется.
        size = rest.abs().max(axis=1)
        floors = {str(f): round(float(((lead_b != USUAL) & (size >= f)).mean()), 3)
                  for f in FLOORS}
        weight = grouped.abs().mean().sort_values(ascending=False).round(3)
        out[split] = {
            "строк": int(len(X)),
            "база модели, мг/кг": round(base, 2),
            "секунд на решение": round(seconds, 5),
            "невязка разложения, мг/кг": round(error, 4),
            "как часто группа ведущая": share.to_dict(),
            "как часто группа ведущая без измерений серы": share_b.to_dict(),
            "доля решений со строкой разбора при пороге вклада": floors,
            "средний вклад по модулю, мг/кг": weight.to_dict(),
        }
        print(f"\n{split}: строк {len(X)}, база {base:.2f} мг/кг, "
              f"невязка {error:.4f}, {seconds * 1000:.2f} мс на решение")
        for name, value in share.items():
            print(f"  А: ведущая {name:22s} {value:6.1%}   "
                  f"средний вклад {weight.get(name, 0):+.2f}")
        for name, value in share_b.items():
            print(f"  Б: ведущая {name:22s} {value:6.1%}")
        print("  В: доля решений со строкой разбора при пороге вклада: "
              + ", ".join(f"{f} -> {v:.1%}" for f, v in floors.items()))

    test = out.get("test", out.get("val", {}))
    converges = bool(test.get("невязка разложения, мг/кг", 1.0) < TOLERANCE)
    rules: dict = {}
    accepted: dict = {}
    for tag, key in (("А. все группы", "как часто группа ведущая"),
                     ("Б. без измерений серы",
                      "как часто группа ведущая без измерений серы")):
        block = test.get(key) or {}
        top = max(block.values()) if block else 1.0
        rules[tag] = {
            "ведущая чаще всего": (max(block, key=block.get) if block else None),
            "доля самой частой": round(float(top), 3),
            "1. ведущая группа меняется (не больше 80 % на одну)": bool(top <= MAX_SHARE),
            "2. разложение сходится к прогнозу": converges,
        }
        accepted[tag] = bool(top <= MAX_SHARE) and converges

    habit = rules["Б. без измерений серы"]["ведущая чаще всего"]
    block_b = test.get("как часто группа ведущая без измерений серы") or {}
    unusual = round(1.0 - float(block_b.get(habit, 0.0)), 3)
    rules["В. только необычная причина"] = {
        "обычная ведущая": habit,
        "доля решений со строкой разбора": unusual,
        "1. не реже 5 % решений": bool(unusual >= 0.05),
        "2. не чаще половины решений": bool(unusual <= 0.50),
        "3. разложение сходится к прогнозу": converges,
    }
    accepted["В. только необычная причина"] = bool(0.05 <= unusual <= 0.50) and converges

    print("\nПравило приёмки (тест):")
    for tag, checks in rules.items():
        print(f"  {tag}: {'ПРИНЯТ' if accepted[tag] else 'НЕ принят'}")
        for name, value in checks.items():
            print(f"      {name}: {value}")
    winners = [tag for tag, ok in accepted.items() if ok]
    print("  ВЫВОД: разбор причины в карточке "
          + (f"ПРИНЯТ в варианте «{winners[0]}»" if winners else "НЕ принят"))

    REPORT.write_text(json.dumps({**report_provenance(cfg),
                                  "условие: доля ведущей группы не выше": MAX_SHARE,
                                  "условие: невязка меньше": TOLERANCE,
                                  "группы": {name: list(prefixes)
                                             for name, prefixes in GROUPS},
                                  "выборки": out, "правило": rules,
                                  "принят по вариантам": accepted,
                                  "принят": bool(winners),
                                  "вариант": winners[0] if winners else None},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
