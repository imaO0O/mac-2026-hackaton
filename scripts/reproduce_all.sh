#!/usr/bin/env bash
# Пересобрать ВСЕ отчёты reports/*.json из выданных данных одной командой.
#
#   bash scripts/reproduce_all.sh              # всё, что можно на этой машине
#   SKIP_SLOW=1 bash scripts/reproduce_all.sh  # без проверок по сидам (часы счёта)
#
# Порядок — по зависимостям: данные → модели → разбор данных → разбор моделей →
# прогоны системы → сравнения, которые читают отчёты прогонов. Нейросети
# пересобираются, только если установлен torch (docs/GPU_SETUP.md); без него их
# отчёты остаются закоммиченными и на остальное не влияют.
#
# Обучение детерминировано: на тех же данных отчёты совпадают с закоммиченными.
# После пересборки `pytest -q` сверяет с ними документацию.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=python
LOG_DIR=${LOG_DIR:-/tmp}
FAILED=()
run () {
  echo "########## $(date +%H:%M) $* ##########"
  if ! "$@" > "$LOG_DIR/reproduce_step.log" 2>&1; then
    FAILED+=("$*"); tail -5 "$LOG_DIR/reproduce_step.log"
  fi
}
slow () { [ -n "${SKIP_SLOW:-}" ] && { echo "пропуск (SKIP_SLOW): $*"; return; }; run "$@"; }

# 1. данные
run $PY scripts/prepare_data.py

# 2. модели качества: 10 моделей в две дорожки
run env LOG_DIR="$LOG_DIR" bash scripts/retrain_quality_all.sh

# 3. нейросети и автоэнкодер — только при установленном torch
if $PY -c "import torch" 2>/dev/null; then
  for spec in "gru 24" "gru 48" "tcn 24" "tcn 48"; do
    set -- $spec
    run $PY scripts/train_sequence.py --horizon 0 --arch "$1" --window "$2"
  done
  run $PY scripts/train_sequence.py --horizon 2 --arch tcn --window 48
  run $PY scripts/train_sequence.py --horizon 0 --arch tcn --window 48 --pretrain
  for base in 42 100 200; do
    run $PY scripts/train_sequence.py --horizon 2 --arch tcn --window 48 --pretrain --seed-base "$base"
  done
  run $PY scripts/train_anomaly_ae.py
else
  echo "torch не установлен: отчёты нейросетей не пересобираются"
fi

# 4. разбор данных
run $PY scripts/backtest_reliability.py
run $PY scripts/check_tag_meaning.py
run $PY scripts/check_avt_formulas.py
run $PY scripts/check_avt_schemes.py
run $PY scripts/check_vak_against_lims.py
run $PY scripts/check_pipeline_point.py
run $PY scripts/find_delays.py
run $PY scripts/check_cetane.py
run $PY scripts/check_catalyst_life.py
run $PY scripts/check_f65_units.py
run $PY scripts/find_regime_episodes.py
run $PY scripts/check_avt_to_ht_lag.py
run $PY scripts/check_t95_sigma.py

# 5. разбор моделей качества
run $PY scripts/check_calibration.py --horizon 0
run $PY scripts/check_drift.py --horizon 0
run $PY scripts/check_sensitivity.py --horizon 0
run $PY scripts/check_t95_calibration.py
run $PY scripts/check_upper_edge_risk.py
run $PY scripts/check_alarm_budget.py --horizon 0
if $PY -c "import torch" 2>/dev/null; then
  run $PY scripts/check_alarm_budget.py --horizon 2 --model seq --arch tcn --window 48 --pretrain
fi
for h in 0 1 2; do slow $PY scripts/check_feature_stability.py --horizon "$h" --seeds 5; done

# 6. прогоны системы
run $PY scripts/run_test_period.py
run $PY scripts/run_test_period.py --every 1h --tag step1h
for seed in 100 200; do
  run $PY scripts/run_test_period.py --quality-path "models/sulfur_h0_s$seed"
done
if $PY -c "import torch" 2>/dev/null; then
  run $PY scripts/run_test_period.py --model seq --seq-horizon 0
  run $PY scripts/run_test_period.py --model seq --seq-horizon 2
  for base in 100 200; do
    run $PY scripts/run_test_period.py --model seq --seq-horizon 2 \
      --seq-path "models/sulfur_seq_tcn48_pre_s${base}_h2"
  done
fi
run $PY scripts/check_adversarial.py
run $PY scripts/compare_architectures.py
run $PY scripts/run_simulation.py
run $PY scripts/check_return_to_base.py
run $PY scripts/check_kinetic_order.py
run $PY scripts/run_blend_scenarios.py
slow $PY scripts/check_objective_weights.py
slow $PY scripts/check_severity_robustness.py
slow $PY scripts/check_severity_robustness.py --spread 0.2

# 7. сравнения, читающие отчёты прогонов
run $PY scripts/check_event_response.py
run $PY scripts/check_offspec_followup.py
run $PY scripts/compare_decision_curves.py
run $PY scripts/check_headline_intervals.py

echo "########## $(date +%H:%M) готово ##########"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "не удалось:"; printf '  %s\n' "${FAILED[@]}"; exit 1
fi
