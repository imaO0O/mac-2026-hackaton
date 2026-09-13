#!/usr/bin/env bash
# Устойчивость бустинга по сидам на уровне РЕШЕНИЙ — то же, что сделано для сети.
# Сидовые модели и отчёты получают суффикс _s<сид> и рабочую модель не трогают.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run () { echo; echo "########## $(date +%H:%M) $* ##########"; "$@" 2>&1 | grep -v Warning | tail -14; }

run $PY scripts/train_quality.py --horizon 0 --seed 100
run $PY scripts/train_quality.py --horizon 0 --seed 200
run $PY scripts/run_test_period.py
run $PY scripts/run_test_period.py --quality-path models/sulfur_h0_s100
run $PY scripts/run_test_period.py --quality-path models/sulfur_h0_s200
# детерминированность: сид 42 под отдельным именем, рабочая модель не трогается
run $PY scripts/train_quality.py --horizon 0 --tag _repro

echo
echo "########## СИДЫ БУСТИНГА ГОТОВЫ $(date +%H:%M) ##########"
