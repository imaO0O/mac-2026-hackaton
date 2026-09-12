#!/usr/bin/env bash
# Досчитать то, что осталось после остановки фоновых счётов.
# Каждая команда самодостаточна: порядок не важен, повторный запуск безвреден.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run () { echo; echo "########## $* ##########"; "$@" 2>&1 | tail -10; }

# устойчивость отбора признаков: h0 уже посчитан
run $PY scripts/check_feature_stability.py --horizon 1 --seeds 5
run $PY scripts/check_feature_stability.py --horizon 2 --seeds 5

# три сети с устаревшими именами каналов
run $PY scripts/train_sequence.py --horizon 2 --arch tcn --window 48
run $PY scripts/train_sequence.py --horizon 2 --arch tcn --window 48 --pretrain --seed-base 100
run $PY scripts/train_sequence.py --horizon 2 --arch tcn --window 48 --pretrain --seed-base 200

# прогон, упавший из-за устаревшей модели
run $PY scripts/run_test_period.py --model seq --seq-horizon 2

echo
echo "########## ДОСЧИТАНО $(date +%H:%M) ##########"
echo "Проверить: .venv/Scripts/python.exe -m pytest tests/test_report_freshness.py -q"
