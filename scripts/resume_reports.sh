#!/usr/bin/env bash
# Досчитать то, что осталось после остановки фоновых счётов.
# Каждая команда самодостаточна, повторный запуск безвреден.
# Порядок: сначала быстрое и влияющее на выводы, медленное — в конце, чтобы
# остановка на середине теряла как можно меньше.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run () { echo; echo "########## $(date +%H:%M) $* ##########"; "$@" 2>&1 | tail -12; }

# 1. три сети с устаревшими именами каналов (~2 мин каждая на GPU)
run $PY scripts/train_sequence.py --horizon 2 --arch tcn --window 48
run $PY scripts/train_sequence.py --horizon 2 --arch tcn --window 48 --pretrain --seed-base 100
run $PY scripts/train_sequence.py --horizon 2 --arch tcn --window 48 --pretrain --seed-base 200

# 2. прогон, упавший из-за устаревшей модели: из него число «49 % против 71 %»
run $PY scripts/run_test_period.py --model seq --seq-horizon 2

# 3. устойчивость отбора признаков (~45 мин на горизонт): h0 уже посчитан
run $PY scripts/check_feature_stability.py --horizon 1 --seeds 5
run $PY scripts/check_feature_stability.py --horizon 2 --seeds 5

echo
echo "########## ДОСЧИТАНО $(date +%H:%M) ##########"
