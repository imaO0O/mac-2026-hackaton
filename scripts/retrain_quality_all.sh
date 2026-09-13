#!/usr/bin/env bash
# Переобучить все модели качества, чьи отчёты лежат в reports/, тем же кодом.
# Нужно, когда меняется то, что пишется в отчёт, а не сама модель: обучение
# детерминировано, модели получаются те же, а отчёты — в формате текущего кода.
# Две дорожки параллельно, чтобы уложиться примерно в час.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
run () { echo "########## $(date +%H:%M) $* ##########"; "$@" 2>&1 | grep -v Warning | tail -3; }
lane_a () {
  run $PY scripts/train_quality.py --horizon 0
  run $PY scripts/train_quality.py --horizon 1
  run $PY scripts/train_quality.py --horizon 2
  run $PY scripts/train_quality.py --horizon 3
  run $PY scripts/train_quality.py --horizon 0 --target t95
}
lane_b () {
  run $PY scripts/train_quality.py --horizon 0 --seed 100
  run $PY scripts/train_quality.py --horizon 0 --seed 200
  run $PY scripts/train_quality.py --horizon 0 --no-vak
  run $PY scripts/train_quality.py --horizon 2 --no-vak
  run $PY scripts/train_quality.py --horizon 2 --target t95
}
lane_a > /tmp/retrain_a.log 2>&1 &
lane_b > /tmp/retrain_b.log 2>&1 &
wait
cat /tmp/retrain_a.log /tmp/retrain_b.log
echo "########## ВСЕ МОДЕЛИ КАЧЕСТВА ПЕРЕОБУЧЕНЫ $(date +%H:%M) ##########"
