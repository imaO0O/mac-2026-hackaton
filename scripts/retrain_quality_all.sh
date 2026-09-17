#!/usr/bin/env bash
# Переобучить все модели качества, чьи отчёты лежат в reports/, тем же кодом.
# Нужно, когда меняется то, что пишется в отчёт, а не сама модель: обучение
# детерминировано, модели получаются те же, а отчёты — в формате текущего кода.
# Две дорожки параллельно, чтобы уложиться примерно в час.
#
# Повторный запуск безвреден и продолжает с места остановки: отчёт, снятый на
# текущей версии матрицы и в текущем формате (оба MARKER), пропускается. Так
# остановка на середине — закрытое приложение, выключенная машина — теряет только
# идущие обучения. Раньше проверялся только формат: после смены версии матрицы
# скрипт пропустил бы все модели как «уже готовые».
#
#   bash scripts/retrain_quality_all.sh              # логи в /tmp
#   LOG_DIR=путь bash scripts/retrain_quality_all.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG_DIR=${LOG_DIR:-/tmp}
MARKER='"risk_source_margin"'
VERSION=$($PY -c 'import sys; sys.path.insert(0, "src"); from nefte.models.dataset import FEATURE_VERSION; print(FEATURE_VERSION)')
VERSION_MARKER="\"feature_version\": $VERSION,"
run () {
  local report="reports/$1"; shift
  if [ -f "$report" ] && grep -q "$MARKER" "$report" && grep -q "$VERSION_MARKER" "$report"; then
    echo "########## $(date +%H:%M) пропуск: $report уже в текущем формате ##########"
    return
  fi
  echo "########## $(date +%H:%M) $* ##########"
  "$@" 2>&1 | grep -v Warning | tail -3
}
lane_a () {
  run quality_metrics_h0.json      $PY scripts/train_quality.py --horizon 0
  run quality_metrics_h1.json      $PY scripts/train_quality.py --horizon 1
  run quality_metrics_h2.json      $PY scripts/train_quality.py --horizon 2
  run quality_metrics_h3.json      $PY scripts/train_quality.py --horizon 3
  run quality_metrics_t95_h0.json  $PY scripts/train_quality.py --horizon 0 --target t95
}
lane_b () {
  run quality_metrics_h0_s100.json  $PY scripts/train_quality.py --horizon 0 --seed 100
  run quality_metrics_h0_s200.json  $PY scripts/train_quality.py --horizon 0 --seed 200
  run quality_metrics_h0_novak.json $PY scripts/train_quality.py --horizon 0 --no-vak
  run quality_metrics_h2_novak.json $PY scripts/train_quality.py --horizon 2 --no-vak
  run quality_metrics_t95_h2.json   $PY scripts/train_quality.py --horizon 2 --target t95
}
lane_a > "$LOG_DIR/retrain_a.log" 2>&1 &
lane_b > "$LOG_DIR/retrain_b.log" 2>&1 &
wait
cat "$LOG_DIR/retrain_a.log" "$LOG_DIR/retrain_b.log"
echo "########## ВСЕ МОДЕЛИ КАЧЕСТВА ПЕРЕОБУЧЕНЫ $(date +%H:%M) ##########"
