#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$HERE/log_full_direct_pimeancov_lambda0p01_bs2048_seed1337"
PID_FILE="$OUT_DIR/train.pid"
CONSOLE_LOG="$OUT_DIR/console.log"

mkdir -p "$OUT_DIR"
if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "direct pi-mean-cov decoder is already running with PID $(<"$PID_FILE")"
  exit 0
fi

: >"$PID_FILE"
setsid --fork bash -c '
  pid_file="$1"
  console_log="$2"
  shift 2
  printf "%s\n" "$$" >"$pid_file"
  exec "$@" >>"$console_log" 2>&1
' _ "$PID_FILE" "$CONSOLE_LOG" bash "$HERE/run_gpu4.sh"

for _ in {1..20}; do
  [[ -s "$PID_FILE" ]] && break
  sleep 0.1
done
echo "launched direct pi-mean-cov decoder on physical GPU4: PID $(<"$PID_FILE")"
