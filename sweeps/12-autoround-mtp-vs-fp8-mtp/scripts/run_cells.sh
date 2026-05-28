#!/usr/bin/env bash
# Run the 2 diagnostic cells × 2 reps for whichever variant is currently serving.
# Usage:  run_cells.sh <variant-label>
# Example: run_cells.sh fp8     OR   run_cells.sh autoround

set -uo pipefail

VARIANT="${1:?usage: run_cells.sh <variant-label>}"
HERE=$(cd "$(dirname "$0")" && pwd)
EXP_ROOT=$(cd "$HERE/.." && pwd)
HARNESS=/home/josh/qwen-vllm-test/llm-inference-bench/llm_decode_bench.py
PORT=8765
OUT="$EXP_ROOT/results/${VARIANT}"
mkdir -p "$OUT"

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: server not healthy on :${PORT}; aborting."
  exit 1
fi

run_cell () {
  local cell="$1" conc="$2" ctx="$3" rep="$4"
  local stem="$OUT/${cell}-rep${rep}"
  if [ -s "${stem}.json" ] && grep -q '"aggregate\|"per_request' "${stem}.json" 2>/dev/null; then
    echo "[SKIP] $cell rep$rep"
    return 0
  fi
  echo "[RUN ] ${VARIANT} $cell rep$rep  conc=$conc ctx=$ctx  @$(date '+%H:%M:%S')"
  local start=$(date +%s)
  echo n | python3 "$HARNESS" \
    --port "$PORT" \
    --model Qwen3.6-27B \
    --concurrency "$conc" \
    --contexts "$ctx" \
    --duration 60 \
    --decode-warmup-seconds 20 \
    --skip-prefill \
    --display-mode plain \
    --no-hw-monitor \
    --output "${stem}.json" \
    > "${stem}.log" 2>&1
  local rc=$?
  local elapsed=$(($(date +%s) - start))
  if [ "$rc" -eq 0 ] && [ -s "${stem}.json" ]; then
    echo "[DONE] ${VARIANT} $cell rep$rep  ${elapsed}s"
  else
    echo "[FAIL] ${VARIANT} $cell rep$rep rc=$rc — see ${stem}.log"
    tail -5 "${stem}.log"
  fi
}

run_cell "cellA-c32-ctx0"    32  "0"     1
run_cell "cellA-c32-ctx0"    32  "0"     2
run_cell "cellB-c16-ctx131k" 16  "131k"  1
run_cell "cellB-c16-ctx131k" 16  "131k"  2

echo ""
echo "=== Results for ${VARIANT} ==="
ls -la "$OUT"
