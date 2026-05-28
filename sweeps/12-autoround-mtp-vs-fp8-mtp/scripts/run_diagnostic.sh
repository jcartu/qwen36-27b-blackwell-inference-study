#!/usr/bin/env bash
# Exp 12 diagnostic: 2 cells of llm_decode_bench.py on AutoRound-int4 + MTP=3.
#
# Methodology matches Exp 8 X1 (the FP8+MTP=3 SOTA at c=16/c=32) exactly:
#   --duration 60 --decode-warmup-seconds 20 --skip-prefill --display-mode plain
#
# Cells:
#   Cell A: c=32 x ctx=0    -- compare to FP8+MTP=3 SOTA 2,083.7 tok/s peak
#   Cell B: c=16 x ctx=131k -- compare to FP8+MTP=3 SOTA 1,047.1 tok/s
#                              (memory-savings sweet spot if AutoRound's 38% lighter weight footprint
#                               can fit higher KV cache density)
#
# Each cell run N=2 reps so we have a mean and a sanity-check spread, then aggregate.

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
EXP_ROOT=$(cd "$HERE/.." && pwd)
HARNESS=/home/josh/qwen-vllm-test/llm-inference-bench/llm_decode_bench.py
PORT=8765
OUT="$EXP_ROOT/results"
LOGS="$EXP_ROOT/logs"
mkdir -p "$OUT" "$LOGS"

# Sanity: server up
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: server not healthy on :${PORT}; aborting."
  exit 1
fi

run_cell () {
  local cell="$1" conc="$2" ctx="$3" rep="$4"
  local stem="$OUT/${cell}-rep${rep}"
  if [ -s "${stem}.json" ] && grep -q '"aggregate' "${stem}.json" 2>/dev/null; then
    echo "[SKIP] $cell rep$rep (already populated)"
    return 0
  fi
  echo "[RUN ] $cell rep$rep  conc=$conc ctx=$ctx"
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
    echo "[DONE] $cell rep$rep  ${elapsed}s"
  else
    echo "[FAIL] $cell rep$rep rc=$rc — see ${stem}.log"
  fi
}

# Cell A: throughput-peak cell
run_cell "cellA-c32-ctx0"    32 "0"   1
run_cell "cellA-c32-ctx0"    32 "0"   2

# Cell B: long-context memory-stress cell
run_cell "cellB-c16-ctx131k" 16 "131k" 1
run_cell "cellB-c16-ctx131k" 16 "131k" 2

echo "=== All cells done ==="
ls -la "$OUT"/cell*.json 2>/dev/null
