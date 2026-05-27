#!/usr/bin/env bash
# run_cell.sh - Run one benchmark cell, idempotent (skips if output exists).
# Usage: run_cell.sh <harness_path> <output_dir> <cell_label> <concurrency> <contexts> [extra_args...]
set -euo pipefail

HARNESS="$1"
OUT_DIR="$2"
CELL="$3"
CONC="$4"
CTX="$5"
shift 5
EXTRA=("$@")

mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/${CELL}.json"
OUT_LOG="$OUT_DIR/${CELL}.log"

if [ -s "$OUT_JSON" ] && grep -q '"aggregate_tokens_per_second"\|"per_request_tokens_per_second"\|"aggregate_tps"' "$OUT_JSON" 2>/dev/null; then
  echo "[SKIP] $CELL — $OUT_JSON already populated"
  exit 0
fi

echo "[RUN ] $CELL  conc=$CONC ctx=$CTX  extra=${EXTRA[*]:-(none)}"
start=$(date +%s)

# v0.4.8 has an interactive upgrade prompt - pipe 'n' to decline
echo n | python3 "$HARNESS" \
  --port 8000 \
  --model Qwen3.6-27B \
  --concurrency "$CONC" \
  --contexts "$CTX" \
  --duration 60 \
  --decode-warmup-seconds 20 \
  --skip-prefill \
  --display-mode plain \
  --no-hw-monitor \
  --output "$OUT_JSON" \
  "${EXTRA[@]}" > "$OUT_LOG" 2>&1 || {
    rc=$?
    echo "[FAIL] $CELL (rc=$rc) — see $OUT_LOG" | tee -a "$OUT_DIR/_failures.log"
    exit $rc
  }

elapsed=$(($(date +%s) - start))
echo "[DONE] $CELL  ${elapsed}s"
