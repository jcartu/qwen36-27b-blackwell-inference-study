#!/usr/bin/env bash
# Exp 11 v2 orchestrator: runs all 6 cells using wikitext token-source.
#
# Differences from v1:
#   - Uses pre-tokenized wikitext-tokens-{single,multi}.safetensors as token-source
#     (avoids the silent fallback to "the quick brown fox" periodic prompt)
#   - NVFP4 cells use --enforce-eager (avoids inductor stride bug)
#   - Single-pass: all cells in one orchestrator
#
# Order:
#   bf16-ref-single, bf16-ref-multi    (primary references using wikitext)
#   bf16-self-single, bf16-self-multi  (noise-floor baselines, same wikitext seed)
#   fp8-single, fp8-multi              (FP8 deltas)
#   nvfp4-single, nvfp4-multi          (NVFP4 deltas, eager mode)

set -euo pipefail

SWEEP=/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld
LAUNCHER=$SWEEP/scripts/launch_collect.sh
WT_SINGLE=$SWEEP/refs/wikitext-tokens-single.safetensors
WT_MULTI=$SWEEP/refs/wikitext-tokens-multi.safetensors

if [[ ! -f "$WT_SINGLE" || ! -f "$WT_MULTI" ]]; then
  echo "ERROR: token sources missing. Run scripts/prep_wikitext_tokens.py first." >&2
  exit 1
fi

cell() {
  # cell <label> <model> <mode> <quant> <output> <token_source> [extra_args]
  local label="$1" model="$2" mode="$3" quant="$4" output="$5" token_source="$6" extra="${7:-}"
  if [[ -f "$output" ]]; then
    echo "[$label] EXISTS at $output - SKIP"
    return 0
  fi
  echo ""
  echo "============================================================"
  echo "[$label] $(date -Iseconds)"
  echo "  model=$model  mode=$mode  quant=$quant  extra=$extra"
  echo "============================================================"
  env LABEL="$label" MODEL="$model" \
      OUTPUT="$output" \
      MODE="$mode" QUANT="$quant" \
      TOKEN_SOURCE="$token_source" \
      EXTRA_ARGS="$extra" \
      bash "$LAUNCHER"
  echo "[$label] done $(date -Iseconds)"
}

# Phase 1: BF16 reference (single + multi) — uses wikitext token source
cell bf16-ref-single Qwen/Qwen3.6-27B single none \
  "$SWEEP/refs/bf16-ref-single.safetensors" "$WT_SINGLE"

cell bf16-ref-multi Qwen/Qwen3.6-27B multi none \
  "$SWEEP/refs/bf16-ref-multi.safetensors" "$WT_MULTI"

# Phase 5: BF16-self (noise floor) — same wikitext seed, second run
cell bf16-self-single Qwen/Qwen3.6-27B single none \
  "$SWEEP/refs/bf16-self-single.safetensors" "$WT_SINGLE"

cell bf16-self-multi Qwen/Qwen3.6-27B multi none \
  "$SWEEP/refs/bf16-self-multi.safetensors" "$WT_MULTI"

# Phase 2a: FP8
cell fp8-single Qwen/Qwen3.6-27B-FP8 single fp8 \
  "$SWEEP/results/fp8-single.safetensors" "$WT_SINGLE"

cell fp8-multi Qwen/Qwen3.6-27B-FP8 multi fp8 \
  "$SWEEP/results/fp8-multi.safetensors" "$WT_MULTI"

# Phase 2c: NVFP4 — needs --enforce-eager (inductor stride bug)
cell nvfp4-single sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP single modelopt_fp4 \
  "$SWEEP/results/nvfp4-single.safetensors" "$WT_SINGLE" --enforce-eager

cell nvfp4-multi sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP multi modelopt_fp4 \
  "$SWEEP/results/nvfp4-multi.safetensors" "$WT_MULTI" --enforce-eager

echo ""
echo "=== ALL DONE $(date -Iseconds) ==="
ls -la $SWEEP/refs/ $SWEEP/results/
