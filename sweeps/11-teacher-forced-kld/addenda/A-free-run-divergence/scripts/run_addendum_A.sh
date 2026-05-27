#!/usr/bin/env bash
# Addendum A orchestrator: runs 5 cells sequentially (GPU 0 reserved).
#
# Cells:
#   C0a  BF16 no-spec  (reference run 1)
#   C0b  BF16 no-spec  (reference run 2 — determinism check)
#   C1   FP8  no-spec  (matches Exp 11 FP8 config)
#   C2   FP8  MTP=3    (Josh's actual question)
#   C3   FP8  MTP=5    (stretch)
#
# Compare step at the end builds divergence.json for all pairs.

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
ADD_DIR="${SWEEP_DIR}/addenda/A-free-run-divergence"
RES="${ADD_DIR}/results"
SCRIPTS="${ADD_DIR}/scripts"

BF16_MODEL="Qwen/Qwen3.6-27B"
FP8_MODEL="Qwen/Qwen3.6-27B-FP8"

mkdir -p "$RES" "${ADD_DIR}/logs"
chmod +x "${SCRIPTS}/launch_free_run.sh"

run_cell() {
  local label="$1" model="$2" quant="$3" spec="${4:-}" eager="${5:-0}"
  echo ""
  echo "############################################################"
  echo "  CELL: $label"
  echo "############################################################"
  LABEL="$label" \
  MODEL="$model" \
  OUTPUT="${RES}/${label}.json" \
  QUANT="$quant" \
  SPEC_CONFIG="$spec" \
  ENFORCE_EAGER="$eager" \
  MAX_TOKENS=1024 \
  bash "${SCRIPTS}/launch_free_run.sh"
}

echo "=== Addendum A: Free-run divergence sweep ==="
echo "  5 cells × 8 prompts × 1024 tokens"
echo "  GPUs 1+2 (TP=2), repne/vllm:v13"
echo "  Start: $(date)"
echo ""

# C0a — BF16 reference, run 1
run_cell "C0a-bf16-ref" "$BF16_MODEL" "none" "" "0"

# C0b — BF16 reference, run 2 (determinism check)
run_cell "C0b-bf16-rerun" "$BF16_MODEL" "none" "" "0"

# C1 — FP8 no spec
run_cell "C1-fp8-no-spec" "$FP8_MODEL" "fp8" "" "0"

# C2 — FP8 + MTP=3
run_cell "C2-fp8-mtp3" "$FP8_MODEL" "fp8" \
  '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy"}' "0"

# C3 — FP8 + MTP=5
run_cell "C3-fp8-mtp5" "$FP8_MODEL" "fp8" \
  '{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"greedy"}' "0"

echo ""
echo "=== All cells complete: $(date) ==="
echo "=== Running comparisons ==="

compare() {
  local ref="$1" cand="$2" out="$3"
  echo "  compare: $cand vs $ref → $out"
  python3 "${SCRIPTS}/compare_free_runs.py" \
    --ref "${RES}/${ref}.json" \
    --cand "${RES}/${cand}.json" \
    --output "${RES}/${out}.json"
}

# Noise floor
compare "C0a-bf16-ref"   "C0b-bf16-rerun"  "cmp-C0b-vs-C0a-noise-floor"
# Main pairs
compare "C0a-bf16-ref"   "C1-fp8-no-spec"  "cmp-C1-fp8-vs-bf16"
compare "C0a-bf16-ref"   "C2-fp8-mtp3"     "cmp-C2-fp8-mtp3-vs-bf16"
compare "C0a-bf16-ref"   "C3-fp8-mtp5"     "cmp-C3-fp8-mtp5-vs-bf16"
# Spec vs no-spec (isolates MTP effect on top of FP8)
compare "C1-fp8-no-spec" "C2-fp8-mtp3"     "cmp-C2-mtp3-vs-fp8-no-spec"
compare "C1-fp8-no-spec" "C3-fp8-mtp5"     "cmp-C3-mtp5-vs-fp8-no-spec"

echo ""
echo "=== Addendum A complete: $(date) ==="
ls -lh "${RES}"/*.json 2>/dev/null
