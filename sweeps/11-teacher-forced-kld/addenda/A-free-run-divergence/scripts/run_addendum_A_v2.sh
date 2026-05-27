#!/usr/bin/env bash
# Addendum A v2 orchestrator: extra no-spec quant cells (Lavd's request).
# Runs 5 new cells sequentially on GPUs 1+2 (GPU 0 reserved).
#
# Cells (all no-spec; all greedy temp=0 seed=42 max_tokens=1024 ignore_eos):
#   C4  AWQ-4bit       (QuantTrio/Qwen3.6-27B-AWQ,                quant=awq_marlin)
#   C5  AWQ-6Bit       (QuantTrio/Qwen3.6-27B-AWQ-6Bit,           quant=awq_marlin)
#   C6  AutoRound-int4 (Intel/Qwen3.6-27B-int4-AutoRound,         quant=auto-round)
#   C7  GPTQ-groxaxo   (groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit,        quant=gptq_marlin) — canonical-base
#   C8  GPTQ-Qwopus    (XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-v1,  quant=gptq_marlin) — fine-tune base (see README)
#
# Reuses launch_free_run.sh. Each cell: ~85-130 s based on smoke timings
# (cache hot → load ~30 s; gen 8×1024 @ ~25 tok/s → ~50-90 s; total ~80-120 s).

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
ADD_DIR="${SWEEP_DIR}/addenda/A-free-run-divergence"
RES="${ADD_DIR}/results"
SCRIPTS="${ADD_DIR}/scripts"

mkdir -p "$RES" "${ADD_DIR}/logs"
chmod +x "${SCRIPTS}/launch_free_run.sh"

run_cell() {
  local label="$1" model="$2" quant="$3"
  echo ""
  echo "############################################################"
  echo "  CELL: $label  ($model, quant=$quant)"
  echo "############################################################"
  LABEL="$label" \
  MODEL="$model" \
  OUTPUT="${RES}/${label}.json" \
  QUANT="$quant" \
  SPEC_CONFIG="" \
  ENFORCE_EAGER="0" \
  MAX_TOKENS=1024 \
  bash "${SCRIPTS}/launch_free_run.sh"
}

echo "=== Addendum A v2: extra no-spec quant cells ==="
echo "  5 cells × 8 prompts × 1024 tokens"
echo "  GPUs 1+2 (TP=2), repne/vllm:v13"
echo "  Start: $(date)"
echo ""

run_cell "C4-awq-4bit"       "QuantTrio/Qwen3.6-27B-AWQ"               "awq_marlin"
run_cell "C5-awq-6bit"       "QuantTrio/Qwen3.6-27B-AWQ-6Bit"          "awq_marlin"
run_cell "C6-autoround-int4" "Intel/Qwen3.6-27B-int4-AutoRound"        "auto-round"
run_cell "C7-gptq-groxaxo"   "groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit"       "gptq_marlin"
run_cell "C8-gptq-qwopus"    "XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-v1" "gptq_marlin"

echo ""
echo "=== All v2 cells complete: $(date) ==="
echo "=== Running comparisons vs C0a (BF16 reference) ==="

compare() {
  local ref="$1" cand="$2" out="$3"
  echo "  compare: $cand vs $ref → $out"
  python3 "${SCRIPTS}/compare_free_runs.py" \
    --ref "${RES}/${ref}.json" \
    --cand "${RES}/${cand}.json" \
    --output "${RES}/${out}.json"
}

compare "C0a-bf16-ref" "C4-awq-4bit"       "cmp-C4-awq4-vs-bf16"
compare "C0a-bf16-ref" "C5-awq-6bit"       "cmp-C5-awq6-vs-bf16"
compare "C0a-bf16-ref" "C6-autoround-int4" "cmp-C6-autoround-vs-bf16"
compare "C0a-bf16-ref" "C7-gptq-groxaxo"   "cmp-C7-gptq-groxaxo-vs-bf16"
compare "C0a-bf16-ref" "C8-gptq-qwopus"    "cmp-C8-gptq-qwopus-vs-bf16"

echo ""
echo "=== Addendum A v2 complete: $(date) ==="
ls -lh "${RES}"/C{4,5,6,7,8}*.json "${RES}"/cmp-C{4,5,6,7,8}*.json 2>/dev/null
