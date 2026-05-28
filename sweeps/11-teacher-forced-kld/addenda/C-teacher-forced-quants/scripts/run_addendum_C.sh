#!/usr/bin/env bash
# Addendum C orchestrator: runs the 5 v2 quant cells (C4-C8) in teacher-forced
# mode (multi + single), then compares against bf16-ref-{multi,single} from
# Exp 11, yielding the data needed for a unified KL/JSD leaderboard.
#
# Sequential on GPUs 1+2 (TP=2), matching all prior Exp 11 + Addendum B runs.
# Expected wall-time: ~45 min (5 cells × (~6 min multi + ~2 min single)).

set -uo pipefail  # NOTE: no -e; individual cell failures must not kill the run

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
ADD_DIR="${SWEEP_DIR}/addenda/C-teacher-forced-quants"
LAUNCH="${ADD_DIR}/scripts/launch_collect_C.sh"
RESULTS="${ADD_DIR}/results"
COMPARE="${ADD_DIR}/compare"
LOGS="${ADD_DIR}/logs"

mkdir -p "$RESULTS" "$COMPARE" "$LOGS"

# Cell registry: LABEL_BASE | MODEL | QUANT
# Override via CELLS_FILTER env var (space-separated label prefixes); empty = all
ALL_CELLS=(
  "C4-awq-4bit|QuantTrio/Qwen3.6-27B-AWQ|awq_marlin"
  "C5-awq-6bit|QuantTrio/Qwen3.6-27B-AWQ-6Bit|awq_marlin"
  "C6-autoround-int4|Lorbus/Qwen3.6-27B-int4-AutoRound|auto-round"
  "C7-gptq-groxaxo|groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit|gptq_marlin"
  "C8-gptq-qwopus|XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-v1|gptq_marlin"
)

CELLS=()
if [[ -n "${CELLS_FILTER:-}" ]]; then
  for spec in "${ALL_CELLS[@]}"; do
    IFS='|' read -r base _ _ <<< "$spec"
    for keep in $CELLS_FILTER; do
      if [[ "$base" == "$keep"* ]]; then CELLS+=("$spec"); break; fi
    done
  done
  echo "# Filtered to: ${CELLS[*]}"
else
  CELLS=("${ALL_CELLS[@]}")
fi

run_cell() {
  local base="$1" model="$2" quant="$3" mode="$4"
  local label="${base}-${mode}"
  local output="${RESULTS}/${label}.safetensors"
  if [[ -f "$output" ]]; then
    echo "EXISTS $output (delete to recompute)"
    return 0
  fi
  echo ""
  echo "######################################################################"
  echo "# Running $label  ($(date '+%H:%M:%S'))"
  echo "######################################################################"
  LABEL="$label" MODEL="$model" QUANT="$quant" MODE="$mode" OUTPUT="$output" \
    bash "$LAUNCH" || {
      echo "!!! FAILED $label (continuing to next cell) !!!"
      return 1
    }
}

# === Phase 1: Collect ===
echo "==================================================================="
echo "Phase 1: teacher-forced collect (10 jobs = 5 cells × 2 modes)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================================="

for spec in "${CELLS[@]}"; do
  IFS='|' read -r base model quant <<< "$spec"
  run_cell "$base" "$model" "$quant" "multi"
  run_cell "$base" "$model" "$quant" "single"
done

echo ""
echo "==================================================================="
echo "Phase 2: pairwise compare vs bf16-ref-{multi,single}.safetensors"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================================="

# Comparison runs CPU-only inside the same image for numerical parity
DOCKER_RUN=(docker run --rm
  --volume "${SWEEP_DIR}":/sweep
  --volume "${SWEEP_DIR}/scripts/decode_logprob_kld.py":/workspace/decode_logprob_kld.py:ro
  --volume "${SWEEP_DIR}/scripts/decode_logprob_kld_multi.py":/workspace/decode_logprob_kld_multi.py:ro
  --env PYTHONPATH=/workspace
  --entrypoint /bin/bash
  repne/vllm:v13
  -lc
)

compare_pair() {
  local script="$1" a="$2" b="$3" out="$4"
  if [[ ! -f "$a" ]]; then echo "SKIP $out (missing A: $a)"; return 0; fi
  if [[ ! -f "$b" ]]; then echo "SKIP $out (missing B: $b)"; return 0; fi
  if [[ -f "$out" ]]; then echo "EXISTS $out (delete to recompute)"; return 0; fi
  local a_in="${a/$SWEEP_DIR/\/sweep}"
  local b_in="${b/$SWEEP_DIR/\/sweep}"
  local out_in="${out/$SWEEP_DIR/\/sweep}"
  echo "--- compare ---"
  echo "  script: $script"
  echo "  a:      $a"
  echo "  b:      $b"
  echo "  out:    $out"
  "${DOCKER_RUN[@]}" "cd /workspace && python3 $script compare --a $a_in --b $b_in --output $out_in"
  sudo chown -R josh:josh "$(dirname "$out")" 2>/dev/null || true
}

BF16_MULTI="${SWEEP_DIR}/refs/bf16-ref-multi.safetensors"
BF16_SINGLE="${SWEEP_DIR}/refs/bf16-ref-single.safetensors"

for spec in "${CELLS[@]}"; do
  IFS='|' read -r base model quant <<< "$spec"
  compare_pair "decode_logprob_kld_multi.py" \
    "$BF16_MULTI" "${RESULTS}/${base}-multi.safetensors" \
    "${COMPARE}/${base}-vs-bf16-multi.json"
  compare_pair "decode_logprob_kld.py" \
    "$BF16_SINGLE" "${RESULTS}/${base}-single.safetensors" \
    "${COMPARE}/${base}-vs-bf16-single.json"
done

echo ""
echo "==================================================================="
echo "Phase 2 complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================================="
ls -la "$COMPARE/"
