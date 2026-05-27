#!/usr/bin/env bash
# Phase 3: Run all pairwise comparisons between BF16 refs and variants.
#
# Produces:
#   compare/<variant>-vs-bf16-single.json
#   compare/<variant>-vs-bf16-multi.json
#   compare/bf16-self-single.json    (noise floor; computed last)
#   compare/bf16-self-multi.json     (noise floor; computed last)
#   compare/leaderboard.csv
#
# Comparisons run inside repne/vllm:v13 to ensure the same torch version
# parses safetensors identically. (No GPUs needed for compare; CPU-only.)

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
REFS="${SWEEP_DIR}/refs"
RESULTS="${SWEEP_DIR}/results"
COMPARE="${SWEEP_DIR}/compare"

mkdir -p "$COMPARE"

# Run inside the same image to guarantee numerical parity
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
  local script="$1"; local a="$2"; local b="$3"; local out="$4"
  if [[ ! -f "$a" ]]; then echo "SKIP $out (missing $a)"; return 0; fi
  if [[ ! -f "$b" ]]; then echo "SKIP $out (missing $b)"; return 0; fi
  if [[ -f "$out" ]]; then echo "EXISTS $out (delete to recompute)"; return 0; fi
  local a_in="${a/$SWEEP_DIR/\/sweep}"
  local b_in="${b/$SWEEP_DIR/\/sweep}"
  local out_in="${out/$SWEEP_DIR/\/sweep}"
  echo "--- $script compare ---"
  echo "  a: $a"
  echo "  b: $b"
  echo "  out: $out"
  "${DOCKER_RUN[@]}" "cd /workspace && python3 $script compare --a $a_in --b $b_in --output $out_in"
  sudo chown -R josh:josh "$(dirname "$out")" 2>/dev/null || true
}

# === Single-prompt comparisons (17 tok × 1 prompt) ===
echo ""; echo "=== Single-prompt comparisons (skip_prefill_next=1 default) ==="
for variant in fp8 fp8-mtp3 nvfp4; do
  compare_pair "decode_logprob_kld.py" \
    "$REFS/bf16-ref-single.safetensors" \
    "$RESULTS/${variant}-single.safetensors" \
    "$COMPARE/${variant}-vs-bf16-single.json"
done

# === Multi-prompt comparisons (64 tok × 8 prompts) ===
echo ""; echo "=== Multi-prompt comparisons (skip_prefill_next=1 default) ==="
for variant in fp8 fp8-mtp3 nvfp4; do
  compare_pair "decode_logprob_kld_multi.py" \
    "$REFS/bf16-ref-multi.safetensors" \
    "$RESULTS/${variant}-multi.safetensors" \
    "$COMPARE/${variant}-vs-bf16-multi.json"
done

# === BF16-self (noise floor) ===
# Optional: only run if bf16-self refs exist (Phase 5)
echo ""; echo "=== BF16-self noise floor (if Phase 5 collected) ==="
compare_pair "decode_logprob_kld.py" \
  "$REFS/bf16-ref-single.safetensors" \
  "$REFS/bf16-self-single.safetensors" \
  "$COMPARE/bf16-self-single.json"
compare_pair "decode_logprob_kld_multi.py" \
  "$REFS/bf16-ref-multi.safetensors" \
  "$REFS/bf16-self-multi.safetensors" \
  "$COMPARE/bf16-self-multi.json"

echo ""
echo "=== Compare outputs ==="
ls -la "$COMPARE"
