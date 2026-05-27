#!/usr/bin/env bash
# Addendum B orchestrator: 2×2 KV-cache isolation grid.
#
# Cells (multi-prompt, 504 positions, same wikitext token source as Exp 11):
#   B1  BF16 weights + BF16 KV  → true clean reference (baseline)
#   B2  BF16 weights + FP8  KV  → isolates KV-cache quantization effect
#   B3  FP8  weights + BF16 KV  → isolates weight quantization effect
#   B4  FP8  weights + FP8  KV  → combined (should match Exp 11 FP8-multi)
#
# All compared vs B1 using existing compare script.
# B4 vs Exp 11 FP8-multi should be near-identical (sanity check).

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
ADD_DIR="${SWEEP_DIR}/addenda/B-kv-cache-isolation"
RES="${ADD_DIR}/results"
SCRIPTS="${ADD_DIR}/scripts"
COMPARE_SCRIPT="${SWEEP_DIR}/scripts/decode_logprob_kld_multi.py"

BF16_MODEL="Qwen/Qwen3.6-27B"
FP8_MODEL="Qwen/Qwen3.6-27B-FP8"

mkdir -p "$RES" "${ADD_DIR}/logs"
chmod +x "${SCRIPTS}/launch_collect_B.sh"

run_cell() {
  local label="$1" model="$2" quant="$3" kv="$4"
  echo ""
  echo "############################################################"
  echo "  CELL B: $label  (weights=$quant  kv=$kv)"
  echo "############################################################"
  LABEL="$label" \
  MODEL="$model" \
  OUTPUT="${RES}/${label}.safetensors" \
  QUANT="$quant" \
  KV_CACHE_DTYPE="$kv" \
  bash "${SCRIPTS}/launch_collect_B.sh"
}

echo "=== Addendum B: KV-cache isolation 2×2 grid ==="
echo "  4 cells × multi-prompt 504 positions"
echo "  GPUs 1+2 (TP=2), repne/vllm:v13"
echo "  Start: $(date)"

run_cell "B1-bf16-kv-auto" "$BF16_MODEL" "none" "auto"
run_cell "B2-bf16-kv-fp8"  "$BF16_MODEL" "none" "fp8"
run_cell "B3-fp8-kv-auto"  "$FP8_MODEL"  "fp8"  "auto"
run_cell "B4-fp8-kv-fp8"   "$FP8_MODEL"  "fp8"  "fp8"

echo ""
echo "=== All B cells complete: $(date) ==="
echo "=== Running comparisons vs B1 ==="

compare_vs_b1() {
  local cand="$1" out_label="$2"
  echo "  compare: $cand vs B1-bf16-kv-auto → $out_label"
  docker run --rm \
    --device "nvidia.com/gpu=GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9" \
    --ipc=host --network host \
    --volume /home/josh/.cache/huggingface:/root/.cache/huggingface \
    --volume "${SWEEP_DIR}":/sweep \
    --volume "${COMPARE_SCRIPT}":/workspace/decode_logprob_kld_multi.py:ro \
    --volume "${SWEEP_DIR}/scripts/teacher_force_logits_processor.py":/workspace/teacher_force_logits_processor.py:ro \
    --env PYTHONPATH=/workspace \
    --entrypoint /bin/bash \
    repne/vllm:v13 -lc "
      python3 /workspace/decode_logprob_kld_multi.py compare \
        --a /sweep/addenda/B-kv-cache-isolation/results/B1-bf16-kv-auto.safetensors \
        --b /sweep/addenda/B-kv-cache-isolation/results/${cand}.safetensors \
        --output /sweep/addenda/B-kv-cache-isolation/results/${out_label}.json \
        --skip-prefill-next 1
    "
  sudo chown -R josh:josh "$RES" 2>/dev/null || true
}

compare_vs_b1 "B2-bf16-kv-fp8" "cmp-B2-vs-B1-kv-effect"
compare_vs_b1 "B3-fp8-kv-auto" "cmp-B3-vs-B1-weight-effect"
compare_vs_b1 "B4-fp8-kv-fp8"  "cmp-B4-vs-B1-combined"

echo ""
echo "=== Sanity: B4 vs Exp 11 FP8-multi (should be near-zero KL) ==="
docker run --rm \
  --device "nvidia.com/gpu=GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9" \
  --ipc=host --network host \
  --volume /home/josh/.cache/huggingface:/root/.cache/huggingface \
  --volume "${SWEEP_DIR}":/sweep \
  --volume "${COMPARE_SCRIPT}":/workspace/decode_logprob_kld_multi.py:ro \
  --volume "${SWEEP_DIR}/scripts/teacher_force_logits_processor.py":/workspace/teacher_force_logits_processor.py:ro \
  --env PYTHONPATH=/workspace \
  --entrypoint /bin/bash \
  repne/vllm:v13 -lc "
    python3 /workspace/decode_logprob_kld_multi.py compare \
      --a /sweep/results/fp8-multi.safetensors \
      --b /sweep/addenda/B-kv-cache-isolation/results/B4-fp8-kv-fp8.safetensors \
      --output /sweep/addenda/B-kv-cache-isolation/results/sanity-B4-vs-exp11-fp8-multi.json \
      --skip-prefill-next 1
  "
sudo chown -R josh:josh "$RES" 2>/dev/null || true

echo ""
echo "=== Addendum B complete: $(date) ==="
ls -lh "${RES}"/*.json 2>/dev/null
