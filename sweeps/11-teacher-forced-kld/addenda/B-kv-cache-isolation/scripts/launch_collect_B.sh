#!/usr/bin/env bash
# Addendum B: KV-cache isolation cell launcher.
# Thin wrapper around the Exp 11 collect scripts, with kv-cache-dtype
# overrideable (the main launch_collect.sh hardcodes "auto").
#
# Required env:
#   LABEL          e.g. B1-bf16-kv-auto
#   MODEL          HF repo id
#   OUTPUT         Absolute host path (safetensors)
#   QUANT          none | fp8
#   KV_CACHE_DTYPE auto | fp8
#
# Mode is always "multi" (504 positions needed for KV-cache effects to show).

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
SCRIPT="decode_logprob_kld_multi.py"

: "${LABEL:?}"
: "${MODEL:?}"
: "${OUTPUT:?}"
: "${QUANT:?}"
: "${KV_CACHE_DTYPE:?}"

TOKEN_SOURCE="${TOKEN_SOURCE:-${SWEEP_DIR}/refs/wikitext-tokens-multi.safetensors}"
GPU_A="GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9"
GPU_B="GPU-d558b2e1-bdf6-47b3-7d99-556339354cd1"
CONTAINER="vllm-addB-${LABEL}"
docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true

OUTPUT_IN_CONTAINER="${OUTPUT/$SWEEP_DIR/\/sweep}"
TOKEN_IN_CONTAINER="${TOKEN_SOURCE/$SWEEP_DIR/\/sweep}"
mkdir -p "$(dirname "$OUTPUT")"

LOGDIR="${SWEEP_DIR}/addenda/B-kv-cache-isolation/logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/${LABEL}.log"

echo "=== Addendum B collect: $LABEL (kv=$KV_CACHE_DTYPE quant=$QUANT) ===" | tee -a "$LOGFILE"

exec > >(tee -a "$LOGFILE") 2>&1

docker run --rm \
  --name "$CONTAINER" \
  --device "nvidia.com/gpu=${GPU_A}" \
  --device "nvidia.com/gpu=${GPU_B}" \
  --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --volume /home/josh/.cache/huggingface:/root/.cache/huggingface \
  --volume /home/josh/.cache/vllm:/root/.cache/vllm \
  --volume /home/josh/.triton/cache:/root/.triton/cache \
  --volume "${SWEEP_DIR}":/sweep \
  --volume "${SWEEP_DIR}/scripts/${SCRIPT}":/workspace/${SCRIPT}:ro \
  --volume "${SWEEP_DIR}/scripts/teacher_force_logits_processor.py":/workspace/teacher_force_logits_processor.py:ro \
  --env OMP_NUM_THREADS=8 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env NCCL_P2P_LEVEL=SYS \
  --env NCCL_NET_GDR_LEVEL=SYS \
  --env PYTHONPATH=/workspace \
  --env B_LABEL="$LABEL" \
  --env B_MODEL="$MODEL" \
  --env B_OUTPUT="$OUTPUT_IN_CONTAINER" \
  --env B_TOKEN_SOURCE="$TOKEN_IN_CONTAINER" \
  --env B_QUANT="$QUANT" \
  --env B_KV_DTYPE="$KV_CACHE_DTYPE" \
  --entrypoint /bin/bash \
  repne/vllm:v13 \
  -lc '
set -euo pipefail
cd /workspace
ARGS=(
  collect
  --label "$B_LABEL"
  --model "$B_MODEL"
  --output "$B_OUTPUT"
  --token-source "$B_TOKEN_SOURCE"
  --num-prompts 8
  --prompt-len 2048
  --max-tokens 64
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.85
  --dtype bfloat16
  --kv-cache-dtype "$B_KV_DTYPE"
  --load-format auto
  --max-model-len 4096
  --max-num-batched-tokens 4096
  --quantization "$B_QUANT"
  --attention-backend FLASHINFER
  --moe-backend auto
  --teacher-force
  --language-model-only
  --disable-custom-all-reduce
)
echo "=== Inner argv ===" && for a in "${ARGS[@]}"; do echo "  $a"; done && echo "==="
exec python3 decode_logprob_kld_multi.py "${ARGS[@]}"
'

sudo chown -R josh:josh "$(dirname "$OUTPUT")" 2>/dev/null || true
echo "=== Done: $LABEL ==="
ls -la "$OUTPUT" "$OUTPUT.json" 2>/dev/null || true
