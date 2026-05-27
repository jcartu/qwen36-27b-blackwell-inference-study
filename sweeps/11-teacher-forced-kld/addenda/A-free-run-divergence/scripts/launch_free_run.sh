#!/usr/bin/env bash
# Addendum A: Free-run generation launcher.
# Wraps free_run_generate.py in the repne/vllm:v13 container on GPUs 1+2 (TP=2).
#
# Required env:
#   LABEL           e.g. bf16-ref-a, fp8-no-spec, fp8-mtp3, fp8-mtp5
#   MODEL           HF repo id
#   OUTPUT          Absolute host path for output JSON
#   QUANT           none | fp8 | modelopt_fp4
#
# Optional env:
#   MAX_TOKENS=1024           (default)
#   KV_CACHE_DTYPE=auto       (default; use "fp8" for FP8-KV cells)
#   SPEC_CONFIG=              (JSON string for MTP; empty = no spec)
#   ENFORCE_EAGER=0           (set to 1 for NVFP4)
#   SEED=42

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
ADDENDUM_DIR="${SWEEP_DIR}/addenda/A-free-run-divergence"
SCRIPT_NAME="free_run_generate.py"

: "${LABEL:?LABEL is required}"
: "${MODEL:?MODEL is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${QUANT:?QUANT is required (none|fp8|modelopt_fp4)}"

MAX_TOKENS="${MAX_TOKENS:-1024}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
SPEC_CONFIG="${SPEC_CONFIG:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
SEED="${SEED:-42}"
TOKEN_SOURCE="${TOKEN_SOURCE:-${SWEEP_DIR}/refs/wikitext-tokens-multi.safetensors}"

GPU_A_UUID="GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9"  # GPU 2
GPU_B_UUID="GPU-d558b2e1-bdf6-47b3-7d99-556339354cd1"  # GPU 1

CONTAINER="vllm-addA-${LABEL}"
docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true

OUTPUT_IN_CONTAINER="${OUTPUT/$SWEEP_DIR/\/sweep}"
TOKEN_SOURCE_IN_CONTAINER="${TOKEN_SOURCE/$SWEEP_DIR/\/sweep}"
mkdir -p "$(dirname "$OUTPUT")"

LOGFILE="${ADDENDUM_DIR}/logs/${LABEL}.log"
mkdir -p "$(dirname "$LOGFILE")"

echo "=== Addendum A free-run: $LABEL ==="
echo "  model:          $MODEL"
echo "  quant:          $QUANT"
echo "  kv_cache_dtype: $KV_CACHE_DTYPE"
echo "  max_tokens:     $MAX_TOKENS"
echo "  spec_config:    ${SPEC_CONFIG:-<none>}"
echo "  enforce_eager:  $ENFORCE_EAGER"
echo "  seed:           $SEED"
echo "  output:         $OUTPUT"
echo "==="

exec > >(tee -a "$LOGFILE") 2>&1

EAGER_FLAG=""
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  EAGER_FLAG="--enforce-eager"
fi

docker run --rm \
  --name "$CONTAINER" \
  --device "nvidia.com/gpu=${GPU_A_UUID}" \
  --device "nvidia.com/gpu=${GPU_B_UUID}" \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  --volume /home/josh/.cache/huggingface:/root/.cache/huggingface \
  --volume /home/josh/.cache/vllm:/root/.cache/vllm \
  --volume /home/josh/.triton/cache:/root/.triton/cache \
  --volume "${SWEEP_DIR}":/sweep \
  --volume "${ADDENDUM_DIR}/scripts/${SCRIPT_NAME}":/workspace/${SCRIPT_NAME}:ro \
  --env OMP_NUM_THREADS=8 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env NCCL_P2P_LEVEL=SYS \
  --env NCCL_NET_GDR_LEVEL=SYS \
  --env PYTHONPATH=/workspace \
  --env FR_LABEL="${LABEL}" \
  --env FR_MODEL="${MODEL}" \
  --env FR_OUTPUT="${OUTPUT_IN_CONTAINER}" \
  --env FR_TOKEN_SOURCE="${TOKEN_SOURCE_IN_CONTAINER}" \
  --env FR_MAX_TOKENS="${MAX_TOKENS}" \
  --env FR_KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
  --env FR_SPEC_CONFIG="${SPEC_CONFIG}" \
  --env FR_QUANT="${QUANT}" \
  --env FR_EAGER_FLAG="${EAGER_FLAG}" \
  --env FR_SEED="${SEED}" \
  --entrypoint /bin/bash \
  repne/vllm:v13 \
  -lc '
set -euo pipefail
cd /workspace
ARGS=(
  --label "$FR_LABEL"
  --model "$FR_MODEL"
  --token-source "$FR_TOKEN_SOURCE"
  --output "$FR_OUTPUT"
  --max-tokens "$FR_MAX_TOKENS"
  --seed "$FR_SEED"
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.85
  --dtype bfloat16
  --kv-cache-dtype "$FR_KV_CACHE_DTYPE"
  --load-format auto
  --max-model-len 4096
  --max-num-batched-tokens 4096
  --quantization "$FR_QUANT"
  --attention-backend FLASHINFER
  --moe-backend auto
  --language-model-only
  --disable-custom-all-reduce
)
if [ -n "$FR_EAGER_FLAG" ]; then
  ARGS+=($FR_EAGER_FLAG)
fi
if [ -n "$FR_SPEC_CONFIG" ]; then
  ARGS+=(--speculative-config "$FR_SPEC_CONFIG")
fi
echo "=== Inner argv ===" && for a in "${ARGS[@]}"; do echo "  $a"; done && echo "==="
exec python3 '"'"'free_run_generate.py'"'"' "${ARGS[@]}"
'

sudo chown -R josh:josh "$(dirname "$OUTPUT")" 2>/dev/null || true
echo "=== Done: $LABEL → $OUTPUT ==="
