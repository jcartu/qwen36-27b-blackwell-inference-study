#!/usr/bin/env bash
# Addendum C: teacher-forced collect for the 5 v2 quant cells (C4-C8).
# Closes the unified-leaderboard gap by running the same teacher-forced
# protocol used in Exp 11 (BF16/FP8/NVFP4) and Addendum B (KV cache).
#
# Required env:
#   LABEL           e.g. C4-awq-4bit-multi
#   MODEL           HF repo id
#   OUTPUT          Absolute host path (safetensors) under SWEEP_DIR
#   QUANT           awq_marlin | auto-round | gptq_marlin
#   MODE            multi | single
#
# Hardware contract (unchanged):
#   GPUs 1+2, TP=2, KV-cache=auto, repne/vllm:v13.

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"

: "${LABEL:?}"; : "${MODEL:?}"; : "${OUTPUT:?}"; : "${QUANT:?}"; : "${MODE:?}"

case "$MODE" in
  multi)  SCRIPT="decode_logprob_kld_multi.py"; MAX_TOKENS="${MAX_TOKENS:-64}"; NUM_PROMPTS="${NUM_PROMPTS:-8}"
          TOKEN_SOURCE_DEFAULT="${SWEEP_DIR}/refs/wikitext-tokens-multi.safetensors" ;;
  single) SCRIPT="decode_logprob_kld.py";       MAX_TOKENS="${MAX_TOKENS:-17}"; NUM_PROMPTS=""
          TOKEN_SOURCE_DEFAULT="${SWEEP_DIR}/refs/wikitext-tokens-single.safetensors" ;;
  *) echo "ERROR: MODE must be multi|single" >&2; exit 2 ;;
esac
TOKEN_SOURCE="${TOKEN_SOURCE:-$TOKEN_SOURCE_DEFAULT}"

GPU_A="GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9"  # GPU 2
GPU_B="GPU-d558b2e1-bdf6-47b3-7d99-556339354cd1"  # GPU 1
CONTAINER="vllm-addC-${LABEL}"
docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true

OUTPUT_IN_CONTAINER="${OUTPUT/$SWEEP_DIR/\/sweep}"
TOKEN_IN_CONTAINER="${TOKEN_SOURCE/$SWEEP_DIR/\/sweep}"
mkdir -p "$(dirname "$OUTPUT")"

LOGDIR="${SWEEP_DIR}/addenda/C-teacher-forced-quants/logs"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/${LABEL}.log"

echo "=== Addendum C collect: $LABEL (mode=$MODE quant=$QUANT) ===" | tee -a "$LOGFILE"
echo "  model:  $MODEL"  | tee -a "$LOGFILE"
echo "  output: $OUTPUT" | tee -a "$LOGFILE"
echo "  tokens: $TOKEN_SOURCE" | tee -a "$LOGFILE"
echo "  GPUs:   1, 2 (TP=2)" | tee -a "$LOGFILE"

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
  --env C_LABEL="$LABEL" \
  --env C_MODEL="$MODEL" \
  --env C_OUTPUT="$OUTPUT_IN_CONTAINER" \
  --env C_TOKEN_SOURCE="$TOKEN_IN_CONTAINER" \
  --env C_QUANT="$QUANT" \
  --env C_MAX_TOKENS="$MAX_TOKENS" \
  --env C_NUM_PROMPTS="$NUM_PROMPTS" \
  --env C_SCRIPT="$SCRIPT" \
  --entrypoint /bin/bash \
  repne/vllm:v13 \
  -lc '
set -euo pipefail
cd /workspace
ARGS=(
  collect
  --label "$C_LABEL"
  --model "$C_MODEL"
  --output "$C_OUTPUT"
  --token-source "$C_TOKEN_SOURCE"
  --prompt-len 2048
  --max-tokens "$C_MAX_TOKENS"
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.85
  --dtype bfloat16
  --kv-cache-dtype auto
  --load-format auto
  --max-model-len 4096
  --max-num-batched-tokens 4096
  --quantization "$C_QUANT"
  --attention-backend FLASHINFER
  --moe-backend auto
  --teacher-force
  --language-model-only
  --disable-custom-all-reduce
)
if [ -n "$C_NUM_PROMPTS" ]; then
  ARGS+=(--num-prompts "$C_NUM_PROMPTS")
fi
echo "=== Inner argv ===" && for a in "${ARGS[@]}"; do echo "  $a"; done && echo "==="
exec python3 "$C_SCRIPT" "${ARGS[@]}"
'

sudo chown -R josh:josh "$(dirname "$OUTPUT")" 2>/dev/null || true
echo "=== Done: $LABEL ==="
ls -la "$OUTPUT" "$OUTPUT.json" 2>/dev/null || true
