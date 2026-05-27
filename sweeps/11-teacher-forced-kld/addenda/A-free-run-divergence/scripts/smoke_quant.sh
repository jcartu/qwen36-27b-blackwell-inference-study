#!/usr/bin/env bash
# Smoke-test a single quantization on GPUs 1+2 (TP=2).
# Loads model, generates 32 tokens, prints status + sample.
#
# Required env:
#   SMOKE_LABEL  e.g. awq-4bit
#   SMOKE_MODEL  HF repo id
#   SMOKE_QUANT  e.g. awq_marlin, gptq_marlin, auto-round, none
# Optional:
#   SMOKE_EAGER=0|1   (default 1 = faster startup, ok for smoke)

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
ADD_DIR="${SWEEP_DIR}/addenda/A-free-run-divergence"
SCRIPTS_DIR="${ADD_DIR}/scripts"
LOG_DIR="${ADD_DIR}/logs/smoke"

: "${SMOKE_LABEL:?SMOKE_LABEL required}"
: "${SMOKE_MODEL:?SMOKE_MODEL required}"
: "${SMOKE_QUANT:?SMOKE_QUANT required}"
SMOKE_EAGER="${SMOKE_EAGER:-1}"

mkdir -p "$LOG_DIR"

GPU_A_UUID="GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9"
GPU_B_UUID="GPU-d558b2e1-bdf6-47b3-7d99-556339354cd1"

CONTAINER="vllm-smoke-${SMOKE_LABEL}"
docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true

LOGFILE="${LOG_DIR}/${SMOKE_LABEL}.log"
{
  echo "=== SMOKE: $SMOKE_LABEL ==="
  echo "  model: $SMOKE_MODEL"
  echo "  quant: $SMOKE_QUANT"
  echo "  eager: $SMOKE_EAGER"
  echo "  start: $(date)"
} | tee "$LOGFILE"

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
  --volume "${SCRIPTS_DIR}/smoke_quant.py":/workspace/smoke_quant.py:ro \
  --env OMP_NUM_THREADS=8 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env NCCL_P2P_LEVEL=SYS \
  --env NCCL_NET_GDR_LEVEL=SYS \
  --env SMOKE_LABEL="${SMOKE_LABEL}" \
  --env SMOKE_MODEL="${SMOKE_MODEL}" \
  --env SMOKE_QUANT="${SMOKE_QUANT}" \
  --env SMOKE_EAGER="${SMOKE_EAGER}" \
  --entrypoint python3 \
  repne/vllm:v13 \
  /workspace/smoke_quant.py 2>&1 | tee -a "$LOGFILE"

EXIT=${PIPESTATUS[0]}
echo "=== END $SMOKE_LABEL exit=$EXIT $(date) ===" | tee -a "$LOGFILE"
exit "$EXIT"
