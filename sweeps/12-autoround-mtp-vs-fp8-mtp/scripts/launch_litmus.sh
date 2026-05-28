#!/usr/bin/env bash
# Litmus test: does AutoRound-int4 + MTP=3 even boot on vLLM-Repne v13 / Blackwell SM120?
#
# Pass criteria:
#   1. Container reaches /health 200 within 8 minutes
#   2. Single completion via /v1/completions returns coherent tokens
#   3. No "speculative config rejected" or "quantization+spec_decode unsupported" in logs
#
# Resource: GPUs 1+2 (display GPU 0 reserved), TP=2
# Model: Lorbus/Qwen3.6-27B-int4-AutoRound (has MTP weights, verified static)

set -uo pipefail

CONTAINER_NAME="vllm-litmus-autoround-mtp"
GPU_1_UUID="GPU-d558b2e1-bdf6-47b3-7d99-556339354cd1"
# Use GPU 2 instead of GPU 0 (display GPU reserved per policy)
GPU_2_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')

echo "Using GPUs: 1=$GPU_1_UUID, 2=$GPU_2_UUID"
echo "Model: Lorbus/Qwen3.6-27B-int4-AutoRound (cached, ~18GB)"
echo "Spec: MTP=3"
echo

docker stop "$CONTAINER_NAME" 2>/dev/null; docker rm "$CONTAINER_NAME" 2>/dev/null

docker run -d \
  --name "$CONTAINER_NAME" \
  --device "nvidia.com/gpu=${GPU_1_UUID}" \
  --device "nvidia.com/gpu=${GPU_2_UUID}" \
  --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8765:8000 \
  --volume /home/josh/.cache/huggingface:/root/.cache/huggingface \
  --volume /home/josh/.cache/vllm:/root/.cache/vllm \
  --volume /home/josh/.triton/cache:/root/.triton/cache \
  --env OMP_NUM_THREADS=8 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  repne/vllm:v13 \
    -O3 \
    --model Lorbus/Qwen3.6-27B-int4-AutoRound \
    --served-model-name Qwen3.6-27B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 16384 \
    --max-cudagraph-capture-size 64 \
    --language-model-only \
    --enable-prefix-caching \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 \
    --attention-backend flashinfer

echo "Launched. Watching for boot signals (8min timeout)..."
SECS=0
LIMIT=480
while (( SECS < LIMIT )); do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[FAIL @${SECS}s] Container exited"
    docker logs --tail 50 "$CONTAINER_NAME" 2>&1
    exit 1
  fi
  if curl -sf http://localhost:8765/health >/dev/null 2>&1; then
    echo "[OK @${SECS}s] /health 200"
    break
  fi
  # Check for known failure patterns in logs
  if docker logs "$CONTAINER_NAME" 2>&1 | grep -qE "speculative.*not supported|quantization.*not compatible|AutoRound.*not supported|EngineCore failed|NotImplementedError|ValueError.*spec"; then
    echo "[FAIL @${SECS}s] Detected incompatibility in logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | grep -E "speculative|quantization|AutoRound|Error|NotImplemented|ValueError" | tail -20
    exit 2
  fi
  sleep 10
  SECS=$((SECS + 10))
done

if (( SECS >= LIMIT )); then
  echo "[TIMEOUT] No /health 200 after ${LIMIT}s"
  docker logs --tail 60 "$CONTAINER_NAME" 2>&1
  exit 3
fi

# Single completion test
echo
echo "=== Single completion test ==="
curl -s -X POST http://localhost:8765/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-27B","prompt":"The capital of France is","max_tokens":20,"temperature":0.0}' \
  | python3 -m json.tool

echo
echo "=== Look for spec_decode acceptance in metrics ==="
sleep 2
docker logs "$CONTAINER_NAME" 2>&1 | grep -iE "speculative|draft|accept|mtp" | tail -20
