#!/usr/bin/env bash
# Launch AutoRound-int4 + MTP=3 with PRODUCTION-PARITY flags vs the FP8+MTP=3 SOTA launcher.
#
# Source: /home/josh/vllm-services/launch-qwen36-27b-tp2-sota.sh
# Diffs vs FP8 SOTA launcher:
#   - model:               Lorbus/Qwen3.6-27B-int4-AutoRound  (was Qwen/Qwen3.6-27B-FP8)
#   - port:                8765                                (was 8000, avoids prod collisions)
#   - GPUs:                1+2                                 (was 0+1, GPU 0 reserved for display)
# All other flags IDENTICAL to FP8+MTP=3 SOTA. This is the apples-to-apples comparison.
#
# Image: repne/vllm:v13

set -euo pipefail

CONTAINER_NAME="vllm-exp12-autoround-mtp"
GPU_1_UUID="GPU-d558b2e1-bdf6-47b3-7d99-556339354cd1"
GPU_2_UUID="GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9"

docker stop "$CONTAINER_NAME" 2>/dev/null || true; docker rm "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --device "nvidia.com/gpu=${GPU_1_UUID}" \
  --device "nvidia.com/gpu=${GPU_2_UUID}" \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
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
    --gpu-memory-utilization 0.88 \
    --max-model-len 262144 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 32768 \
    --max-cudagraph-capture-size 256 \
    --language-model-only \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-prefix-caching \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 \
    --attention-backend flashinfer \
    --default-chat-template-kwargs.preserve_thinking true

echo "Container: $CONTAINER_NAME (port 8765)"
echo "Wait for /health 200; cold-boot ~5-7min, warm-boot ~1-2min (compile cache hit)."
