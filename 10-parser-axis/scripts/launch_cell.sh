#!/usr/bin/env bash
# launch_cell.sh — start a single Exp 10 vLLM endpoint cell
#
# Usage: launch_cell.sh <id> <image> <quant> <spec_method> <spec_tokens> <tool_parser> <reasoning_parser> <model>
#
# Idempotently kills any existing vllm-qwen36-27b-tp2 container and starts the
# requested configuration. Blocks until /health returns 200 or timeout.
#
# Exits 0 on healthy endpoint; non-zero on boot failure.

set -uo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 <id> <image> <quant> <spec_method> <spec_tokens> <tool_parser> <reasoning_parser> <model>" >&2
  exit 64
fi

ID="$1"
IMAGE="$2"
QUANT="$3"           # fp8 | bf16
SPEC_METHOD="$4"     # mtp | dflash
SPEC_TOKENS="$5"     # 3 | 8
TOOL_PARSER="$6"     # qwen3_xml | qwen3_coder
REASONING_PARSER="$7"
MODEL="$8"

CONTAINER="vllm-qwen36-27b-tp2"
PORT=8000
BOOT_TIMEOUT=600   # seconds — cold boot can take up to ~5-6 min; warm boot ~60s
GPU_1_UUID="GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9"
GPU_2_UUID="GPU-d558b2e1-bdf6-47b3-7d99-556339354cd1"

log() { printf '[%s][launch:%s] %s\n' "$(date '+%H:%M:%S')" "$ID" "$*"; }

log "Stopping any existing $CONTAINER"
docker stop "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true

# Build vLLM args based on quant + spec method
VLLM_ARGS=(
  -O3
  --model "$MODEL"
  --served-model-name Qwen3.6-27B
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.88
  --max-model-len 134144
  --max-num-seqs 128
  --max-num-batched-tokens 32768
  --max-cudagraph-capture-size 256
  --language-model-only
  --enable-auto-tool-choice
  --reasoning-parser "$REASONING_PARSER"
  --tool-call-parser "$TOOL_PARSER"
  --enable-prefix-caching
  --speculative-config.method "$SPEC_METHOD"
  --speculative-config.num_speculative_tokens "$SPEC_TOKENS"
  --attention-backend flashinfer
  --default-chat-template-kwargs.preserve_thinking true
)

# DFlash needs the draft model + extra args.
# IMPORTANT: --gdn-decode-backend and --speculative-config.use_local_argmax_reduction
# are rejected by `vllm serve` (Exp 09 BF16+DFlash recipe omitted them). They appear in
# the active FP8 launcher (launch-qwen36-27b-sota.sh) but only because that launcher
# is never re-launched on v13 — it was authored against an older image. Match the
# Exp 09 working recipe exactly to avoid container-boot failures.
if [[ "$SPEC_METHOD" == "dflash" ]]; then
  VLLM_ARGS+=(
    --speculative-config.model z-lab/Qwen3.6-27B-DFlash
    --speculative-config.attention_backend flashinfer
    --speculative-config.draft_sample_method greedy
  )
fi

log "Starting $CONTAINER (image=$IMAGE quant=$QUANT spec=$SPEC_METHOD/N$SPEC_TOKENS parser=$TOOL_PARSER)"
docker run -d \
  --name "$CONTAINER" \
  --device "nvidia.com/gpu=${GPU_1_UUID}" \
  --device "nvidia.com/gpu=${GPU_2_UUID}" \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  --volume /home/josh/.cache/huggingface:/root/.cache/huggingface \
  --volume /home/josh/.cache/vllm:/root/.cache/vllm \
  --volume /home/josh/.cache/flashinfer:/root/.cache/flashinfer \
  --volume /home/josh/.triton/cache:/root/.triton/cache \
  --env OMP_NUM_THREADS=8 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  "$IMAGE" \
  "${VLLM_ARGS[@]}" > /tmp/exp10-launch-$ID.cid 2>&1

CID_FILE=/tmp/exp10-launch-$ID.cid
if [[ ! -s "$CID_FILE" ]]; then
  log "FAILED: docker run produced no output"
  cat "$CID_FILE" 2>/dev/null || true
  exit 1
fi
CID=$(head -c 12 "$CID_FILE" 2>/dev/null | tr -d '[:space:]')
log "Container started: $CID"

# Wait for /health
log "Waiting up to ${BOOT_TIMEOUT}s for /health"
START_T=$(date +%s)
while true; do
  ELAPSED=$(( $(date +%s) - START_T ))
  if [[ $ELAPSED -gt $BOOT_TIMEOUT ]]; then
    log "TIMEOUT after ${ELAPSED}s; last 30 lines of container log:"
    docker logs --tail 30 "$CONTAINER" 2>&1 | sed "s/^/[$ID logs] /"
    exit 2
  fi
  STATUS=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
  if [[ "$STATUS" == "200" ]]; then
    log "HEALTHY after ${ELAPSED}s"
    # Verify served model is right + parser config
    MODEL_LIST=$(curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null)
    log "Models endpoint: $(echo "$MODEL_LIST" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join(m['id'] for m in d.get('data',[])))" 2>/dev/null || echo 'parse-err')"
    exit 0
  fi
  # Check if container is still running
  if ! docker ps -q --filter "name=$CONTAINER" | grep -q .; then
    log "CONTAINER CRASHED after ${ELAPSED}s; last 50 lines:"
    docker logs --tail 50 "$CONTAINER" 2>&1 | sed "s/^/[$ID logs] /"
    exit 3
  fi
  sleep 5
done
