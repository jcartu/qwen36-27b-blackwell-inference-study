#!/usr/bin/env bash
# Phase 1+ production collect launcher for Exp 11.
#
# Wraps decode_logprob_kld{,_multi}.py to run a single (label, model, variant)
# cell on repne/vllm:v13 with GPUs 1+2 (TP=2). GPU 0 is reserved for display.
#
# Required env:
#   LABEL          e.g. bf16-ref-single, fp8-single, fp8-mtp3-multi, nvfp4-multi
#   MODEL          HF repo id (e.g. Qwen/Qwen3.6-27B, Qwen/Qwen3.6-27B-FP8,
#                  sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP)
#   OUTPUT         Absolute host path to write safetensors (under SWEEP_DIR)
#   MODE           "single" or "multi" (chooses script)
#   QUANT          "none" | "fp8" | "modelopt_fp4" (passed to --quantization)
#
# Optional env:
#   MTP3=1                 Enable MTP=3 speculative decode (FP8 only)
#   TOKEN_SOURCE=<path>    Pin tokens to a previous run's safetensors
#   MAX_TOKENS=<int>       Override default (17 single, 64 multi)
#   NUM_PROMPTS=<int>      Override default (8, multi only)
#   PROMPT_LEN=<int>       Override default (2048)
#   EXTRA_ARGS=""          Extra args to forward to the collect script
#
# Pure pass-through to the container. No prompt-coordination logic here.

set -euo pipefail

SWEEP_DIR="/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld"
CONTAINER_PREFIX="vllm-exp11"

: "${LABEL:?LABEL is required}"
: "${MODEL:?MODEL is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${MODE:?MODE is required (single|multi)}"
: "${QUANT:?QUANT is required (none|fp8|modelopt_fp4)}"

CONTAINER="${CONTAINER_PREFIX}-${LABEL}"
docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true

# GPUs 1 and 2 (GPU 0 = display, ~3342 MiB baseline, reserved)
GPU_A_UUID="GPU-ae60c9dd-9ddc-cc7d-3a9e-f78cbeb00de9"  # GPU 2
GPU_B_UUID="GPU-d558b2e1-bdf6-47b3-7d99-556339354cd1"  # GPU 1

# Defaults
PROMPT_LEN="${PROMPT_LEN:-2048}"
case "$MODE" in
  single)
    SCRIPT="decode_logprob_kld.py"
    MAX_TOKENS="${MAX_TOKENS:-17}"
    MODE_ARGS=""
    ;;
  multi)
    SCRIPT="decode_logprob_kld_multi.py"
    MAX_TOKENS="${MAX_TOKENS:-64}"
    NUM_PROMPTS="${NUM_PROMPTS:-8}"
    MODE_ARGS="--num-prompts ${NUM_PROMPTS}"
    ;;
  *) echo "ERROR: MODE must be 'single' or 'multi', got '$MODE'" >&2; exit 2 ;;
esac

# Optional MTP=3 (sent via env to avoid shell-quoting hell with JSON)
SPEC_CONFIG_JSON=""
if [[ "${MTP3:-0}" == "1" ]]; then
  SPEC_CONFIG_JSON='{"method":"mtp","num_speculative_tokens":3}'
fi

# Optional token source (host path → container path)
TOKEN_SOURCE_ARGS=""
if [[ -n "${TOKEN_SOURCE:-}" ]]; then
  if [[ ! -f "$TOKEN_SOURCE" ]]; then
    echo "ERROR: TOKEN_SOURCE file does not exist: $TOKEN_SOURCE" >&2
    exit 3
  fi
  # We mount SWEEP_DIR → /sweep so translate the path
  TOKEN_SOURCE_IN_CONTAINER="${TOKEN_SOURCE/$SWEEP_DIR/\/sweep}"
  TOKEN_SOURCE_ARGS="--token-source ${TOKEN_SOURCE_IN_CONTAINER}"
fi

# OUTPUT host → container
if [[ "$OUTPUT" != "$SWEEP_DIR"/* ]]; then
  echo "ERROR: OUTPUT must be under $SWEEP_DIR, got $OUTPUT" >&2
  exit 4
fi
OUTPUT_IN_CONTAINER="${OUTPUT/$SWEEP_DIR/\/sweep}"
mkdir -p "$(dirname "$OUTPUT")"

LOGFILE="${SWEEP_DIR}/logs/${LABEL}.log"
mkdir -p "$(dirname "$LOGFILE")"

echo "=== Exp 11 collect: $LABEL ==="
echo "  model:        $MODEL"
echo "  mode:         $MODE"
echo "  quant:        $QUANT"
echo "  output:       $OUTPUT"
echo "  prompt_len:   $PROMPT_LEN"
echo "  max_tokens:   $MAX_TOKENS"
echo "  mtp3:         ${MTP3:-0}"
echo "  token_source: ${TOKEN_SOURCE:-<wikitext default>}"
echo "  log:          $LOGFILE"
echo "  container:    $CONTAINER"
echo "  GPUs:         1, 2"
echo "==="

# Stream to console AND file
exec > >(tee -a "$LOGFILE") 2>&1

# Build the inner python command argv as env vars; inner shell assembles
# them safely with proper quoting. SPEC_CONFIG is passed through env to
# preserve its JSON quoting through the shell layers.
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
  --volume "${SWEEP_DIR}/scripts/teacher_force_logits_processor.py":/workspace/teacher_force_logits_processor.py:ro \
  --volume "${SWEEP_DIR}/scripts/${SCRIPT}":/workspace/${SCRIPT}:ro \
  --env OMP_NUM_THREADS=8 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env PYTHONPATH=/workspace \
  --env EXP11_LABEL="${LABEL}" \
  --env EXP11_MODEL="${MODEL}" \
  --env EXP11_OUTPUT="${OUTPUT_IN_CONTAINER}" \
  --env EXP11_PROMPT_LEN="${PROMPT_LEN}" \
  --env EXP11_MAX_TOKENS="${MAX_TOKENS}" \
  --env EXP11_MODE_ARGS="${MODE_ARGS}" \
  --env EXP11_QUANT="${QUANT}" \
  --env EXP11_SCRIPT="${SCRIPT}" \
  --env EXP11_SPEC_CONFIG="${SPEC_CONFIG_JSON}" \
  --env EXP11_TOKEN_SOURCE="${TOKEN_SOURCE_ARGS}" \
  --env EXP11_EXTRA_ARGS="${EXTRA_ARGS:-}" \
  --entrypoint /bin/bash \
  repne/vllm:v13 \
  -lc '
set -euo pipefail
cd /workspace
ARGS=(
  collect
  --label "$EXP11_LABEL"
  --model "$EXP11_MODEL"
  --output "$EXP11_OUTPUT"
  --prompt-len "$EXP11_PROMPT_LEN"
  --max-tokens "$EXP11_MAX_TOKENS"
  --tensor-parallel-size 2
  --gpu-memory-utilization 0.85
  --dtype bfloat16
  --kv-cache-dtype auto
  --load-format auto
  --max-model-len 4096
  --max-num-batched-tokens 4096
  --quantization "$EXP11_QUANT"
  --attention-backend FLASHINFER
  --moe-backend auto
  --hf-overrides {}
  --teacher-force
  --disable-custom-all-reduce
  --language-model-only
)
if [ -n "$EXP11_MODE_ARGS" ]; then
  # word-split mode args (e.g. "--num-prompts 8")
  read -r -a EXTRA <<< "$EXP11_MODE_ARGS"
  ARGS+=("${EXTRA[@]}")
fi
if [ -n "$EXP11_TOKEN_SOURCE" ]; then
  read -r -a EXTRA <<< "$EXP11_TOKEN_SOURCE"
  ARGS+=("${EXTRA[@]}")
fi
if [ -n "$EXP11_SPEC_CONFIG" ]; then
  ARGS+=(--speculative-config "$EXP11_SPEC_CONFIG")
fi
if [ -n "$EXP11_EXTRA_ARGS" ]; then
  read -r -a EXTRA <<< "$EXP11_EXTRA_ARGS"
  ARGS+=("${EXTRA[@]}")
fi
echo "=== Inner argv ==="
for a in "${ARGS[@]}"; do echo "  $a"; done
echo "=================="
exec python3 "$EXP11_SCRIPT" "${ARGS[@]}"
'

# Fix ownership on outputs (docker writes as root)
sudo chown -R josh:josh "$(dirname "$OUTPUT")" 2>/dev/null || true

echo "=== Done: $LABEL ==="
ls -la "$OUTPUT" "$OUTPUT.json" 2>/dev/null || true
