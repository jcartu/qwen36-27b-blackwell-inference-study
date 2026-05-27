#!/usr/bin/env bash
# Phase D — tool-eval-bench cross-provider showdown
# Runs the same scenarios across {our FP8, optionally our BF16, frontier APIs}
# Budget: 1 hour wall clock, $5 API cost cap (estimated ~$2 actual)
#
# Triggered manually after Pass B completes. Idempotent: skips endpoints whose
# results dir already exists with a results.json.
#
# Author: Sisyphus, 2026-05-27

set -uo pipefail
PHASE_D_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$PHASE_D_DIR/tool-eval-bench"
RESULTS="$PHASE_D_DIR/results"
LOG="$PHASE_D_DIR/run.log"
TEB="$BENCH/.venv/bin/tool-eval-bench"

mkdir -p "$RESULTS"

# Sanity
[[ -x "$TEB" ]] || { echo "tool-eval-bench venv not ready at $TEB"; exit 1; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "missing OPENROUTER_API_KEY"; exit 1; }

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# Each row: id|provider|base_url|api_key_var|model|notes
ENDPOINTS=(
  # OUR LOCAL ENDPOINT - whatever is currently running
  "D1-local|local|http://localhost:8000|-|Qwen3.6-27B|current vLLM endpoint"
  # ANTHROPIC via OpenRouter
  "D3-claude-sonnet-4.6|openrouter|https://openrouter.ai/api|OPENROUTER_API_KEY|anthropic/claude-sonnet-4.6|Anthropic flagship"
  "D4-claude-haiku-4.5|openrouter|https://openrouter.ai/api|OPENROUTER_API_KEY|anthropic/claude-haiku-4.5|Anthropic cheap-tier"
  # OPENAI via OpenRouter (avoids gpt-5 max_completion_tokens issue)
  "D5-gpt-5.5|openrouter|https://openrouter.ai/api|OPENROUTER_API_KEY|openai/gpt-5.5|OpenAI flagship"
  "D6-gpt-5-mini|openrouter|https://openrouter.ai/api|OPENROUTER_API_KEY|openai/gpt-5-mini|OpenAI cheap-tier"
  "D7-gpt-5-nano|openrouter|https://openrouter.ai/api|OPENROUTER_API_KEY|openai/gpt-5-nano|OpenAI ultra-cheap baseline"
  # GOOGLE via OpenRouter
  "D8-gemini-3.5-flash|openrouter|https://openrouter.ai/api|OPENROUTER_API_KEY|google/gemini-3.5-flash|Google flash-tier"
  # CEREBRAS direct - apex Qwen for "best possible Qwen" comparison
  "D9-qwen-235b-cerebras|cerebras|https://api.cerebras.ai|CEREBRAS_API_KEY|qwen-3-235b-a22b-instruct-2507|Best Qwen at apex hardware"
)

# Scenario selection: --short gives 15 well-distributed scenarios for the budget
SCENARIOS_ARG=(--short)
COMMON_ARGS=(
  --no-warmup
  --no-live
  --no-probe-engine
  --skip-coherence  # don't need coherence on remote APIs - we already have it from sweep
  --timeout 90
  --seed 42
  --temperature 0.2
  --trials 1
  --json  # silence Rich output, use JSON to stdout pipeline
)

log "=== Phase D begin ==="
log "Endpoints to run: ${#ENDPOINTS[@]}"
log "Scenarios: --short (~15)"
log "Cost cap: ~\$5 (estimated actual: ~\$2)"
log "Budget: 60 min wall clock"

START_T=$(date +%s)
SUCCESS=()
FAILED=()
SKIPPED=()

for entry in "${ENDPOINTS[@]}"; do
  IFS='|' read -r id provider base_url key_var model notes <<<"$entry"
  ELAPSED=$(( $(date +%s) - START_T ))
  if [[ $ELAPSED -gt 3600 ]]; then
    log "!! 60-min budget reached - stopping. Remaining: $id and beyond"
    break
  fi

  out_dir="$RESULTS/$id"
  if [[ -f "$out_dir/teb-results.json" ]] || [[ -f "$out_dir/results.json" ]]; then
    log "SKIP  $id (already done)"
    SKIPPED+=("$id")
    continue
  fi
  mkdir -p "$out_dir"

  log "RUN   $id  ($provider, $model) — $notes"

  # Build invocation
  args=(
    "$TEB"
    --base-url "$base_url"
    --model "$model"
    "${SCENARIOS_ARG[@]}"
    "${COMMON_ARGS[@]}"
    --output-dir "$out_dir"
    --json-file "$out_dir/teb-results.json"
  )
  if [[ "$key_var" != "-" ]]; then
    key_val="${!key_var:-}"
    if [[ -z "$key_val" ]]; then
      log "FAIL  $id (missing $key_var)"
      FAILED+=("$id:missing-key")
      continue
    fi
    args+=( --api-key "$key_val" )
  fi

  t0=$(date +%s)
  if timeout 600 "${args[@]}" > "$out_dir/console.log" 2>&1; then
    t1=$(date +%s)
    dur=$((t1-t0))
    log "DONE  $id (${dur}s)"
    SUCCESS+=("$id")
  else
    rc=$?
    log "FAIL  $id (rc=$rc, see $out_dir/console.log)"
    FAILED+=("$id:rc=$rc")
  fi
done

ELAPSED=$(( $(date +%s) - START_T ))
log "=== Phase D done in ${ELAPSED}s ==="
log "  Success: ${#SUCCESS[@]} -> ${SUCCESS[*]:-<none>}"
log "  Failed:  ${#FAILED[@]}  -> ${FAILED[*]:-<none>}"
log "  Skipped: ${#SKIPPED[@]} -> ${SKIPPED[*]:-<none>}"

# Aggregate CSV
log "=== Building aggregate CSV ==="
python3 "$PHASE_D_DIR/aggregate.py" "$RESULTS" > "$PHASE_D_DIR/leaderboard.csv" 2>>"$LOG" \
  && log "CSV: $PHASE_D_DIR/leaderboard.csv" \
  || log "aggregate.py failed (CSV not built)"

log "=== Phase D complete ==="
