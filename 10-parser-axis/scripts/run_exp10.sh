#!/usr/bin/env bash
# Exp 10 orchestrator — 3-stage tournament + 7 frontier yardsticks
#
# Stage 1: parser axis on v13+FP8+MTP=3 (xml vs coder) → winner P*
# Stage 2: image axis on BF16+DFlash with P* (v13 vs nightly)
# Stage 3: image axis on FP8+MTP=3 with P* (v13 from Stage 1 already; only nightly needed)
#
# Frontier yardsticks run in parallel (API-bound, no GPU contention).
#
# Idempotent: skips cells that already have teb-results.json.
# Logs everything to logs/exp10-driver.log.
#
# Author: Sisyphus, 2026-05-27

set -uo pipefail

EXP_DIR="/home/josh/qwen-vllm-test/sweeps/10-parser-axis"
SCRIPTS="$EXP_DIR/scripts"
RESULTS="$EXP_DIR/results"
LOGS="$EXP_DIR/logs"
LAUNCH="$SCRIPTS/launch_cell.sh"
TEB="/home/josh/qwen-vllm-test/sweeps/v13-kitchen-sink-bf16dflash-and-fp8mtp3/phase-D-tooleval/tool-eval-bench/.venv/bin/tool-eval-bench"
DRIVER_LOG="$LOGS/exp10-driver.log"

mkdir -p "$RESULTS"/{stage1,stage2,stage3,frontier} "$LOGS"

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$DRIVER_LOG"
}

# ── Pre-flight ──────────────────────────────────────────────────────────────
[[ -x "$TEB" ]] || { echo "tool-eval-bench not found: $TEB"; exit 1; }
[[ -x "$LAUNCH" ]] || { echo "launch_cell.sh not executable: $LAUNCH"; exit 1; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "missing OPENROUTER_API_KEY for frontier"; exit 1; }
[[ -n "${CEREBRAS_API_KEY:-}" ]] || { echo "missing CEREBRAS_API_KEY for frontier"; exit 1; }

START_T=$(date +%s)
log "===================== Exp 10 begin ====================="

# ── Common bench args ───────────────────────────────────────────────────────
BENCH_COMMON_ARGS=(
  --no-warmup
  --no-live
  --no-probe-engine
  --skip-coherence
  --timeout 120
  --seed 42
  --temperature 0.0
  --trials 1
  --json
)
# 69-scenario full suite = default (no --short, no --scenarios filter)

# ── Helpers ─────────────────────────────────────────────────────────────────
run_local_cell() {
  # $1=stage_dir $2=id $3=image $4=quant $5=spec_method $6=spec_tokens $7=tool_parser $8=model
  local stage_dir="$1" id="$2" image="$3" quant="$4" sm="$5" st="$6" tp="$7" model="$8"
  local out_dir="$RESULTS/$stage_dir/$id"
  if [[ -f "$out_dir/teb-results.json" ]]; then
    log "SKIP  $id (already done)"
    return 0
  fi
  mkdir -p "$out_dir"

  log "LAUNCH $id (image=$image quant=$quant spec=$sm/N$st parser=$tp)"
  if ! "$LAUNCH" "$id" "$image" "$quant" "$sm" "$st" "$tp" qwen3 "$model" \
      >> "$LOGS/launch-$id.log" 2>&1; then
    log "FAIL  $id container boot failed; see $LOGS/launch-$id.log"
    return 1
  fi

  log "BENCH $id (69 scenarios, ~3-4 min)"
  local t0 t1 dur
  t0=$(date +%s)
  if timeout 1800 "$TEB" \
      --base-url http://localhost:8000 \
      --model Qwen3.6-27B \
      --backend vllm \
      "${BENCH_COMMON_ARGS[@]}" \
      --output-dir "$out_dir" \
      --json-file "$out_dir/teb-results.json" \
      > "$out_dir/console.log" 2>&1; then
    t1=$(date +%s); dur=$((t1-t0))
    local score
    score=$(python3 -c "import json; d=json.load(open('$out_dir/teb-results.json')); print(d.get('final_score','?'))" 2>/dev/null || echo "?")
    log "DONE  $id (${dur}s, final_score=$score)"
    return 0
  else
    local rc=$?
    log "FAIL  $id bench rc=$rc; see $out_dir/console.log"
    return 1
  fi
}

run_frontier_cell() {
  # $1=id $2=provider $3=base_url $4=key_var $5=model
  local id="$1" provider="$2" base_url="$3" key_var="$4" model="$5"
  local out_dir="$RESULTS/frontier/$id"
  if [[ -f "$out_dir/teb-results.json" ]]; then
    log "SKIP  $id (already done)"
    return 0
  fi
  mkdir -p "$out_dir"
  local key_val="${!key_var:-}"
  [[ -z "$key_val" ]] && { log "FAIL  $id (missing $key_var)"; return 1; }

  log "BENCH $id ($provider, $model, 69 scenarios)"
  local t0 t1 dur
  t0=$(date +%s)
  if timeout 1800 "$TEB" \
      --base-url "$base_url" \
      --model "$model" \
      --backend vllm \
      --api-key "$key_val" \
      "${BENCH_COMMON_ARGS[@]}" \
      --output-dir "$out_dir" \
      --json-file "$out_dir/teb-results.json" \
      > "$out_dir/console.log" 2>&1; then
    t1=$(date +%s); dur=$((t1-t0))
    local score
    score=$(python3 -c "import json; d=json.load(open('$out_dir/teb-results.json')); print(d.get('final_score','?'))" 2>/dev/null || echo "?")
    log "DONE  $id (${dur}s, final_score=$score)"
    return 0
  else
    local rc=$?
    log "FAIL  $id rc=$rc; see $out_dir/console.log"
    return 1
  fi
}

pick_winner() {
  # $1=path_a $2=path_b $3=label_a $4=label_b → echoes "xml" or "coder" (or whatever parsers were)
  local path_a="$1" path_b="$2" label_a="$3" label_b="$4"
  python3 - "$path_a" "$path_b" "$label_a" "$label_b" <<'EOF'
import json, sys
pa, pb, la, lb = sys.argv[1:5]
da = json.load(open(f'{pa}/teb-results.json'))
db = json.load(open(f'{pb}/teb-results.json'))
sa, sb = da['final_score'], db['final_score']
ra = da['scores'].get('responsiveness', 0)
rb = db['scores'].get('responsiveness', 0)
mta = da['scores'].get('median_turn_ms', 99999)
mtb = db['scores'].get('median_turn_ms', 99999)
# Higher final_score wins; tiebreak responsiveness (higher); then median_turn_ms (lower)
if sa != sb:
    winner = la if sa > sb else lb
elif ra != rb:
    winner = la if ra > rb else lb
else:
    winner = la if mta < mtb else lb
print(winner)
EOF
}

pick_winner_efficiency() {
  # For Stage 2 BF16 — "more efficient" = throughput proxy. Use responsiveness (which is computed from latency).
  # Tiebreak: median_turn_ms (lower better); then final_score.
  local path_a="$1" path_b="$2" label_a="$3" label_b="$4"
  python3 - "$path_a" "$path_b" "$label_a" "$label_b" <<'EOF'
import json, sys
pa, pb, la, lb = sys.argv[1:5]
da = json.load(open(f'{pa}/teb-results.json'))
db = json.load(open(f'{pb}/teb-results.json'))
ra = da['scores'].get('responsiveness', 0)
rb = db['scores'].get('responsiveness', 0)
mta = da['scores'].get('median_turn_ms', 99999)
mtb = db['scores'].get('median_turn_ms', 99999)
sa, sb = da['final_score'], db['final_score']
# Efficiency: higher responsiveness wins; tiebreak lower median_turn_ms; final_score last.
if ra != rb:
    winner = la if ra > rb else lb
elif mta != mtb:
    winner = la if mta < mtb else lb
else:
    winner = la if sa > sb else lb
print(winner)
EOF
}

# ── Frontier yardsticks (LAUNCH IN BACKGROUND — API-bound, runs while GPU work proceeds) ──
log "──── Launching 7 frontier yardsticks in background ────"
(
  run_frontier_cell "Y1-claude-sonnet-4.6" openrouter "https://openrouter.ai/api" OPENROUTER_API_KEY "anthropic/claude-sonnet-4.6"
  run_frontier_cell "Y2-claude-haiku-4.5"  openrouter "https://openrouter.ai/api" OPENROUTER_API_KEY "anthropic/claude-haiku-4.5"
  run_frontier_cell "Y3-gpt-5.5"           openrouter "https://openrouter.ai/api" OPENROUTER_API_KEY "openai/gpt-5.5"
  run_frontier_cell "Y4-gpt-5-mini"        openrouter "https://openrouter.ai/api" OPENROUTER_API_KEY "openai/gpt-5-mini"
  run_frontier_cell "Y5-gpt-5-nano"        openrouter "https://openrouter.ai/api" OPENROUTER_API_KEY "openai/gpt-5-nano"
  run_frontier_cell "Y6-gemini-3.5-flash"  openrouter "https://openrouter.ai/api" OPENROUTER_API_KEY "google/gemini-3.5-flash"
  run_frontier_cell "Y7-qwen-235b-cerebras" cerebras "https://api.cerebras.ai" CEREBRAS_API_KEY "qwen-3-235b-a22b-instruct-2507"
  log "──── Frontier yardsticks complete ────"
  touch "$LOGS/frontier.done"
) > "$LOGS/frontier.log" 2>&1 &
FRONTIER_PID=$!
log "Frontier background PID: $FRONTIER_PID"

# ── Stage 1: v13 + FP8 + MTP=3, xml vs coder ─────────────────────────────────
log ""
log "════════════════════════ STAGE 1: parser axis (v13+FP8+MTP=3) ════════════════════════"
run_local_cell stage1 S1-v13-fp8-xml   repne/vllm:v13 fp8 mtp 3 qwen3_xml   Qwen/Qwen3.6-27B-FP8
run_local_cell stage1 S1-v13-fp8-coder repne/vllm:v13 fp8 mtp 3 qwen3_coder Qwen/Qwen3.6-27B-FP8

P_STAR=$(pick_winner \
  "$RESULTS/stage1/S1-v13-fp8-xml" \
  "$RESULTS/stage1/S1-v13-fp8-coder" \
  "qwen3_xml" "qwen3_coder")
log ""
log "★★★ STAGE 1 WINNER: $P_STAR ★★★"
echo "$P_STAR" > "$RESULTS/stage1/winner.txt"

# ── Stage 2: BF16+DFlash, v13 vs nightly with P* ─────────────────────────────
log ""
log "════════════════════════ STAGE 2: image axis (BF16+DFlash, parser=$P_STAR) ════════════════════════"
run_local_cell stage2 "S2-v13-bf16-${P_STAR}"     repne/vllm:v13           bf16 dflash 8 "$P_STAR" Qwen/Qwen3.6-27B
run_local_cell stage2 "S2-nightly-bf16-${P_STAR}" vllm/vllm-openai:nightly bf16 dflash 8 "$P_STAR" Qwen/Qwen3.6-27B

BF16_IMAGE_WINNER=$(pick_winner_efficiency \
  "$RESULTS/stage2/S2-v13-bf16-${P_STAR}" \
  "$RESULTS/stage2/S2-nightly-bf16-${P_STAR}" \
  "repne/vllm:v13" "vllm/vllm-openai:nightly")
log ""
log "★★★ STAGE 2 BF16 IMAGE WINNER (by efficiency): $BF16_IMAGE_WINNER ★★★"
echo "$BF16_IMAGE_WINNER" > "$RESULTS/stage2/winner.txt"

# ── Stage 3: FP8+MTP=3 on nightly with P* (v13 already done in Stage 1) ──────
log ""
log "════════════════════════ STAGE 3: FP8 image generalization (nightly+FP8+MTP=3+$P_STAR) ════════════════════════"
run_local_cell stage3 "S3-nightly-fp8-${P_STAR}" vllm/vllm-openai:nightly fp8 mtp 3 "$P_STAR" Qwen/Qwen3.6-27B-FP8

# Compare nightly-fp8 vs v13-fp8 (S1 winner with P*)
S1_WINNER_PATH="$RESULTS/stage1/S1-v13-fp8-${P_STAR/qwen3_/}"
# the S1 ids are S1-v13-fp8-xml or S1-v13-fp8-coder; P_STAR is qwen3_xml or qwen3_coder
S1_WINNER_PATH="$RESULTS/stage1/S1-v13-fp8-${P_STAR#qwen3_}"

FP8_IMAGE_WINNER=$(pick_winner_efficiency \
  "$S1_WINNER_PATH" \
  "$RESULTS/stage3/S3-nightly-fp8-${P_STAR}" \
  "repne/vllm:v13" "vllm/vllm-openai:nightly")
log ""
log "★★★ STAGE 3 FP8 IMAGE WINNER (by efficiency): $FP8_IMAGE_WINNER ★★★"
echo "$FP8_IMAGE_WINNER" > "$RESULTS/stage3/winner.txt"

# ── Wait for frontier yardsticks ────────────────────────────────────────────
log ""
log "════════════════════════ WAITING FOR FRONTIER YARDSTICKS ════════════════════════"
if [[ -f "$LOGS/frontier.done" ]]; then
  log "Frontier already complete"
else
  log "Waiting on frontier PID $FRONTIER_PID..."
  wait "$FRONTIER_PID" || log "frontier exited with non-zero rc"
fi

# ── Final summary ───────────────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - START_T ))
log ""
log "═════════════════════════ Exp 10 done in ${ELAPSED}s ($(echo "scale=1; $ELAPSED/60" | bc) min) ═════════════════════════"
log "  Stage 1 winner: $P_STAR"
log "  Stage 2 BF16 image winner: $BF16_IMAGE_WINNER"
log "  Stage 3 FP8 image winner: $FP8_IMAGE_WINNER"
log "  Frontier yardsticks: $(ls "$RESULTS/frontier" 2>/dev/null | wc -l) cells"
log "  Results: $RESULTS"
log "  Logs:    $LOGS"
