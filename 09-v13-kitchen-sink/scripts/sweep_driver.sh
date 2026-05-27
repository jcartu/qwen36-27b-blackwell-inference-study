#!/usr/bin/env bash
# sweep_driver.sh - Master driver for v13 kitchen-sink dual-config sweep.
# Designed for unattended overnight execution. Idempotent. Resumable.
set -uo pipefail   # NOTE: NOT -e — we want to continue past individual cell failures

SWEEP=/home/josh/qwen-vllm-test/sweeps/v13-kitchen-sink-bf16dflash-and-fp8mtp3
SCRIPTS=$SWEEP/scripts
LOGS=$SWEEP/logs
HARNESS_V048=$SCRIPTS/harness-v0.4.8/llm_decode_bench.py
HARNESS_UP=$SCRIPTS/harness-upstream/llm_decode_bench.py
HARNESS_UP_CJK=$SCRIPTS/harness-upstream/llm_cjk_watchdog.py
LAUNCHER=/home/josh/vllm-services/launch-qwen36-27b-tp2-sota.sh

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGS/driver.log"
}

abort() {
  log "FATAL: $*"
  exit 1
}

# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------
check_display_gpu() {
  # Display GPU UUID prefix: 538bf008. If it has >4000 MiB used we have a problem.
  local mem
  mem=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader | awk -F, '/538bf008/{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -z "$mem" ]; then mem=0; fi
  if [ "$mem" -gt 4000 ]; then
    log "DISPLAY GPU CONTAMINATED: ${mem} MiB — halting sweep"
    return 1
  fi
  return 0
}

wait_for_server() {
  local label="$1" max_wait="${2:-420}"
  local elapsed=0
  log "Waiting for server ready ($label, timeout ${max_wait}s)..."
  while [ $elapsed -lt $max_wait ]; do
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
      log "  server ready at T+${elapsed}s"
      check_display_gpu || return 1
      return 0
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  log "  TIMEOUT: server not ready after ${max_wait}s"
  return 1
}

container_status() {
  docker inspect -f '{{.State.Status}}' vllm-qwen36-27b-tp2 2>/dev/null || echo "missing"
}

restart_container_if_needed() {
  local status
  status=$(container_status)
  if [ "$status" != "running" ]; then
    log "Container status=$status — relaunching"
    bash "$LAUNCHER" > "$LOGS/relaunch_$(date +%s).log" 2>&1
    sleep 5
  fi
  wait_for_server "post-restart-check"
}

# ---------------------------------------------------------------------------
# Cell runner with retry + restart
# ---------------------------------------------------------------------------
run_one() {
  # run_one <harness> <out_dir> <cell_label> <conc> <ctx> [extra args...]
  local harness="$1" out_dir="$2" cell="$3" conc="$4" ctx="$5"
  shift 5
  local extra=("$@")
  local attempts=0
  while [ $attempts -lt 2 ]; do
    attempts=$((attempts + 1))
    if "$SCRIPTS/run_cell.sh" "$harness" "$out_dir" "$cell" "$conc" "$ctx" "${extra[@]}" 2>&1 | tee -a "$LOGS/driver.log"; then
      return 0
    fi
    log "  cell $cell FAILED (attempt $attempts) — checking container"
    if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
      log "  server unreachable — waiting for auto-restart"
      restart_container_if_needed || { log "  container relaunch failed"; return 1; }
    fi
    sleep 10
  done
  log "  cell $cell FAILED PERMANENTLY"
  echo "$cell" >> "$out_dir/_permanent_failures.log"
  return 1
}

# ---------------------------------------------------------------------------
# Phase A1: Methodology validation
# Anchor cells: c=1 ctx=0, c=8 ctx=0, c=1 ctx=32k, c=8 ctx=32k
# Two harness versions → 8 runs
# ---------------------------------------------------------------------------
phase_A1() {
  log "=== PHASE A1: BF16+DFlash methodology validation ==="
  local OUT_V048=$SWEEP/A-bf16-dflash/phase1-method-validation/v0.4.8
  local OUT_UP=$SWEEP/A-bf16-dflash/phase1-method-validation/upstream-main
  mkdir -p "$OUT_V048" "$OUT_UP"

  for ctx in 0 32k; do
    for conc in 1 8; do
      local cell="c${conc}-ctx${ctx}"
      run_one "$HARNESS_V048" "$OUT_V048" "$cell" "$conc" "$ctx"
      run_one "$HARNESS_UP"   "$OUT_UP"   "$cell" "$conc" "$ctx"
    done
  done
  log "Phase A1 complete"
}

# ---------------------------------------------------------------------------
# Phase A2: P2P fabric diagnostic
# ---------------------------------------------------------------------------
phase_A2() {
  log "=== PHASE A2: P2P fabric diagnostic ==="
  local OUT=$SWEEP/A-bf16-dflash/phase2-p2p
  mkdir -p "$OUT"
  if [ -s "$OUT/p2p_fabric.json" ]; then
    log "  [SKIP] p2p_fabric.json already exists"
    return 0
  fi
  python3 "$HARNESS_UP" \
    --p2pmark-only \
    --p2pmark-mode all \
    --p2pmark-max-gpus 2 \
    --display-mode plain \
    --output "$OUT/p2p_fabric.json" > "$OUT/p2p_fabric.log" 2>&1 || {
      log "  p2p diag exited non-zero (may be OK if no p2pmark binary)"
    }
  log "Phase A2 complete"
}

# ---------------------------------------------------------------------------
# Phase A3: Full matrix N=5 reps
# Concurrency: 1, 2, 4, 8, 16, 32
# Contexts: 0, 16k, 32k, 64k, 131k
# Use v0.4.8 (the local canonical) for all 5 reps
# ---------------------------------------------------------------------------
phase_A3() {
  log "=== PHASE A3: BF16+DFlash full matrix N=5 ==="
  local concurrencies="1 2 4 8 16 32"
  local contexts="0 16k 32k 64k 131k"
  for rep in 1 2 3 4 5; do
    local OUT=$SWEEP/A-bf16-dflash/phase3-matrix/run$rep
    mkdir -p "$OUT"
    log "--- Rep $rep ---"
    for ctx in $contexts; do
      for conc in $concurrencies; do
        local cell="c${conc}-ctx${ctx}"
        run_one "$HARNESS_V048" "$OUT" "$cell" "$conc" "$ctx"
        check_display_gpu || abort "display GPU contamination during A3"
      done
    done
  done
  log "Phase A3 complete"
}

# ---------------------------------------------------------------------------
# Phase A4: Quality profiles
# ---------------------------------------------------------------------------
phase_A4() {
  log "=== PHASE A4: BF16+DFlash quality profiles ==="
  local OUT=$SWEEP/A-bf16-dflash/phase4-quality
  mkdir -p "$OUT"

  for profile in lavd hotel-lights; do
    local jf="$OUT/${profile}.json"
    if [ -s "$jf" ]; then
      log "  [SKIP] $profile"
      continue
    fi
    log "  running profile=$profile"
    python3 "$HARNESS_UP" \
      --port 8000 --model Qwen3.6-27B \
      --test-profile "$profile" \
      --completion-stats \
      --display-mode plain \
      --no-hw-monitor \
      --output "$jf" > "${jf%.json}.log" 2>&1 || {
        log "  profile $profile FAILED"
      }
  done
  log "Phase A4 complete"
}

# ---------------------------------------------------------------------------
# Phase A5: CJK watchdog (background, parallel)
# Started independently — see start_cjk_watchdog below
# ---------------------------------------------------------------------------
start_cjk_watchdog() {
  local config="$1"  # bf16 or fp8
  local OUT=$SWEEP/${config}-phase5-cjk
  mkdir -p "$OUT"
  if pgrep -f "llm_cjk_watchdog.*$config" > /dev/null; then
    log "  CJK watchdog already running for $config"
    return 0
  fi
  log "Starting CJK watchdog ($config)..."
  nohup python3 "$HARNESS_UP_CJK" \
    --port 8000 \
    --model Qwen3.6-27B \
    --context-tokens 65536 \
    --max-tokens 800 \
    --no-overlay \
    -L \
    > "$OUT/cjk-loop.log" 2>&1 &
  echo $! > "$OUT/cjk.pid"
  log "  CJK watchdog pid=$(cat $OUT/cjk.pid)"
}

stop_cjk_watchdog() {
  local config="$1"
  local OUT=$SWEEP/${config}-phase5-cjk
  if [ -f "$OUT/cjk.pid" ]; then
    local pid=$(cat "$OUT/cjk.pid")
    if kill "$pid" 2>/dev/null; then
      log "Stopped CJK watchdog (pid=$pid)"
    fi
    rm -f "$OUT/cjk.pid"
  fi
  pkill -f "llm_cjk_watchdog" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# A→B handoff: swap launcher to FP8+MTP=3 and relaunch
# ---------------------------------------------------------------------------
handoff_to_fp8() {
  log "=== A→B HANDOFF: swapping launcher to FP8+MTP=3 ==="
  local LAUNCHER_BAK="${LAUNCHER}.bak.bf16dflash-$(date +%Y%m%dT%H%M%S)"
  cp "$LAUNCHER" "$LAUNCHER_BAK"
  log "  backed up bf16+dflash launcher to $LAUNCHER_BAK"

  # Build FP8 launcher: change model + drop DFlash 5-line block, replace with MTP=3 2-line block
  python3 << 'PYEOF'
import re
from pathlib import Path
p = Path("/home/josh/vllm-services/launch-qwen36-27b-tp2-sota.sh")
src = p.read_text()

# 1. Swap model: BF16 -> FP8 (only on --model line)
src = re.sub(
    r"(--model Qwen/Qwen3\.6-27B)(?!-FP8)(\b)",
    r"\1-FP8\2",
    src,
    count=1,
)

# 2. Rewrite speculative-config block: DFlash (5 dotted lines) -> MTP=3 (2 dotted lines)
lines = src.split("\n")
out = []
i = 0
spec_done = False
while i < len(lines):
    L = lines[i]
    if "--speculative-config.method dflash" in L and not spec_done:
        indent = L[:len(L) - len(L.lstrip())]
        out.append(f"{indent}--speculative-config.method mtp \\")
        out.append(f"{indent}--speculative-config.num_speculative_tokens 3 \\")
        # Skip the original DFlash block: this line + all subsequent --speculative-config.* lines
        j = i + 1
        while j < len(lines) and "--speculative-config." in lines[j]:
            j += 1
        i = j
        spec_done = True
        continue
    out.append(L)
    i += 1

if not spec_done:
    raise SystemExit("FATAL: could not find --speculative-config.method dflash in launcher")

p.write_text("\n".join(out))
print("Launcher rewritten for FP8+MTP=3")
PYEOF

  if [ $? -ne 0 ]; then
    log "FATAL: launcher rewrite failed"
    return 1
  fi
  log "  launcher rewritten - new model/spec lines:"
  grep -nE 'model |speculative-config\.' "$LAUNCHER" | tee -a "$LOGS/driver.log"

  # Stop existing container
  docker stop vllm-qwen36-27b-tp2 2>&1 | tee -a "$LOGS/driver.log"
  docker rm vllm-qwen36-27b-tp2 2>&1 | tee -a "$LOGS/driver.log" || true

  # Launch FP8 config
  bash "$LAUNCHER" > "$LOGS/fp8_launch.log" 2>&1
  wait_for_server "FP8+MTP=3 startup" 600 || {
    log "FATAL: FP8+MTP=3 failed to start — engaging fallback"
    return 1
  }
  log "FP8+MTP=3 launched successfully"
}

handoff_fallback_v12_fp8() {
  log "=== FALLBACK: FP8+MTP=3 on v13 failed — trying v12 ==="
  python3 << 'PYEOF'
from pathlib import Path
p = Path("/home/josh/vllm-services/launch-qwen36-27b-tp2-sota.sh")
src = p.read_text()
src = src.replace("repne/vllm:v13", "repne/vllm:v12")
p.write_text(src)
print("Reverted image to repne/vllm:v12")
PYEOF
  docker stop vllm-qwen36-27b-tp2 2>&1 | tee -a "$LOGS/driver.log" || true
  docker rm vllm-qwen36-27b-tp2 2>&1 | tee -a "$LOGS/driver.log" || true
  bash "$LAUNCHER" > "$LOGS/fp8_v12_launch.log" 2>&1
  wait_for_server "FP8 v12 fallback" 600 || abort "FP8 fallback to v12 also failed"
  log "Fallback to v12 FP8+MTP=3 succeeded"
  echo "FP8_FALLBACK_BUILD=v12" >> "$SWEEP/B-fp8-mtp3/_fallback.flag"
}

# ---------------------------------------------------------------------------
# Phase B1/B3/B4 = mirror of A1/A3/A4 against FP8 endpoint
# ---------------------------------------------------------------------------
phase_B1() {
  log "=== PHASE B1: FP8+MTP=3 methodology validation ==="
  local OUT_V048=$SWEEP/B-fp8-mtp3/phase1-method-validation/v0.4.8
  local OUT_UP=$SWEEP/B-fp8-mtp3/phase1-method-validation/upstream-main
  mkdir -p "$OUT_V048" "$OUT_UP"
  for ctx in 0 32k; do
    for conc in 1 8; do
      local cell="c${conc}-ctx${ctx}"
      run_one "$HARNESS_V048" "$OUT_V048" "$cell" "$conc" "$ctx"
      run_one "$HARNESS_UP"   "$OUT_UP"   "$cell" "$conc" "$ctx"
    done
  done
  log "Phase B1 complete"
}

phase_B3() {
  log "=== PHASE B3: FP8+MTP=3 full matrix N=5 ==="
  local concurrencies="1 2 4 8 16 32"
  local contexts="0 16k 32k 64k 131k"
  for rep in 1 2 3 4 5; do
    local OUT=$SWEEP/B-fp8-mtp3/phase3-matrix/run$rep
    mkdir -p "$OUT"
    log "--- Rep $rep ---"
    for ctx in $contexts; do
      for conc in $concurrencies; do
        local cell="c${conc}-ctx${ctx}"
        run_one "$HARNESS_V048" "$OUT" "$cell" "$conc" "$ctx"
        check_display_gpu || abort "display GPU contamination during B3"
      done
    done
  done
  log "Phase B3 complete"
}

phase_B4() {
  log "=== PHASE B4: FP8+MTP=3 quality profiles ==="
  local OUT=$SWEEP/B-fp8-mtp3/phase4-quality
  mkdir -p "$OUT"
  for profile in lavd hotel-lights; do
    local jf="$OUT/${profile}.json"
    if [ -s "$jf" ]; then log "  [SKIP] $profile"; continue; fi
    python3 "$HARNESS_UP" \
      --port 8000 --model Qwen3.6-27B \
      --test-profile "$profile" \
      --completion-stats \
      --display-mode plain \
      --no-hw-monitor \
      --output "$jf" > "${jf%.json}.log" 2>&1 || log "  profile $profile FAILED"
  done
  log "Phase B4 complete"
}

# ---------------------------------------------------------------------------
# Phase C: CSV synthesis (per-config + master)
# ---------------------------------------------------------------------------
phase_C_csv() {
  log "=== PHASE C: CSV synthesis ==="
  python3 "$SCRIPTS/extract_csv.py" \
    "$SWEEP/A-bf16-dflash/results-A.csv" \
    "$SWEEP/A-bf16-dflash" 2>&1 | tee -a "$LOGS/driver.log"
  python3 "$SCRIPTS/extract_csv.py" \
    "$SWEEP/B-fp8-mtp3/results-B.csv" \
    "$SWEEP/B-fp8-mtp3" 2>&1 | tee -a "$LOGS/driver.log"
  python3 "$SCRIPTS/extract_csv.py" \
    "$SWEEP/experiment-09-master-results.csv" \
    "$SWEEP/A-bf16-dflash" "$SWEEP/B-fp8-mtp3" 2>&1 | tee -a "$LOGS/driver.log"
  log "CSVs written"
}

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
main() {
  log "===================================================="
  log "v13 KITCHEN-SINK DUAL-CONFIG SWEEP — DRIVER START"
  log "PID: $$"
  log "===================================================="

  check_display_gpu || abort "Display GPU already contaminated at start"

  # ===== PASS A: BF16+DFlash N=8 =====
  start_cjk_watchdog bf16
  phase_A1
  phase_A2
  phase_A3
  phase_A4
  stop_cjk_watchdog bf16

  # ===== A→B HANDOFF =====
  if ! handoff_to_fp8; then
    handoff_fallback_v12_fp8
  fi

  # ===== PASS B: FP8+MTP=3 =====
  start_cjk_watchdog fp8
  phase_B1
  phase_B3
  phase_B4
  stop_cjk_watchdog fp8

  # ===== PHASE C: CSV synthesis =====
  phase_C_csv

  log "===================================================="
  log "DRIVER COMPLETE"
  log "===================================================="
}

main "$@"
