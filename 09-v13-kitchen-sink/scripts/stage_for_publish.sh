#!/usr/bin/env bash
# stage_for_publish.sh - copy sweep artifacts into the GitHub repo's Exp 09 layout
#
# Layout convention (mirrors Exp 08):
#   09-v13-kitchen-sink/
#     A-bf16-dflash/
#       phase1-method-validation/{v0.4.8,upstream-main}/c{N}_ctx{T}.{json,log}
#       phase3-matrix/runs/c{N}_ctx{T}_run{R}/{results.json,bench.log}
#       phase4-quality/{lavd,hotel-lights}.{json,log}
#       _reports/{REPORT.md,A-summary.md,A-summary.json,A-results-rich.csv,A-results-master.csv}
#     B-fp8-mtp3/  (same layout as A)
#     phase-D-tooleval/
#       run_phase_d.sh, aggregate.py, results/D{N}-*/, leaderboard.csv
#     scripts/  (sweep_driver.sh, extract_csv.py, to_master_schema.py, analyze.py, harness-*/)
#     README.md, PLAN.md, exp09-results.csv (combined master schema)

set -euo pipefail
SWEEP="${SWEEP:-/home/josh/qwen-vllm-test/sweeps/v13-kitchen-sink-bf16dflash-and-fp8mtp3}"
EXP="${EXP:-/home/josh/qwen-vllm-test/_publish-staging/qwen36-27b-blackwell-inference-study/09-v13-kitchen-sink}"

log() { printf '[stage] %s\n' "$*"; }

# Pass A and B identical sub-trees
stage_pass() {
  local letter=$1   # A or B
  local cfg=$2      # bf16-dflash or fp8-mtp3
  local src="$SWEEP/$letter-$cfg"
  local dst="$EXP/$letter-$cfg"
  [[ -d "$src" ]] || { log "skip Pass $letter (no source dir)"; return; }

  # phase1-method-validation
  for harness in v0.4.8 upstream-main; do
    sd="$src/phase1-method-validation/$harness"
    [[ -d "$sd" ]] || continue
    dd="$dst/phase1-method-validation/$harness"
    mkdir -p "$dd"
    # Rename c1-ctx0 -> c1_ctx0 (dash to underscore) per repo convention
    for f in "$sd"/c*-ctx*.json "$sd"/c*-ctx*.log; do
      [[ -e "$f" ]] || continue
      name=$(basename "$f")
      # c1-ctx32k -> c1_ctx32k (keep nominal label, not bench-reported)
      newname="${name//-/_}"
      cp -n "$f" "$dd/$newname"
    done
    log "Pass $letter phase1 $harness: $(ls $dd/*.json 2>/dev/null | wc -l) json"
  done

  # phase3-matrix as runs/c{N}_ctx{T}_run{R}/
  for repdir in "$src/phase3-matrix"/run*; do
    [[ -d "$repdir" ]] || continue
    rep=$(basename "$repdir")  # run1..run5
    rn="${rep#run}"
    for j in "$repdir"/*.json; do
      [[ -e "$j" ]] || continue
      # c1-ctx131k.json -> need actual ctx (134144) from JSON, but use canonical 131072 for filename
      cell=$(basename "$j" .json)
      # Normalize: c1-ctx0 -> c1_ctx0, c1-ctx16k -> c1_ctx16384, ...
      conc=$(echo "$cell" | sed 's/c\([0-9]*\)-ctx.*/\1/')
      ctx_label=$(echo "$cell" | sed 's/c[0-9]*-ctx//')
      case "$ctx_label" in
        0) ctx=0 ;;
        16k) ctx=16384 ;;
        32k) ctx=32768 ;;
        64k) ctx=65536 ;;
        131k) ctx=131072 ;;
        *) ctx="$ctx_label" ;;
      esac
      dd="$dst/phase3-matrix/runs/c${conc}_ctx${ctx}_run${rn}"
      mkdir -p "$dd"
      cp -n "$j" "$dd/results.json"
      [[ -f "${j%.json}.log" ]] && cp -n "${j%.json}.log" "$dd/bench.log"
    done
  done
  log "Pass $letter phase3 matrix: $(find $dst/phase3-matrix/runs -name results.json 2>/dev/null | wc -l) cells"

  # phase4-quality
  if [[ -d "$src/phase4-quality" ]]; then
    mkdir -p "$dst/phase4-quality"
    cp -n "$src/phase4-quality"/*.{json,log} "$dst/phase4-quality/" 2>/dev/null || true
    log "Pass $letter phase4 quality: $(ls $dst/phase4-quality/*.json 2>/dev/null | wc -l) json"
  fi

  # _reports
  if [[ -d "$src/_reports" ]]; then
    mkdir -p "$dst/_reports"
    cp -n "$src/_reports"/* "$dst/_reports/" 2>/dev/null || true
    log "Pass $letter _reports: $(ls $dst/_reports/ 2>/dev/null | wc -l) files"
  fi
}

stage_pass A bf16-dflash
stage_pass B fp8-mtp3

# Phase D
if [[ -d "$SWEEP/phase-D-tooleval" ]]; then
  mkdir -p "$EXP/phase-D-tooleval"
  # Scripts always
  cp -n "$SWEEP/phase-D-tooleval/run_phase_d.sh" "$EXP/phase-D-tooleval/" 2>/dev/null || true
  cp -n "$SWEEP/phase-D-tooleval/aggregate.py" "$EXP/phase-D-tooleval/" 2>/dev/null || true
  # Results if they exist
  if [[ -d "$SWEEP/phase-D-tooleval/results" ]]; then
    rsync -a --exclude='__pycache__' --exclude='*.pyc' \
      "$SWEEP/phase-D-tooleval/results/" "$EXP/phase-D-tooleval/results/" 2>/dev/null || true
  fi
  for f in run.log leaderboard.csv REPORT.md; do
    [[ -f "$SWEEP/phase-D-tooleval/$f" ]] && cp -n "$SWEEP/phase-D-tooleval/$f" "$EXP/phase-D-tooleval/"
  done
  log "Phase D staged"
fi

# Scripts dir (driver + helpers)
mkdir -p "$EXP/scripts"
for f in sweep_driver.sh run_cell.sh extract_csv.py to_master_schema.py analyze.py stage_for_publish.sh; do
  [[ -f "$SWEEP/scripts/$f" ]] && cp -n "$SWEEP/scripts/$f" "$EXP/scripts/"
done
# Snapshot harnesses (just the entrypoint files, not full venv)
for h in harness-v0.4.8 harness-upstream; do
  if [[ -d "$SWEEP/scripts/$h" ]]; then
    mkdir -p "$EXP/scripts/$h"
    rsync -a --exclude='__pycache__' --exclude='.venv' --exclude='*.pyc' \
      "$SWEEP/scripts/$h/" "$EXP/scripts/$h/" 2>/dev/null || true
  fi
done
log "scripts/ staged"

# Top-level files
for f in PLAN.md REPORT_TEMPLATE.md STATUS.md; do
  [[ -f "$SWEEP/$f" ]] && cp -n "$SWEEP/$f" "$EXP/" 2>/dev/null || true
done

# Logs (driver log can be useful evidence)
if [[ -f "$SWEEP/logs/driver.log" ]]; then
  mkdir -p "$EXP/logs"
  cp -n "$SWEEP/logs/driver.log" "$EXP/logs/" 2>/dev/null || true
fi

log "=== Staging complete ==="
log "Exp dir: $EXP"
log "Size: $(du -sh $EXP | cut -f1)"
