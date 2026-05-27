#!/usr/bin/env bash
# stage_for_publish.sh — copy Exp 10 sweep artifacts into the publish repo.
#
# Repo layout target:
#   _publish-staging/qwen36-27b-blackwell-inference-study/
#     10-parser-axis/
#       README.md                            (rendered from EXP10_README_TEMPLATE.md)
#       EXP10_README_TEMPLATE.md             (source of truth for re-renders)
#       leaderboard.csv                      (from build_leaderboard.py)
#       configs/matrix.tsv
#       scripts/{launch_cell.sh,run_exp10.sh,build_leaderboard.py,render_exp10_readme.py,stage_for_publish.sh}
#       results/
#         stage1/{S1-v13-fp8-xml,S1-v13-fp8-coder}/{teb-results.json,console.log}
#         stage1/winner.txt
#         stage2/{S2-v13-bf16-P*,S2-nightly-bf16-P*}/{teb-results.json,console.log}
#         stage2/winner.txt
#         stage3/S3-nightly-fp8-P*/{teb-results.json,console.log}
#         stage3/winner.txt
#         frontier/Y{1..7}-*/{teb-results.json,console.log}
#       logs/{exp10-driver.log,frontier.log}
#
# Idempotent: existing files are overwritten.

set -euo pipefail

SWEEP="${SWEEP:-/home/josh/qwen-vllm-test/sweeps/10-parser-axis}"
EXP="${EXP:-/home/josh/qwen-vllm-test/_publish-staging/qwen36-27b-blackwell-inference-study/10-parser-axis}"

log() { printf '[stage] %s\n' "$*"; }

[[ -d "$SWEEP" ]] || { echo "sweep dir missing: $SWEEP"; exit 1; }
[[ -f "$SWEEP/results/stage1/winner.txt" ]] || {
  echo "Stage 1 winner.txt missing — orchestrator hasn't completed Stage 1 yet."
  echo "Tail of driver log:"
  tail -5 "$SWEEP/logs/exp10-driver.log" 2>/dev/null || true
  exit 1
}

mkdir -p "$EXP"/{configs,scripts,results/{stage1,stage2,stage3,frontier},logs}

# ── Top-level files ─────────────────────────────────────────────────────────
log "Top-level: README, template, leaderboard"
cp -f "$SWEEP/EXP10_README_TEMPLATE.md" "$EXP/EXP10_README_TEMPLATE.md"
[[ -f "$SWEEP/README.md" ]] && cp -f "$SWEEP/README.md" "$EXP/README.md" || log "  (no rendered README.md yet — run render_exp10_readme.py first)"
[[ -f "$SWEEP/leaderboard.csv" ]] && cp -f "$SWEEP/leaderboard.csv" "$EXP/leaderboard.csv" || log "  (no leaderboard.csv yet — run build_leaderboard.py first)"

# ── Configs ─────────────────────────────────────────────────────────────────
log "Configs"
cp -f "$SWEEP/configs/matrix.tsv" "$EXP/configs/matrix.tsv"

# ── Scripts (verbatim, for repro) ───────────────────────────────────────────
log "Scripts"
for f in launch_cell.sh run_exp10.sh build_leaderboard.py render_exp10_readme.py stage_for_publish.sh; do
  if [[ -f "$SWEEP/scripts/$f" ]]; then
    cp -f "$SWEEP/scripts/$f" "$EXP/scripts/$f"
  fi
done

# ── Results per stage ───────────────────────────────────────────────────────
stage_copy() {
  local stage="$1"
  local src="$SWEEP/results/$stage"
  local dst="$EXP/results/$stage"
  [[ -d "$src" ]] || return 0
  mkdir -p "$dst"
  local n=0
  for cell_dir in "$src"/*/; do
    [[ -d "$cell_dir" ]] || continue
    local cell
    cell=$(basename "$cell_dir")
    mkdir -p "$dst/$cell"
    for f in teb-results.json console.log SKIP_REASON.md launch-*.log; do
      # shellcheck disable=SC2086
      for src_f in $cell_dir/$f; do
        [[ -f "$src_f" ]] && cp -f "$src_f" "$dst/$cell/"
      done
    done
    n=$((n+1))
  done
  [[ -f "$src/winner.txt" ]] && cp -f "$src/winner.txt" "$dst/winner.txt"
  log "  $stage: $n cells"
}

log "Results"
stage_copy stage1
stage_copy stage2
stage_copy stage3
stage_copy frontier

# ── Logs ────────────────────────────────────────────────────────────────────
log "Logs"
for f in exp10-driver.log frontier.log; do
  [[ -f "$SWEEP/logs/$f" ]] && cp -f "$SWEEP/logs/$f" "$EXP/logs/$f"
done

log "Done. Staged to: $EXP"
log "Next steps:"
log "  cd $EXP/.. && git add 10-parser-axis && git status"
log "  git commit -m 'Exp 10: staged parser × image tournament' && git push"
