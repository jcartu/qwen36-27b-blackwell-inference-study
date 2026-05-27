# v13 Kitchen-Sink Sweep — Live Status

**Driver PID**: see `logs/sweep.pid`
**Driver log**: `logs/driver.log`
**Started**: 2026-05-27 02:55:07 UTC

## Quick-look commands (paste these when you wake up)

```bash
SWEEP=/home/josh/qwen-vllm-test/sweeps/v13-kitchen-sink-bf16dflash-and-fp8mtp3

# 1. Is driver still running?
ps -p $(cat $SWEEP/logs/sweep.pid) -o pid,stat,etime,cmd

# 2. Where is it now?
tail -40 $SWEEP/logs/driver.log

# 3. Phase progress
echo "A1 (need 8):"   ; ls $SWEEP/A-bf16-dflash/phase1-method-validation/v0.4.8/*.json 2>/dev/null | wc -l ; ls $SWEEP/A-bf16-dflash/phase1-method-validation/upstream-main/*.json 2>/dev/null | wc -l
echo "A3 (need 30 per rep × 5 reps = 150):"
for r in 1 2 3 4 5; do echo -n "  run$r: " ; ls $SWEEP/A-bf16-dflash/phase3-matrix/run$r/*.json 2>/dev/null | wc -l ; done
echo "A4 (need 2):"  ; ls $SWEEP/A-bf16-dflash/phase4-quality/*.json 2>/dev/null | wc -l
echo "B1 (need 8):"  ; ls $SWEEP/B-fp8-mtp3/phase1-method-validation/v0.4.8/*.json 2>/dev/null | wc -l ; ls $SWEEP/B-fp8-mtp3/phase1-method-validation/upstream-main/*.json 2>/dev/null | wc -l
echo "B3 (need 150):"
for r in 1 2 3 4 5; do echo -n "  run$r: " ; ls $SWEEP/B-fp8-mtp3/phase3-matrix/run$r/*.json 2>/dev/null | wc -l ; done

# 4. GPU sanity (display should be ≈3277 MiB throughout)
nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader

# 5. Any permanent failures?
find $SWEEP -name "_permanent_failures.log" -exec echo "--- {} ---" \; -exec cat {} \;
```

## Phase plan

| Pass | Phase | Cells | Per-cell | Subtotal |
|---|---|---:|---:|---:|
| **A: BF16+DFlash N=8** | A1 method validation | 4 × 2 versions = 8 | ~100s | ~13min |
| | A2 P2P fabric | 1 | ~5min | ~5min |
| | A3 matrix N=5 | 6 conc × 5 ctx × 5 reps = 150 | ~100s avg | ~250min |
| | A4 quality profiles | 2 (lavd, hotel-lights) | ~10min | ~20min |
| | **Pass A total** | | | **~4.8h** |
| handoff | swap to FP8+MTP=3 | | | ~10min |
| **B: FP8+MTP=3** | B1 method | 8 | ~100s | ~13min |
| | B3 matrix N=5 | 150 | ~100s avg | ~250min |
| | B4 quality | 2 | ~10min | ~20min |
| | **Pass B total** | | | **~4.8h** |
| **C: synthesis** | CSV extraction | | | ~5min |
| **TOTAL** | | | | **~9.7h** |

## Idempotency

Driver is **fully resumable**. Re-running `bash scripts/sweep_driver.sh` will:
- Skip any cell whose JSON exists and has a valid `aggregate_tps` field
- Restart only failed/missing cells
- Re-attempt failed cells once on retry

## Safety guardrails

- **Display GPU check** before every phase + every cell in A3/B3. If `538bf008` GPU's memory exceeds 4000 MiB → driver aborts immediately.
- **Container auto-restart**: if vLLM endpoint becomes unreachable, driver waits for Docker restart policy + relaunches if needed.
- **FP8 fallback**: if `repne/vllm:v13` can't run FP8+MTP=3, driver auto-reverts launcher to `repne/vllm:v12` for Pass B (flag dropped at `B-fp8-mtp3/_fallback.flag`).
- **No interactive prompts**: v0.4.8 harness has a "Upgrade and restart? [Y/n]" prompt that the driver pipes `n` to.

## Known pre-sweep results

- v13 BF16+DFlash c=1 ctx=0 single-stream: 92-96 tok/s (multi-run noise)
- v12 BF16+DFlash same config: 93.7 tok/s (yesterday's reference)
- v13 steady-state ~87.8 GiB/rank at GMU 0.92 → ~83-86 GiB/rank at GMU 0.88
- KV budget at GMU 0.88: 1,003,012 tokens, max concurrency 3.83x at 262k

## On wake-up: if everything completed

The driver will have produced:
- `A-bf16-dflash/results-A.csv` (Pass A all reps merged)
- `B-fp8-mtp3/results-B.csv` (Pass B all reps merged)
- `experiment-09-master-results.csv` (both passes, jcartu format)

I'll then proceed to write the per-config + cross-config reports, push to GitHub as Experiment 09, and do the workspace cleanup.

## On wake-up: if driver died mid-run

1. Check `logs/driver.log` for the last action
2. Just re-run `setsid nohup bash $SWEEP/scripts/sweep_driver.sh > $SWEEP/logs/sweep.stdout.log 2>&1 &` — idempotency handles it.
3. If a permanent failure log exists, those cells need manual investigation.
