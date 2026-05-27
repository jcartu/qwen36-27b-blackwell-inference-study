# v13 Kitchen Sink Sweep — BF16+DFlash + FP8+MTP=3 — Plan

**Date**: 2026-05-27
**Build under test**: `repne/vllm:v13` (vLLM `0.1.dev17130+g155fef0e5`)
**Model**: `Qwen/Qwen3.6-27B` (BF16) + `z-lab/Qwen3.6-27B-DFlash` (N=8 spec drafter)
**Hardware**: 2× RTX PRO 6000 Blackwell SM120 (TP=2)
**Harness**: `voipmonitor/llm-inference-bench` (BOTH v0.4.8 pinned AND upstream main)

## Goal
Produce the canonical v13 reference matrix for BOTH config families (BF16+DFlash N=8 AND FP8+MTP=3) for upstream PR, with methodology validation against jcartu's v0.20.1-era SOTA data. Two full sweeps back-to-back, then cross-config synthesis.

## Container config (will restart with these BEFORE starting)
- `gpu-memory-utilization=0.88` (down from 0.92 to avoid prefill OOM at 131k×high-conc)
- All other v12-era flags preserved: TP=2, GPU UUIDs 1+2 (idx 1+2 host), max-model-len 262144, num_speculative_tokens 8, max-num-seqs 128, max-num-batched-tokens 32768, FlashInfer attn, prefix caching, thinking enabled.

## Phases

### Phase 0: Pre-flight
- Restart container at GMU 0.88, wait for ready
- Verify model serves
- Snapshot baseline GPU state

### Phase 1: Methodology validation (BOTH harness versions)
Run 4 anchor cells under both v0.4.8 (current local) and upstream main. If results agree within ±3%, upstream main becomes canonical for Phase 3.

Anchor cells:
- c=1 × ctx=0   (pure decode, single stream)
- c=1 × ctx=131k (long-context single stream)
- c=16 × ctx=0  (concurrency scaling)
- c=16 × ctx=131k (combined stress)

Per cell: `--duration 60 --decode-warmup-seconds 20 --max-tokens 2048 --kv-budget <auto>`

Output: `phase1-method-validation/{v0.4.8,upstream-main}/c{N}_ctx{K}/`

### Phase 2: Upstream-only diagnostics
- `--p2pmark` (NEW in upstream): Blackwell NVLink P2P measurement → `phase2-p2p/p2p_fabric.json`

### Phase 3: Full matrix, canonical version, N=5 reps
- Concurrency: 1, 2, 4, 8, 16, 32  (yes including 32 — v12 crashed, want to verify v13)
- Context: 0, 16k, 32k, 64k, 131k
- Per cell: `--duration 60 --decode-warmup-seconds 20`
- 30 cells × 5 reps = 150 measurements
- `--run-burst --burst-requests-per-concurrency 5` (adds Burst/E2E layer)

Output: `phase3-matrix/run{1..5}/benchmark_results.json`

### Phase 4: Quality profiles (NEW upstream features)
While engine is warm, run on canonical version:
- LAVD context consistency: `--profile lavd-test`
- Hotel-lights reasoning: `--profile hotel-lights`

Output: `phase4-quality/{lavd,hotel-lights}.json`

### Phase 5: Continuous CJK leak watchdog
Background process for entire sweep duration:
`python3 llm_cjk_watchdog.py --loop --port 8000 --context-tokens 65536 > phase5-cjk.log 2>&1 &`

### Phase 6: Report + CSV
- `RESULTS.md` (markdown report mirroring jcartu's SOTA.md structure)
- `master-results-v13.csv` (same schema as jcartu's `master-results.csv`)
- Per-cell statistics: mean, std, min, max, p50/p90/p99 latency, accept rate

## Time estimates
| Phase | Time |
|---|---:|
| **PASS A — BF16+DFlash N=8** | |
| A0 — Pre-flight + restart at GMU 0.88 | 8 min |
| A1 — Methodology validation (4 cells × 2 harness versions) | 25 min |
| A2 — P2P diagnostic (upstream-only, hardware) | 10 min |
| A3 — Full matrix N=5 (30 cells × 60s × 5 reps + warmup) | ~110 min |
| A4 — Quality profiles (LAVD + hotel-lights) | 30 min |
| A5 — CJK watchdog | (continuous, parallel) |
| A6 — Per-config report + CSV | 20 min |
| **A→B handoff** | |
| Swap launcher to FP8+MTP=3, relaunch, wait for ready | 8 min |
| **PASS B — FP8+MTP=3** | |
| B0 — Pre-flight (already at GMU 0.88) | 3 min |
| B1 — Methodology validation (4 cells × 2 versions) | 25 min |
| B2 — P2P diag SKIP (already captured, hardware identical) | 0 min |
| B3 — Full matrix N=5 | ~110 min |
| B4 — Quality profiles | 30 min |
| B5 — CJK watchdog | (continuous) |
| B6 — Per-config report + CSV | 20 min |
| **C — Cross-config synthesis** | |
| BF16+DFlash vs FP8+MTP=3 head-to-head report + PR draft | 30 min |
| Buffer (OOM recovery, restarts, re-runs) | 60 min |
| **TOTAL** | **~8.0h** |

## Risk register
- **R1**: v13 OOM on 131k prefill at c≥8 → mitigated by GMU 0.88
- **R2**: DFlash crash at c=32 (v12 behavior) → documented either way; auto-restart handles
- **R3**: Upstream main introduces bug → Phase 1 detects, fall back to v0.4.8 canonical
- **R4**: CJK watchdog catches drift → escalate immediately, halt sweep
- **R5**: Display GPU touched → IMMEDIATE HALT, no exceptions

## Deliverable
PR to `voipmonitor/llm-inference-bench` adding:
```
docs/v13-blackwell-bf16-dflash-2026-05.md      # per-config report
docs/v13-blackwell-fp8-mtp3-2026-05.md         # per-config report
docs/v13-blackwell-sota-2026-05.md             # cross-config synthesis (the centerpiece)
docs/artifacts-v13/A-bf16-dflash/{phase1..phase4}/
docs/artifacts-v13/B-fp8-mtp3/{phase1..phase4}/
docs/artifacts-v13/p2p_fabric.json
docs/artifacts-v13/master-results-v13.csv     # both configs, same schema as jcartu's
```
Cross-referenced from `jcartu/qwen36-27b-blackwell-inference-study` as the v13 SOTA update for both production configs.
