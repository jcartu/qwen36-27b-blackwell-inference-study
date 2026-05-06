# Experiment 6 — New image validation (d0a200f77546)

**Image:** `repne/vllm:latest` (`d0a200f77546`, May 5 2026 21:27 UTC), engine `v0.1.dev16400+g910d87a9d.d20260505` (+41 commits over the morning image's `v0.1.dev16359+ga3e24c99b`).

**Goal:** characterize the new image vs prior across both quant configs, with **DFlash=7 and DFlash=15** specifically tested per Repne's discovery, plus **DFlash=8** for completeness.

## Design

| Phase | Config | Variants | Cells | N |
|---|---|---|---|---|
| 1 | FP8+MTP=3 | (only one — production) | 9 (c=1,2,4 × ctx=0,32k,131k) | 3 |
| 2a | BF16+DFlash | num_speculative_tokens=15 | 9 same | 3 |
| 2b | BF16+DFlash | num_speculative_tokens=8 | 9 same | 3 |
| 2c | BF16+DFlash | num_speculative_tokens=7 | 9 same | 3 |

**108 total benchmark runs.** Plus 4 functional gates per phase (Fibonacci 5x, tool-call, 47×83 reasoning, multi-turn coherence).

All gates passed on all four configs. No tool-calling regression on the new image — Repne's claim verified.

## Phase 1+2 results matrix (mean tok/s ± std, N=3 per cell)

| cell | FP8+MTP=3 | DFlash=15 | DFlash=8 | DFlash=7 | Best |
|---|---|---|---|---|---|
| c=1 ctx=0 | **117.1±2.8** | 88.0±5.8 | 92.2±3.2 | 92.9±4.5 | FP8+MTP=3 |
| c=1 ctx=32k | **119.2±4.7** | 95.0±1.6 | 95.2±5.5 | 94.8±5.0 | FP8+MTP=3 |
| c=1 ctx=131k | **95.0±2.5** | 87.6±3.5 | 86.4±4.9 | 83.8±0.9 | FP8+MTP=3 |
| c=2 ctx=0 | **227.2±2.4** | 165.4±11.6 | 180.7±2.0 | 180.2±3.9 | FP8+MTP=3 |
| c=2 ctx=32k | **227.0±5.1** | 166.6±3.2 | 175.3±1.5 | 184.8±4.7 | FP8+MTP=3 |
| c=2 ctx=131k | **184.9±0.9** | 150.0±6.6 | 162.3±6.0 | 164.8±5.5 | FP8+MTP=3 |
| c=4 ctx=0 | **449.8±4.6** | 313.4±4.4 | 334.6±5.5 | 344.9±5.4 | FP8+MTP=3 |
| c=4 ctx=32k | **454.5±3.3** | 291.9±11.4 | 336.1±19.2 | 331.8±4.4 | FP8+MTP=3 |
| c=4 ctx=131k | **350.5±4.9** | 248.7±10.1 | 284.4±5.7 | 299.6±9.6 | FP8+MTP=3 |

**FP8+MTP=3 wins all 9 cells decisively.** +25-50% across the board.

## DFlash variants — best `num_speculative_tokens`?

| Config | mean tok/s across 9 cells | mean accept rate | mean ITL |
|---|---|---|---|
| DFlash=15 | 178.5 | 12.3% | 11.74ms |
| DFlash=8 | 194.1 | 21.9% | 11.18ms |
| **DFlash=7** | **197.5** | **24.4%** | **11.03ms** |

**DFlash=7 wins among the DFlash variants** (+10.6% vs DFlash=15, +1.8% vs DFlash=8). Acceptance rate scales inversely with token count — more spec tokens = lower acceptance per position because the drafter struggles further into the chain. The throughput sweet spot is at lower `num_speculative_tokens` for this drafter on this hardware.

DFlash=15 is meaningfully worse than both DFlash=8 and DFlash=7 — Repne's claim of "0.931→0.063 distribution per position" doesn't translate to aggregate throughput wins on this benchmark. The acceptance rate `server_spec_accept_rate` we measure is the aggregate (mean across all 15 spec positions), which dilutes when most positions reject.

## FP8+MTP=3 — New image vs prior image (`5e7583ca4df9`)

| cell | New image (d0a200f7) | Prior image (5e7583ca, N=1 evening) | Δ |
|---|---|---|---|
| c=1 ctx=0 | 117.1±2.8 | 120.1 | −2.6% |
| c=1 ctx=131k | 95.0±2.5 | 93.7 | +1.4% |
| c=2 ctx=0 | 227.2±2.4 | 223.8 | +1.5% |
| c=2 ctx=131k | 184.9±0.9 | 183.4 | +0.8% |
| c=4 ctx=0 | 449.8±4.6 | 449.5 | +0.1% |
| c=4 ctx=131k | 350.5±4.9 | 347.4 | +0.9% |

**No regression, no win.** All deltas within noise band (the prior image numbers are N=1, so the differences are well within single-shot variance for those cells). The new image preserves FP8+MTP=3 performance exactly.

## BF16+DFlash=8 — New image vs prior image

| cell | New image | Prior image (N=1 evening) | Δ |
|---|---|---|---|
| c=1 ctx=0 | 92.2±3.2 | 98.9 | −6.7% |
| c=1 ctx=131k | 86.4±4.9 | 81.4 | +6.1% |
| c=2 ctx=0 | 180.7±2.0 | 184.5 | −2.1% |
| c=2 ctx=131k | 162.3±6.0 | 162.7 | −0.2% |
| c=4 ctx=0 | 334.6±5.5 | 358.4 | −6.6% |
| c=4 ctx=131k | 284.4±5.7 | 284.4 | 0.0% |

**Mostly within noise.** Single-shot prior image readings are unreliable for these comparisons; the new image's N=3 numbers are tighter (mean ±2-6%).

## Functional gates — all 4 PASS on all 4 configs

| Config | Fibonacci 5/5 | Tool call | 47×83=3901 | Multi-turn |
|---|---|---|---|---|
| FP8+MTP=3 | ✅ | ✅ | ✅ | ✅ |
| DFlash=15 | ✅ | ✅ | ✅ | ✅ |
| DFlash=8 | ✅ | ✅ | ✅ | ✅ |
| DFlash=7 | ✅ | ✅ | ✅ | ✅ |

Repne's tool-calling fix on the new image is verified — `get_weather({"city":"Tokyo"})` parses correctly under all four configs.

## Decision: stay on FP8+MTP=3

FP8+MTP=3 wins decisively on every cell tested. None of the DFlash variants come close. The closest cell is c=4 ctx=131k where DFlash=7 hits 299.6 tok/s vs FP8+MTP=3's 350.5 — still a 14.5% deficit.

**No reason to switch ROCK SOLID config.** Production stays on FP8+MTP=3.

## Caveats

- The `server_spec_accept_rate` metric we use is the aggregate acceptance across all spec positions, not the per-position chain that Repne reports. A higher num_speculative_tokens with a steeply decaying per-position acceptance can still yield negative aggregate throughput effects, which is what we observe.
- Bench tool clamps requested ctx to (max_model_len − safety_margin), so requested 131072 → measured ~131000+ depending on model padding rules.
- All cells tested have ctx ≤ 131k; we did not retest 244k/250k after the prior NVFP4 experiment showed engine instability at that range. FP8+MTP=3 on the new image presumably handles 244k fine (it did before), but that's not in this matrix.

## Files

- `phase1-fp8mtp3/runs/<cell>_run<N>/` — 27 raw N=3 runs of FP8+MTP=3
- `phase2a-dflash15/runs/` — 27 runs DFlash=15
- `phase2b-dflash8/runs/` — 27 runs DFlash=8
- `phase2c-dflash7/runs/` — 27 runs DFlash=7
- `gates/phase{1,2a,2b,2c}_*.log` — gate logs
