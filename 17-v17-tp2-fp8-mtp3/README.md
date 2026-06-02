# Exp 17 — Qwen3.6-27B TP=2 **FP8 + MTP=3**, post CPU-tune

**Date:** 2026-06-03
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (SM120), TP=2
**Image:** `repne/vllm:v17` — vLLM `0.1.dev17236+g50272be4a.d20260602`
**Harness:** [`voipmonitor/llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) `v0.4.24`
**Repository path:** `jcartu/qwen36-27b-blackwell-inference-study/17-v17-tp2-fp8-mtp3`

---

## What this is

FP8 counterpart to [Exp 16](../16-v17-tp2-bf16-posttune/). Same TP=2, same harness, same host
**(post CPU-tune)** — but **FP8 W8A8 weights + MTP=3 speculative** instead of BF16+DFlash.

## CPU-tune note (applies to all post-tune runs)

This run was taken **after** the host CPU turbo fix (removed `intel_idle.max_cstate=2` boot cap →
C6 restored; `performance` governor; single-core boost 2,898 → ~3,450 MHz). The fix delivers
**~10% uplift at low concurrency (c=1–4)** where per-token CPU work (scheduler, MTP draft/verify,
detokenization) is on the critical path; high-concurrency cells are GPU-bound and largely unaffected.
A residual single-core (3.5 GHz) and uncore (2.5 GHz) cap remains, both **BIOS-only** (VR current
capability + uncore ratio).

## Config

```
--model Qwen/Qwen3.6-27B-FP8 --tensor-parallel-size 2 --gpu-memory-utilization 0.80
--max-model-len 262144 --max-num-seqs 128 --max-num-batched-tokens 32768
--max-cudagraph-capture-size 256 --enable-prefix-caching
--speculative-config.method mtp --speculative-config.num_speculative_tokens 3
--attention-backend flashinfer
```
Bench: `llm_decode_bench.py --port 8000 --model Qwen3.6-27B --kv-budget 1711960`

GPU KV cache: **1,711,960 tokens** (6.53× concurrency at 262k — ~2× the BF16 budget, since FP8
weights free ~half the VRAM). Only **10/41 cells KV-skipped** (vs 14-15 for BF16).

## Results

### Aggregate decode throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 129.0 | 250.4 | 490.6 | 936.1 | 1607.7 | 2153.4 | 2696.4 | 2959.7 |
| 16k | 130.0 | 250.4 | 491.5 | 898.8 | 1371.7 | 1883.3 | 2278.0 | ∅ |
| 32k | 119.3 | 245.3 | 468.2 | 828.2 | 1238.8 | 1668.6 | ∅ | ∅ |
| 64k | 111.7 | 218.9 | 430.6 | 713.3 | 984.1 | ∅ | ∅ | ∅ |
| 128k | 96.7 | 192.1 | 368.3 | 535.3 | ∅ | ∅ | ∅ | ∅ |

**Peak: 2,960 tok/s @ c=128 × ctx=0.** FP8+MTP=3 scales cleanly to c=128 (BF16+DFlash plateaued
~c=32). This is the fastest configuration measured in this study.

### FP8+MTP=3 vs BF16+DFlash (both TP=2, post-tune: Exp 17 vs Exp 16)

| cell | BF16 (Exp 16) | FP8 (Exp 17) | FP8 advantage |
|---|---:|---:|---:|
| c=1 × 0 | 96.6 | 129.0 | **+34%** |
| c=8 × 0 | 606.7 | 936.1 | **+54%** |
| c=32 × 0 | 1,170.6 | 2,153.4 | **+84%** |
| c=128 × 0 | 1,113.1 | 2,959.7 | **+166%** |
| peak | 1,170.6 | 2,959.7 | **+153%** |
| prefill 8k | 7,994 | 9,494 | +19% |

FP8 is dramatically faster, especially at high concurrency, and uses less VRAM (165 GB vs 173 GB
peak across both cards). FP8+MTP=3 remains the production-recommended config.

### Prefill throughput

| ctx | tokens | TTFT s | tok/s | N |
|---|---:|---:|---:|---:|
| 8k | 8,193 | 0.86 | 9,494 | 1 |
| 16k | 16,226 | 1.74 | 9,302 | 1 |
| 32k | 32,298 | 3.70 | 8,722 | 1 |
| 64k | 64,426 | 8.11 | 7,945 | 1 |
| 128k | 128,703 | 19.25 | 6,684 | 1 |

### Hardware (sampled during measured cells)

- GPUs: 2× RTX PRO 6000 Blackwell
- Run duration: 20.8 min (558 samples)
- GPU util: avg 88% / max 100%
- VRAM: max 165.0 GB
- Power: avg 796 W / max 1122 W
- Temp: avg 65°C / max 84°C

## Files

- `results/benchmark_results.json` — full harness output
- `logs/run_console_output.txt` — run console with final summary tables
