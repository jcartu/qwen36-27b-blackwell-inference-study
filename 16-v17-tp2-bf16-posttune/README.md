# Exp 16 — Qwen3.6-27B TP=2 BF16, **post CPU-tune** (turbo-fix validation)

**Date:** 2026-06-03
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell (SM120), TP=2
**Image:** `repne/vllm:v17` — vLLM `0.1.dev17236+g50272be4a.d20260602`
**Harness:** [`voipmonitor/llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) `v0.4.24`
**Repository path:** `jcartu/qwen36-27b-blackwell-inference-study/16-v17-tp2-bf16-posttune`

---

## What this is

A re-run of the [Exp 13](../13-v17-decode-matrix/) TP=2 BF16 sweep **after a host CPU performance fix**.
Identical model/config/harness — the only change is the **host CPU power/turbo state**.

## The CPU fix (why this is ~10% faster at low concurrency)

The host (Xeon w9-3495X, ASUS W790E-SAGE SE) was previously running with self-inflicted
power-management settings that **hard-capped CPU turbo at 2,898 MHz** and starved vLLM's
per-token CPU work (scheduler, DFlash drafter sample/verify, detokenization):

| | Before (Exp 13) | After (Exp 16) |
|---|---|---|
| Kernel boot args | `intel_idle.max_cstate=2 processor.max_cstate=2` (no C6 deep idle → no turbo headroom) | removed (C6 restored) |
| CPU governor | `powersave` | `performance` |
| EPP | `balance_performance` | `performance` |
| Single-core boost | **2,898 MHz** (hard-clamped) | **~3,450 MHz** (+19%) |
| stray host load | a runaway memory-indexer pegging 8 cores | stopped during bench |

**Effect on inference:** the gain concentrates at **low concurrency (c=1–4)**, where per-token CPU
overhead dominates and CPU clock is on the critical path. At high concurrency the GPU is the
bottleneck so the CPU fix barely moves it. This is the expected signature of a CPU-frequency fix.

> A residual ceiling remains (single-core tops ~3.5 GHz, not the rated 4.8 GHz, and uncore is
> pinned at 2.5 GHz). Those are **BIOS-level** limits (VR current capability + uncore/mesh ratio)
> that cannot be changed from the OS — see the diagnosis notes. Closing them would recover the
> rest of the gap.

## Config (identical to Exp 13)

```
--model Qwen/Qwen3.6-27B --tensor-parallel-size 2 --gpu-memory-utilization 0.80
--max-model-len 262144 --max-num-seqs 128 --max-num-batched-tokens 32768
--max-cudagraph-capture-size 256 --enable-prefix-caching
--speculative-config.method dflash --speculative-config.model z-lab/Qwen3.6-27B-DFlash
--speculative-config.num_speculative_tokens 8 --attention-backend flashinfer
```
Bench: `llm_decode_bench.py --port 8000 --model Qwen3.6-27B --kv-budget 854152`

## Results

### Aggregate decode throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 96.6 | 179.3 | 351.6 | 606.7 | 858.9 | 1170.6 | 1065.0 | 1113.1 |
| 16k | 102.4 | 191.9 | 343.9 | 584.9 | 902.0 | 1125.9 | ∅ | ∅ |
| 32k | 101.9 | 189.4 | 348.9 | 570.2 | 849.3 | ∅ | ∅ | ∅ |
| 64k | 90.2 | 174.9 | 327.2 | 545.9 | ∅ | ∅ | ∅ | ∅ |
| 128k | 93.1 | 167.6 | 304.0 | ∅ | ∅ | ∅ | ∅ | ∅ |

**Peak: 1,171 tok/s @ c=32 × ctx=0** (vs 1,138 in Exp 13).

### Improvement vs Exp 13 (BF16 pre-tune) — same config, CPU fix only

| cell | Exp 13 (pre-tune) | Exp 16 (post-tune) | Δ |
|---|---:|---:|---:|
| c=1 × 0 | 86.5 | 96.6 | **+11.7%** |
| c=4 × 0 | 320.7 | 351.6 | **+9.6%** |
| c=8 × 0 | 574.8 | 606.7 | +5.5% |
| c=16 × 0 | 865.6 | 858.9 | ~flat |
| c=32 × 0 (peak) | 1,138.1 | 1,170.6 | +2.9% |
| c=1 × 16k | 92.0 | 102.4 | **+11.3%** |
| c=1 × 32k | 93.4 | 101.9 | **+9.1%** |
| c=1 × 128k | 83.6 | 93.1 | **+11.4%** |

**Takeaway: ~10% uplift at low concurrency (c=1–4), shrinking to ~flat at high concurrency** — exactly
where CPU-side per-token work matters. Interactive/agentic single-stream workloads benefit most.

### Prefill throughput

| ctx | tokens | TTFT s | tok/s | N |
|---|---:|---:|---:|---:|
| 8k | 8,193 | 1.02 | 7,994 | 1 |
| 16k | 16,226 | 2.06 | 7,891 | 1 |
| 32k | 32,298 | 4.25 | 7,591 | 1 |
| 64k | 64,426 | 9.21 | 6,994 | 1 |
| 128k | 128,703 | 21.37 | 6,022 | 1 |

### Hardware (sampled during measured cells)

- GPUs: 2× RTX PRO 6000 Blackwell
- Run duration: 19.7 min (527 samples)
- GPU util: avg 84% / max 100%
- VRAM: max 172.7 GB
- Power: avg 827 W / max 1118 W
- Temp: avg 66°C / max 83°C

## Files

- `results/benchmark_results.json` — full harness output
- `logs/run_console_output.txt` — run console with final summary tables
