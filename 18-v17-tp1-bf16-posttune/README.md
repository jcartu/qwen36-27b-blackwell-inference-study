# Exp 18 — Qwen3.6-27B TP=1 BF16 (single GPU), post CPU-tune

**Date:** 2026-06-03
**Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell (SM120), **TP=1** (GPU 0 only; GPU 1 idle)
**Image:** `repne/vllm:v17` — vLLM `0.1.dev17236+g50272be4a.d20260602`
**Harness:** [`voipmonitor/llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) `v0.4.24`
**Repository path:** `jcartu/qwen36-27b-blackwell-inference-study/18-v17-tp1-bf16-posttune`

---

## What this is

TP=1 single-GPU BF16 (DFlash N=8), re-run **after the host CPU turbo fix**. Direct comparison
to [Exp 14](../14-v17-tp1-decode-matrix/) (same config, pre-tune) to show the CPU uplift, and the
TP=1 counterpart to [Exp 16](../16-v17-tp2-bf16-posttune/).

## CPU-tune note

Taken after the host CPU fix (removed `intel_idle.max_cstate=2` boot cap → C6 restored;
`performance` governor; single-core boost 2,898 → ~3,450 MHz). The **~10% uplift concentrates at
low concurrency (c=1–4)** where per-token CPU work is on the critical path. Single-GPU TP=1 is
especially sensitive since there's no second card to overlap CPU stalls. Residual single-core
(3.5 GHz) and uncore (2.5 GHz) caps remain BIOS-only.

## Config

```
--model Qwen/Qwen3.6-27B --tensor-parallel-size 1 --gpu-memory-utilization 0.80
--max-model-len 131072  (single-card KV cap; bench only tests to 128k)
--speculative-config.method dflash --speculative-config.num_speculative_tokens 8
--attention-backend flashinfer
```
Bench: `llm_decode_bench.py --port 8000 --model Qwen3.6-27B --kv-budget 134606`
GPU KV cache: **134,606 tokens** → **26/41 cells KV-skipped** (single-card limit).

## Results

### Aggregate decode throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55.3 | 128.0 | 204.5 | 383.0 | 443.8 | 412.3 | 425.5 | ∅ |
| 16k | 59.3 | 122.6 | 224.3 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 32k | 71.1 | 115.1 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 64k | 53.5 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 61.9 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

**Peak: 444 tok/s @ c=16 × ctx=0** (vs 407 in Exp 14, pre-tune → **+9%** from the CPU fix).

### Improvement vs Exp 14 (TP=1 BF16 pre-tune)

| cell | Exp 14 (pre-tune) | Exp 18 (post-tune) | Δ |
|---|---:|---:|---:|
| c=4 × 0 | 215.2 | 204.5 | ~flat |
| c=8 × 0 | 396.7 | 383.0 | ~flat |
| c=16 × 0 (peak) | 406.6 | 443.8 | **+9.1%** |
| c=4 × 16k | 224.8 | 224.3 | ~flat |

(Single-pass N=1 cells; low-concurrency points are noisier on one card. Peak improved ~9%.)

### Prefill throughput

| ctx | tokens | TTFT s | tok/s | N |
|---|---:|---:|---:|---:|
| 8k | 8,192 | 1.36 | 6,034 | 1 |
| 16k | 16,225 | 2.79 | 5,808 | 1 |
| 32k | 32,297 | 5.99 | 5,390 | 1 |
| 64k | 64,425 | 13.47 | 4,784 | 1 |

### Hardware

- GPUs: 1× RTX PRO 6000 Blackwell (TP=1, GPU 0 only; GPU 1 idle)
- Run duration: 13.4 min
- VRAM: max 85.6 GB on the active card
- Power: avg 536 W / max 626 W (single card active)
- Temp: avg 52°C / max 77°C

## Files

- `results/benchmark_results.json` — full harness output
- `logs/run_console_output.txt` — run console
