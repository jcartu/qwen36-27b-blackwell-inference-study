# Exp 19 — Qwen3.6-27B TP=1 **FP8 + MTP=3** (single GPU), post CPU-tune

**Date:** 2026-06-03
**Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell (SM120), **TP=1** (GPU 0 only; GPU 1 idle)
**Image:** `repne/vllm:v17` — vLLM `0.1.dev17236+g50272be4a.d20260602`
**Harness:** [`voipmonitor/llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) `v0.4.24`
**Repository path:** `jcartu/qwen36-27b-blackwell-inference-study/19-v17-tp1-fp8-mtp3`

---

## What this is

The final cell of the 2×2 matrix (TP ∈ {1,2} × {BF16, FP8}, all post CPU-tune):
single-GPU **FP8 W8A8 + MTP=3**. FP8 counterpart to [Exp 18](../18-v17-tp1-bf16-posttune/) and
TP=1 counterpart to [Exp 17](../17-v17-tp2-fp8-mtp3/).

## Key result: FP8 fits FULL 262k context on ONE card

Unlike TP=1 BF16 (Exp 18, which needed max-model-len capped to 131k), **FP8 weights free enough
VRAM that the full 262,144 context fits on a single card** at `--gpu-memory-utilization 0.80`.
GPU KV cache: **621,350 tokens** (2.37× concurrency at 262k) → only **14/41 cells KV-skipped**
(vs 26 for TP=1 BF16). This makes TP=1 FP8 the best single-card config: full context + high throughput.

## CPU-tune note

Post host CPU fix (C6 restored, `performance` governor, single-core 2,898 → ~3,450 MHz). ~10%
uplift concentrated at low concurrency (c=1–4) where per-token CPU work is on the critical path.
Residual single-core 3.5 GHz + uncore 2.5 GHz caps are BIOS-only.

## Config

```
--model Qwen/Qwen3.6-27B-FP8 --tensor-parallel-size 1 --gpu-memory-utilization 0.80
--max-model-len 262144  (full context fits on one card with FP8)
--speculative-config.method mtp --speculative-config.num_speculative_tokens 3
--attention-backend flashinfer
```
Bench: `llm_decode_bench.py --port 8000 --model Qwen3.6-27B --kv-budget 621350`

## Results

### Aggregate decode throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 91.5 | 181.4 | 356.2 | 652.8 | 1097.3 | 1636.1 | 2061.8 | 2043.7 |
| 16k | 86.8 | 169.8 | 332.9 | 581.7 | 946.4 | 1344.1 | ∅ | ∅ |
| 32k | 83.2 | 165.0 | 310.8 | 527.8 | 850.3 | ∅ | ∅ | ∅ |
| 64k | 75.0 | 148.8 | 277.3 | 465.0 | ∅ | ∅ | ∅ | ∅ |
| 128k | 62.1 | 121.4 | 239.5 | ∅ | ∅ | ∅ | ∅ | ∅ |

**Peak: 2,062 tok/s @ c=64 × ctx=0.** A single card with FP8+MTP=3 nearly reaches the *old TP=2
BF16 baseline* peak — and far exceeds TP=1 BF16 (444).

### Cross-config comparison (all post-tune, ctx=0)

| config | c=1 | c=8 | peak |
|---|---:|---:|---:|
| TP=1 BF16 (Exp 18) | 55.3 | 383.0 | 444 @ c=16 |
| TP=1 FP8 (this, Exp 19) | 91.5 | 652.8 | **2,062 @ c=64** |
| TP=2 BF16 (Exp 16) | 96.6 | 606.7 | 1,171 @ c=32 |
| TP=2 FP8 (Exp 17) | 129.0 | 936.1 | 2,960 @ c=128 |

TP=1 FP8 beats TP=2 BF16 at high concurrency and holds full context — a strong single-card option.

### Prefill throughput

| ctx | tokens | TTFT s | tok/s | N |
|---|---:|---:|---:|---:|
| 8k | 8,193 | 0.99 | 8,300 | 1 |
| 16k | 16,226 | 2.05 | 7,909 | 1 |
| 32k | 32,298 | 5.17 | 6,247 | 1 |
| 64k | 64,426 | 11.84 | 5,442 | 1 |
| 128k | 128,703 | 30.05 | 4,283 | 1 |

### Hardware

- GPUs: 1× RTX PRO 6000 Blackwell (TP=1, GPU 0 only; GPU 1 idle)
- Run duration: 20.8 min
- VRAM: max 83.1 GB on the active card
- Power: avg 517 W / max 630 W (single card active)
- Temp: avg 51°C / max 77°C

## Files

- `results/benchmark_results.json` — full harness output
- `logs/run_console_output.txt` — run console
