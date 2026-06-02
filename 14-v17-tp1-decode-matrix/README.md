# Exp 14 — Qwen3.6-27B decode/prefill matrix on Repne v17, **TP=1 (single GPU)**

**Date:** 2026-06-02
**Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120), **TP=1** (GPU 0 only; GPU 1 idle)
**Image:** `repne/vllm:v17` — vLLM `0.1.dev17236+g50272be4a.d20260602`
**Harness:** [`voipmonitor/llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) `v0.4.24`
**Repository path:** `jcartu/qwen36-27b-blackwell-inference-study/14-v17-tp1-decode-matrix`

---

## What this is

The same decode + prefill sweep as [Exp 13](../13-v17-decode-matrix/), but on a **single GPU (TP=1)**
instead of TP=2. Direct A/B on tensor-parallel degree, same v17 image, same harness invocation.

## Key constraint: single-card KV cache

On one card, BF16 27B + DFlash drafter weights take **53.7 GiB**, leaving only **14.5 GiB** for
KV cache at `--gpu-memory-utilization 0.80`. That is **not** enough to hold the full 262,144-token
context (which needs ~26.7 GiB KV), so:

- `--max-model-len` was reduced to **131072** (the card's safe max; vLLM estimated ~134,784).
  The bench only tests up to 128k context, so no bench cell is affected by this cap.
- GPU KV cache size: **134,606 tokens** (vs 854,152 on TP=2 — 6.3× smaller).
- **26 of 41 cells were KV-skipped** (`∅`). High-concurrency × high-context simply does not fit on
  one card. This is the expected, honest result of TP=1, not a failure.

> A follow-up run (Exp 15) raises `--gpu-memory-utilization` to fit more/all of the 262k context on
> the single card.

## Configuration

```
--model Qwen/Qwen3.6-27B --served-model-name Qwen3.6-27B
--tensor-parallel-size 1            # <-- the change vs Exp 13
--gpu-memory-utilization 0.80
--max-model-len 131072              # <-- capped to fit single-card KV (Exp 13 used 262144)
--max-num-seqs 128
--max-num-batched-tokens 32768
--max-cudagraph-capture-size 256
--enable-prefix-caching
--speculative-config.method dflash
--speculative-config.model z-lab/Qwen3.6-27B-DFlash
--speculative-config.num_speculative_tokens 8
--speculative-config.draft_sample_method greedy
--attention-backend flashinfer
```

Bench invocation:

```
llm_decode_bench.py --port 8000 --model Qwen3.6-27B --kv-budget 134606
```

## Results

### Aggregate decode throughput (tok/s)

`∅` = cell skipped, KV budget (134,606 tokens) exceeded.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 67.1 | 115.5 | 215.2 | 396.7 | **406.6** | 396.8 | 385.0 | ∅ |
| 16k | 54.5 | 111.2 | 224.8 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 32k | 57.0 | 115.8 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 64k | 49.0 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 60.8 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

**Peak: 407 tok/s @ c=16 × ctx=0.** Throughput saturates at c≥8 (single-card compute bound); past
c=8 there is no further scaling and c=128×0 is KV-skipped.

### Prefill throughput

| ctx | tokens | TTFT s | tok/s | N |
|---|---:|---:|---:|---:|
| 8k | 8,191 | 1.39 | 5,914 | 1 |
| 16k | 16,221 | 2.84 | 5,711 | 1 |
| 32k | 32,291 | 6.06 | 5,329 | 1 |
| 64k | 64,414 | 13.74 | 4,689 | 1 |

(128k prefill cell KV-skipped on the single card.)

### Hardware (sampled during measured cells)

- GPUs: 1× RTX PRO 6000 Blackwell (TP=1, GPU 0 only; GPU 1 idle)
- Run duration: 13.6 min (363 samples)
- GPU util: avg 43% / max 100%
- VRAM: max 86.1 GB on the active card
- Power: avg 618 W / max 717 W (single card active)
- Temp: avg 59°C / max 83°C

## TP=1 vs TP=2 (Exp 13) — same image, same bench

| cell | TP=1 (this run) | TP=2 (Exp 13) | TP=2 / TP=1 |
|---|---:|---:|---:|
| decode c=1 × 0 | 67.1 | 86.5 | 1.29× |
| decode c=4 × 0 | 215.2 | 320.7 | 1.49× |
| decode c=8 × 0 | 396.7 | 574.8 | 1.45× |
| decode peak (c=16/32 × 0) | 406.6 | 1,138.1 | **2.80×** |
| prefill 8k | 5,914 | 7,803 | 1.32× |
| prefill 64k | 4,689 | 6,953 | 1.48× |

TP=2 roughly doubles peak decode throughput and, more importantly, unlocks the high-concurrency ×
high-context cells that TP=1 cannot fit at all. For this model on this hardware, **TP=2 is clearly
the better configuration** when both cards are available — both for raw throughput and for context
capacity.

## Files

- `results/benchmark_results.json` — full harness output
- `logs/run_console_output.txt` — run console with final summary tables
