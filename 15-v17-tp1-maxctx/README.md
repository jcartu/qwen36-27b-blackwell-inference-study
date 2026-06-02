# Exp 15 — Qwen3.6-27B TP=1 at max single-card context (mem-util 0.92, ~254k)

**Date:** 2026-06-02
**Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120), **TP=1** (GPU 0 only; GPU 1 idle)
**Image:** `repne/vllm:v17` — vLLM `0.1.dev17236+g50272be4a.d20260602`
**Harness:** [`voipmonitor/llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) `v0.4.24`
**Repository path:** `jcartu/qwen36-27b-blackwell-inference-study/15-v17-tp1-maxctx`

---

## What this is

Follow-up to [Exp 14](../14-v17-tp1-decode-matrix/): same TP=1 single-GPU sweep, but with
`--gpu-memory-utilization` raised to **0.92** to fit as much of the 262k context as the single
card safely allows. Goal was the full 262,144 context; the card tops out a hair short, so this
run serves **253,952 tokens** — ~97% of full context, the practical single-card maximum.

## Finding the safe ceiling (memory walk)

Getting maximum context on one card without crashing took three tries:

| mem-util | max-model-len | Result |
|---|---|---|
| 0.80 | 131072 | Boots, but only 134,606-token KV (Exp 14). Half the context. |
| 0.92 | 262144 | **Startup fails** — 25.91 GiB KV < 26.66 GiB needed for full 262k. |
| 0.94 | 262144 | Boots (273k KV) **but OOMs under decode load** — `torch.OutOfMemoryError`, only 776 MiB free; engine dies mid-bench. |
| **0.92** | **253952** | ✅ **Stable.** 255,042-token KV, survives full concurrent decode. This run. |

**Lesson:** on a single 96 GB card, 0.94 fits the weights+KV at startup but leaves no headroom for
decode-time activations + CUDA graphs + the DFlash drafter forward pass. 0.92 with `max-model-len`
capped to ~254k (plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) is the safe maximum.
Validated with an 8-way concurrent decode load test before the bench: 0 engine errors.

## Configuration

```
--model Qwen/Qwen3.6-27B --served-model-name Qwen3.6-27B
--tensor-parallel-size 1
--gpu-memory-utilization 0.92        # <-- raised from 0.80 (Exp 14)
--max-model-len 253952               # <-- single-card safe max (~97% of full 262144)
--max-num-seqs 128
--max-num-batched-tokens 32768
--max-cudagraph-capture-size 256
--enable-prefix-caching
--speculative-config.method dflash
--speculative-config.model z-lab/Qwen3.6-27B-DFlash
--speculative-config.num_speculative_tokens 8
--speculative-config.draft_sample_method greedy
--attention-backend flashinfer
# env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Bench invocation:

```
llm_decode_bench.py --port 8000 --model Qwen3.6-27B --kv-budget 255042
```

GPU KV cache size: **255,042 tokens** (vs 134,606 at 0.80 in Exp 14 — 1.9× larger).
**23 of 41 cells KV-skipped** (vs 26 at 0.80) — the bigger budget unlocks a few more cells
(c=8×16k, c=16/32/64×0, 128k prefill).

## Results

### Aggregate decode throughput (tok/s)

`∅` = cell skipped, KV budget (255,042 tokens) exceeded.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 57.2 | 123.5 | 218.9 | 375.9 | 640.6 | 631.0 | **642.1** | ∅ |
| 16k | 68.5 | 113.7 | 209.2 | 400.1 | ∅ | ∅ | ∅ | ∅ |
| 32k | 63.9 | 112.9 | 214.3 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 64k | 96.1 | 104.4 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 47.2 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

**Peak: 642 tok/s @ c=64 × ctx=0.** Decode saturates at c≥16 on the single card (640–642 tok/s
flat across c=16/32/64). The bigger KV budget vs Exp 14 mainly buys *capacity* (more cells fit),
not higher peak throughput — peak rose from 407 (Exp 14) to 642 here because c=16/32/64×0 now run
where Exp 14 KV-skipped them.

### Prefill throughput

| ctx | tokens | TTFT s | tok/s | N |
|---|---:|---:|---:|---:|
| 8k | 8,193 | 1.40 | 5,871 | 1 |
| 16k | 16,223 | 2.84 | 5,721 | 1 |
| 32k | 32,293 | 6.05 | 5,341 | 1 |
| 64k | 64,416 | 13.59 | 4,741 | 1 |
| 128k | 128,683 | 33.27 | 3,867 | 1 |

(128k prefill now runs — it was KV-skipped in Exp 14.)

### Hardware (sampled during measured cells)

- GPUs: 1× RTX PRO 6000 Blackwell (TP=1, GPU 0 only; GPU 1 idle)
- Run duration: 14.4 min (384 samples)
- GPU util: avg 43% / max 100%
- VRAM: max 94.8 GB on the active card (~94% of 95.6 GB — near the safe ceiling)
- Power: avg 615 W / max 712 W (single card active)
- Temp: avg 58°C / max 81°C

## TP=1 mem-util comparison (Exp 14 vs this run) + TP=2 reference

| cell | TP=1 @0.80 (Exp 14) | TP=1 @0.92 (this) | TP=2 @0.80 (Exp 13) |
|---|---:|---:|---:|
| decode c=1 × 0 | 67.1 | 57.2 | 86.5 |
| decode c=8 × 0 | 396.7 | 375.9 | 574.8 |
| decode peak × 0 | 406.6 (c=16) | 642.1 (c=64) | 1,138.1 (c=32) |
| prefill 8k | 5,914 | 5,871 | 7,803 |
| prefill 128k | (skipped) | 3,867 | — |
| max context served | 131,072 | **253,952** | 262,144 |
| cells run / 41 | 14 | 18 | ~26 |

Raising mem-util on the single card primarily extends **context capacity** (131k → 254k) and
unlocks more matrix cells; per-cell decode throughput is roughly comparable (slightly lower at
low concurrency due to less free SM headroom). TP=2 remains the throughput and capacity winner.

## Files

- `results/benchmark_results.json` — full harness output
- `logs/run_console_output.txt` — run console with final summary tables
