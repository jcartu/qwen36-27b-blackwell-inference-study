# Exp 12 — Qwen3.6-27B decode/prefill matrix on Repne vLLM v17

**Date:** 2026-06-02
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120), TP=2
**Image:** `repne/vllm:v17` — vLLM `0.1.dev17236+g50272be4a.d20260602`
**Harness:** [`voipmonitor/llm-inference-bench`](https://github.com/voipmonitor/llm-inference-bench) `v0.4.24`
**Repository path:** `jcartu/qwen36-27b-blackwell-inference-study/13-v17-decode-matrix`

---

## What this is

A straight decode + prefill sweep on the new **Repne v17** image, run to validate the v17 build
and capture a fresh baseline on the 2-card rig. This is a single-run data point (N=1 per cell),
not a tuned SOTA attempt — see the note below.

## Configuration

```
--model Qwen/Qwen3.6-27B --served-model-name Qwen3.6-27B
--tensor-parallel-size 2
--gpu-memory-utilization 0.80
--max-model-len 262144
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
llm_decode_bench.py --port 8000 --model Qwen3.6-27B --kv-budget 854152
```

KV budget set manually to **854,152 tokens** (engine-reported GPU KV cache size) because
v17's Prometheus endpoint under-reports local KV and does not export the DCP/CP multiplier;
this enables exact over-capacity cell skips (`∅`).

## Results

### Aggregate decode throughput (tok/s)

`∅` = cell skipped, KV budget exceeded.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 86.5 | 172.2 | 320.7 | 574.8 | 865.6 | **1138.1** | 1059.4 | 1067.7 |
| 16k | 92.0 | 179.8 | 337.2 | 573.3 | 859.6 | 1110.2 | ∅ | ∅ |
| 32k | 93.4 | 176.2 | 313.0 | 538.4 | 841.9 | ∅ | ∅ | ∅ |
| 64k | 91.9 | 160.9 | 309.9 | 534.0 | ∅ | ∅ | ∅ | ∅ |
| 128k | 83.6 | 160.7 | 258.9 | ∅ | ∅ | ∅ | ∅ | ∅ |

**Peak: 1,138 tok/s @ c=32 × ctx=0.** Throughput plateaus past c=32 at ctx=0 (c=64/128 do not
improve over c=32), indicating compute saturation at this gpu-mem fraction.

### Prefill throughput

| ctx | tokens | TTFT s | tok/s | N |
|---|---:|---:|---:|---:|
| 8k | 8,192 | 1.05 | 7,803 | 1 |
| 16k | 16,222 | 2.10 | 7,735 | 1 |
| 32k | 32,292 | 4.29 | 7,520 | 1 |
| 64k | 64,415 | 9.26 | 6,953 | 1 |
| 128k | 128,682 | 21.47 | 5,994 | 1 |

### Hardware (sampled during measured cells)

- GPUs: 2× RTX PRO 6000 Blackwell (TP=2)
- Run duration: 19.8 min (529 samples)
- GPU util: avg 81% / max 100%
- VRAM: avg/max ~86 GB per card (~90% of 95.6 GB)
- Power (sum of both GPUs): avg 813 W / max 1,117 W (limit 1,200 W)
- Temp: avg 68°C / max 84°C
- PCIe: rx avg 10,749 MB/s, tx avg 10,500 MB/s

## How this compares to repo SOTA

This run does **not** beat the existing repo SOTA (Exp 08, FP8+MTP=3: 2,083 tok/s @ c=32×0).
It is a different configuration — **v17 image, DFlash N=8 speculative, `--gpu-memory-utilization 0.80`** —
run as a single-pass validation of the new build, so `SOTA.md` and the per-cell records are unchanged.
A tuned v17 sweep (higher gpu-mem fraction, MTP/DFlash axis, N≥2 repeats) would be needed for a
fair SOTA comparison.

## Files

- `results/benchmark_results.json` — full harness output (all cells, percentiles, hardware samples)
- `logs/` — run log
