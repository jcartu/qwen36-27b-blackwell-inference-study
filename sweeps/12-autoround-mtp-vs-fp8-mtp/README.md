# Exp 12 — AutoRound-int4 + MTP=3 vs FP8 + MTP=3 (Diagnostic)

**Date:** 2026-05-28
**Image:** `repne/vllm:v13`
**Hardware:** 3× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120); display reserved on GPU 0, compute on GPU 1+2 (TP=2)
**Model under test (int4):** `Lorbus/Qwen3.6-27B-int4-AutoRound`
**Baseline (FP8):** `Qwen/Qwen3.6-27B-FP8`
**Speculative decoding:** MTP, num_spec_tokens=3 on both variants
**Methodology:** identical to SOTA cells — `llm_decode_bench.py --duration 60 --decode-warmup-seconds 20 --skip-prefill`, N=2 reps per cell

---

## Question

Does AutoRound-int4 + MTP=3 dethrone FP8 + MTP=3 as the throughput SOTA on Blackwell?

The motivating evidence: AutoRound-int4 has the lowest teacher-forced KL of any int4 quant we tested (35 mbits vs NVFP4's 508 mbits — see SOTA.md §2.2), so quality is no longer a blocker. The remaining question is **whether int4 weights + MTP can match or beat FP8 + MTP at production-realistic throughput** when both run under the same vLLM SOTA configuration.

## Cells

Two diagnostic cells covering the two regimes where the two quants might diverge:

| Cell | Concurrency | Context | Why this cell |
|---|---|---|---|
| **A** | c=32 | ctx=0 (~1k tokens) | Peak throughput regime. SOTA published cell. If int4 wins anywhere on raw compute, it wins here. |
| **B** | c=16 | ctx=131k | KV-pressure regime. Long contexts + many sequences. If int4's smaller weights free up KV cache space, this is where it pays off. |

Each cell measured with N=2 reps under identical config.

## Configuration (both variants)

```
--tensor-parallel-size 2
--gpu-memory-utilization 0.88
--max-model-len 262144
--max-num-seqs 128
--max-num-batched-tokens 32768
--max-cudagraph-capture-size 256
--speculative-config '{"method":"mtp","num_spec_tokens":3}'
```

Source-of-truth launcher: `/home/josh/vllm-services/launch-qwen36-27b-tp2-sota.sh`
This experiment's launchers (`scripts/launch_fp8_mtp.sh`, `scripts/launch_autoround_mtp.sh`) differ only in port (8765) and GPU IDs (1+2) for harness isolation.

## Results

```
Cell                   Variant           rep1      rep2       mean    ±std   per-user
─────────────────────────────────────────────────────────────────────────────────────
c=32 × ctx=0           fp8             2145.2    2119.1     2132.2    18.5       66.5
c=32 × ctx=0           autoround       2093.4    2076.7     2085.0    11.8       61.1
                       Δ (AR vs FP8)                          -47.1         (-2.21%)

c=16 × ctx=131k        fp8              713.8     726.1      720.0     8.7       40.1
c=16 × ctx=131k        autoround        743.3     747.6      745.5     3.0       40.1
                       Δ (AR vs FP8)                          +25.5         (+3.54%)
```

Per-cell raw JSON in `results/{fp8,autoround}/`.

### Cross-checks

- **FP8 baseline reproduces SOTA.md.** Published FP8 c=32×0 = 2,083.7 ±12.6 (Exp 08 X1). Our measurement: 2,132.2 ±18.5 — within +2.3%, on the high side of the published spread (consistent with warm torch.compile + AOT cache from the prod launcher staying primed).
- **c=16 × ctx=131k is a new cell.** SOTA.md only publishes c=16×64k (1,047.1) and c=8×131k (534.4). This is the first measurement at the c=16×131k operating point.
- **MTP confirmed active on both runs.** vLLM logged `Detected MTP model. Sharing target model embedding/lm_head weights with the draft model` for both variants. AutoRound's litmus run (separate, c=1) showed 70% avg draft acceptance, 84% at position 0, mean acceptance length 3.09 — confirming the MTP head materially contributes.

## Verdict

**AutoRound-int4 + MTP=3 does not dethrone FP8 + MTP=3 as throughput SOTA on Blackwell.**

- Peak throughput (c=32 × ctx=0): **FP8 +2.2% over AutoRound** — within run-to-run noise of comparable scale (FP8 ±0.9%, AR ±0.6%), so the practical read is "tie, slight edge to FP8."
- Long-context (c=16 × ctx=131k): **AutoRound +3.5% over FP8** — small but consistent (AR's stdev is 3× tighter than FP8's at this cell). Likely a real signal from reduced weight-memory footprint freeing KV bandwidth, though within a single-experiment confidence band.

**Net:** AutoRound-int4 + MTP=3 is **competitive but not superior** to FP8 + MTP=3 on this Blackwell setup. The hypothesized memory-savings win does not materialize at the SOTA configuration because `gpu-memory-utilization=0.88` consumes the full KV budget on both variants regardless of weight footprint (FP8 uses 14.3 GiB/rank for weights, AutoRound uses ~6 GiB/rank for weights, but both end at ~85 GiB/rank total GPU usage because vLLM expands the KV pool to fill available memory).

The +3.5% at long context is the only cell where AutoRound wins; the −2.2% at peak throughput cancels it. **No SOTA change recommended.**

### When AutoRound-int4 + MTP=3 would matter

- **Memory-constrained deployments:** if you can't set `gpu-memory-utilization=0.88` (e.g., shared GPU, or smaller cards), AutoRound's ~6 GiB/rank weights leave dramatically more room for KV cache than FP8's 14.3 GiB/rank.
- **Lower TP configurations:** TP=1 on a single Blackwell would be infeasible for FP8 27B + KV at long context; AutoRound makes it plausible.
- **Single-stream latency-sensitive workloads:** AutoRound's c=1 = 118.0 tok/s vs FP8's 117.1 tok/s (litmus result, separate config) — basically identical, but with much smaller memory footprint as bonus.

For peak multi-tenant throughput on the current 3×Blackwell production setup, FP8 + MTP=3 remains SOTA.

## Files

```
sweeps/12-autoround-mtp-litmus/
├── README.md                              (this file)
├── scripts/
│   ├── launch_fp8_mtp.sh                  FP8+MTP=3 SOTA-parity launcher
│   ├── launch_autoround_mtp.sh            AutoRound+MTP=3 SOTA-parity launcher
│   ├── wait_health.sh                     /health poller, 900s cap
│   └── run_cells.sh <variant>             2 cells × 2 reps via llm_decode_bench.py
└── results/
    ├── fp8/                               4 JSONs + 4 logs
    └── autoround/                         4 JSONs + 4 logs
```

## Reproduction

```bash
# FP8 baseline
bash scripts/launch_fp8_mtp.sh
bash scripts/wait_health.sh vllm-exp12-fp8-mtp 8765 900
bash scripts/run_cells.sh fp8
docker stop vllm-exp12-fp8-mtp && docker rm vllm-exp12-fp8-mtp

# AutoRound
bash scripts/launch_autoround_mtp.sh
bash scripts/wait_health.sh vllm-exp12-autoround-mtp 8765 900
bash scripts/run_cells.sh autoround
docker stop vllm-exp12-autoround-mtp && docker rm vllm-exp12-autoround-mtp
```
