# Performance characterization — Qwen3.6-27B FP8+MTP=3 (Repne fork)

## Aggregate throughput vs concurrency

```
Throughput (tok/s) at ctx=0:

c=1   ▏ 117  ████
c=2   ▏ 227  ████████
c=4   ▏ 450  ████████████████
c=8   ▏ 875  █████████████████████████████████
c=16  ▏ 1521 █████████████████████████████████████████████████████████
c=32  ▏ 2084 ████████████████████████████████████████████████████████████████████████████ ⭐

Throughput (tok/s) at ctx=131k:

c=1   ▏  95  ████
c=2   ▏ 185  ████████
c=4   ▏ 351  ███████████████
c=8   ▏ 534  ███████████████████████
```

## Speedup vs no-spec baseline

| concurrency | no-spec | MTP=3 | MTP=3 speedup |
|---|---:|---:|---:|
| c=8 ctx=0 | 573.6 | 875.0 | **1.53×** |
| c=16 ctx=0 | 1,126.7 | 1,520.6 | **1.35×** |
| c=32 ctx=0 | 1,875.5 | 2,083.7 | **1.11×** |

The MTP=3 advantage compresses as concurrency rises (less spare GPU compute to amortize spec misses) but remains positive everywhere we measured.

## Speculative-decoding `n` sweep (FP8+MTP, Repne fork)

```
Mean throughput across 9 cells (c=1,2,4 × ctx=0,32k,131k):

n=2   ▏ ~245  ████████████████████████████████████████████ (mini-matrix only)
n=3   ▏ 252.7 █████████████████████████████████████████████ (production winner)
n=4   ▏ ~245  ████████████████████████████████████████████ (mini-matrix only)
n=5   ▏ 257.3 ████████████████████████████████████████████  (wins c=1-4, loses c=8+)
n=6   ▏ ~240  ███████████████████████████████████████████ (mini-matrix only)
```

Crossover concurrency for MTP=3 vs MTP=5 advantage: **c=8**. Below: MTP=5 by ~2-7%. Above: MTP=3 by 10-20%.

## DFlash `n` sweep (BF16, Repne fork)

```
Mean throughput across 9 cells:

n=7   ▏ 197.5 ████████████████████████████████████ ⭐ (best DFlash)
n=8   ▏ 194.1 ███████████████████████████████████  (Repne's default)
n=15  ▏ 178.5 █████████████████████████████████   (overshooting)
```

All DFlash variants are ~25-50% slower than FP8+MTP=3 at the same concurrency. DFlash never reaches FP8+MTP=3 throughput in any cell tested.

## Quality: KLD distribution (Q8_0 vs BF16)

```
Mean    KLD : 0.001828 ± 0.000189
Median  KLD : 0.000559
95th    KLD : 0.002880
99th    KLD : 0.008801
99.9th  KLD : 0.079328
Maximum KLD : 10.957  (rare token-position outlier)

Token probability shift Δp:
  Mean    Δp :  0.011 %
  Median  Δp :  0.000 %
  95th    Δp :  1.392 %
  99th    Δp :  2.737 %
  RMS     Δp :  1.388 ± 0.081 %

Same top-token agreement: 97.90 ± 0.045 %
```

This is the KL-divergence floor for 8-bit weight quantization on this model. By transitive inference (Qwen team's claim + functional-test parity + this measurement), FP8 W8A8 falls at approximately the same noise floor.

## Architecture under test

```
   Host: Linux 7.0.2-arch1-1 / Intel Xeon W (W790E-SAGE)
                          │
                          │ PCIe Gen5 x16 (per GPU, verified under load)
                ┌─────────┼─────────┐
                │                   │
        ┌───────▼───────┐   ┌───────▼───────┐
        │ RTX PRO 6000  │   │ RTX PRO 6000  │
        │   Blackwell   │   │   Blackwell   │
        │   SM120       │   │   SM120       │
        │   96 GiB      │   │   96 GiB      │
        └───────┬───────┘   └───────┬───────┘
                │                   │
                └─── TP=2, FP8 ─────┘
                         │
                  vLLM (Repne fork)
                  v0.1.dev16400+g910d87a9d
                         │
              FP8 weights, BF16 KV
              MTP n=3 speculative decoding
              gumbel draft sampling
              flashinfer attention
              max_seq_len = 262 144 tokens
                         │
                  Port 11435
                  ROCK SOLID config in opencode
```

## 24-hour sprint timeline

```
  May 5 morning   ┃  Exp 1 — Image validation (BF16+DFlash, N=5×5 randomized)
  May 5 midday    ┃  Exp 2 — Scheduler pessimal-pocket investigation
  May 5 afternoon ┃  Exp 3 — NVFP4-MTP viability (dies at 244k)
  May 5 evening   ┃  Exp 4 — FP8+MTP=3 vs upstream v0.20.1
                  ┃  Exp 5 — BF16+DFlash vs upstream v0.20.1 (5-6× win)
  May 5 late      ┃  Exp 6 — New image (d0a200f7) FP8 vs DFlash variants
  May 6 overnight ┃  Exp 7 — Phaelon-triggered quality sprint (4 phases)
  May 6 dawn      ┃  Exp 8 — High-concurrency + AesSedai KLD perplexity ⭐
                  ┃           Peak 2,083 tok/s recorded
                  ┃           ROCK SOLID reverted to MTP=3 (correct answer)
```

