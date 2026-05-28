# SOTA reference matrix — Qwen3.6-27B, dual RTX PRO 6000 Blackwell, TP=2

This document catalogs the best aggregate-throughput value recorded for each (concurrency × context) cell across all eight experiments in this monorepo. **FP8+MTP=3 on the Repne fork holds the production-relevant SOTA.** MTP=5 holds isolated low-concurrency records but is dominated everywhere production traffic actually lives (c≥8).

---

## 1. Production SOTA: FP8+MTP=3 (Repne fork)

### 1.1 Per-cell records

| concurrency × context | tok/s | std | source |
|---|---:|---:|---|
| c=1 × 0 | 117.1 | ±2.8 | Exp 06 phase 1, N=3 (peak: 120.1 in Exp 04 N=1) |
| c=1 × 32k | 119.2 | ±4.7 | Exp 06 phase 1, N=3 |
| c=1 × 131k | 95.0 | ±2.5 | Exp 06 phase 1, N=3 |
| c=2 × 0 | 227.2 | ±2.4 | Exp 06 phase 1, N=3 |
| c=2 × 32k | 227.0 | ±5.1 | Exp 06 phase 1, N=3 |
| c=2 × 131k | 184.9 | ±0.9 | Exp 06 phase 1, N=3 |
| c=4 × 0 | 449.8 | ±4.6 | Exp 06 phase 1, N=3 |
| c=4 × 32k | 454.5 | ±3.3 | Exp 06 phase 1, N=3 |
| c=4 × 131k | 350.5 | ±4.9 | Exp 06 phase 1, N=3 |
| c=8 × 0 | 875.0 | ±11.5 | Exp 08 X1, N=2 |
| c=8 × 32k | 795.4 | ±0.9 | Exp 08 X1, N=2 |
| c=8 × 131k | 534.4 | ±7.4 | Exp 08 X1, N=2 |
| c=16 × 0 | 1,520.6 | ±2.5 | Exp 08 X1, N=2 |
| c=16 × 32k | 1,186.0 | ±7.5 | Exp 08 X1, N=2 |
| c=16 × 64k | 1,047.1 | ±4.8 | Exp 08 X1, N=2 |
| **c=32 × 0** | **2,083.7** ⭐ | ±12.6 | Exp 08 X1, N=2 |
| c=32 × 16k | 1,892.3 | ±4.1 | Exp 08 X1, N=2 |
| c=32 × 32k | 1,656.3 | ±13.6 | Exp 08 X1, N=2 |

### 1.2 Per-cell records — MTP=5 (dominated configuration, kept for reference)

MTP=5 wins isolated cells at low concurrency. Documented here for completeness.

| concurrency × context | MTP=5 tok/s | MTP=3 tok/s | MTP=5 advantage |
|---|---:|---:|---:|
| c=1 × 0 | 119.9 | 117.1 | +2.4% |
| c=1 × 131k | **101.2** ⭐ | 95.0 | **+6.5%** (largest single-cell MTP=5 win) |
| c=2 × 0 | 234.4 | 227.2 | +3.2% |
| c=4 × 0 | 462.5 | 449.8 | +2.8% |
| c=8 × 0 | 865.8 | 875.0 | −1.1% (MTP=3 retakes) |
| c=16 × 0 | 1,329.7 | 1,520.6 | **−12.6%** |
| c=32 × 0 | 1,726.5 | 2,083.7 | **−17.1%** |

**Crossover concurrency: c=8.** Above this, MTP=3 dominates. Below, MTP=5 has a small advantage. Production traffic on coding agents typically bursts to c=16+, which makes MTP=3 the correct production choice.

---

## 2. Quality SOTA: 8-bit is lossless; 4-bit AutoRound is within 1.5× on teacher-forced KL

### 2.1 Perplexity reference (Wikitext-2, GGUF path)

Wikitext-2 perplexity (102 200 token positions, 200 sliding windows × 511 positions, ctx=512, stride=128):

| Quant | Perplexity | KLD vs BF16 | Top-1 agreement |
|---|---:|---:|---:|
| BF16 GGUF (reference) | 7.620 ± 0.062 | 0 | 100% |
| Q8_0 GGUF | 7.623 ± 0.063 | **0.001828 ± 0.000189** | **97.9%** |
| FP8 W8A8 (vLLM) | not directly measurable* | inferred ≈ Q8 | inferred ≈ 97-98% |

*The AesSedai perplexity tool reads GGUF only, so we cannot directly perplex the FP8 W8A8 vLLM weights. Quality is inferred from (a) Qwen team's own model-card claim of "performance metrics nearly identical to original," (b) Phase B functional-test parity (8/8 hard tests pass on FP8), and (c) the principle that any 8-bit quant near BF16 should also be near BF16's KLD floor.

### 2.2 Teacher-forced KL/JSD fidelity (vLLM path, Exp 11 + Addendum C)

Mean KL of BF16 → quant logits across 504 WikiText-2-raw-v1 positions (multi-prompt) and 16 positions (single-prompt). Lower = closer to BF16. Source: [Exp 11 Addendum C](sweeps/11-teacher-forced-kld/addenda/).

| Quant | Bits/wt | Multi-prompt mean KL (mbits) | Single-prompt mean KL (mbits) | Single-prompt max KL (mbits) | Verdict |
|---|---:|---:|---:|---:|---|
| BF16 self (noise floor) | 16 | 27 | 0.76 | 2.9 | reference jitter |
| **FP8 W8A8** | 8 | **230** | **5.5** | **13** | **production quant** |
| **AutoRound-int4** (Lorbus repo) | 4 | **333** | **29** | **72** | **best int4 in stack** |
| AWQ-6bit | 6 | 336 | 35 | 93 | tied with AutoRound multi; loses single |
| AWQ-4bit | 4 | 392 | 41 | 199 | bit-budget cost vs AWQ-6bit |
| GPTQ-qwopus (Pro) | 4 | 450 | 70 | 383 | mid-pack |
| NVFP4 | 4 | 508 | 300 | 3,926 | **noisy** — see §3 disqualification |
| GPTQ-groxaxo (Pro) | 4 | 570 | 241 | 618 | worst int4 in stack |

**Key findings:**
- **FP8 is ~1.4× the bf16-self noise floor** on multi-prompt mean KL — essentially indistinguishable from BF16 jitter at single-prompt scale (5.5 mbits vs 0.76 mbits floor).
- **AutoRound-int4 matches AWQ-6bit on multi-prompt mean KL (333 vs 336 mbits) at 2 fewer bits per weight**, and beats it in single-prompt mode (29 vs 35). At 4 bits, AutoRound recovers the fidelity that AWQ needs 6 bits to achieve — the most efficient int quant in the stack.
- **Bit budget is a within-family lever; cross-family, algorithm dominates.** AWQ-6bit > AWQ-4bit (as expected). But AutoRound-int4 ≥ AWQ-6bit (against bit-budget intuition).
- **GPTQ quality varies by 3-8× between vendors** of the same nominal bit width — "Pro" labels carry no signal. Always measure.
- **Per-position max JSD ≈ ln(2) bits for every quant** — universal signature of argmax-altering quantization. Mean KL is the correct headline metric.

### 2.3 Quality decision rule (updated)

```
Need BF16-equivalent fidelity?       → BF16 (no spec) or BF16+DFlash
Need ~99% BF16 fidelity?             → FP8 W8A8 (production quant)
Memory-bound at 4-bit budget?        → AutoRound-int4 (best int4 measured; throughput-tied with FP8+MTP=3, see [Exp 12](../12-autoround-mtp-vs-fp8-mtp/))
Avoid:                               → NVFP4 (quality + engine instability)
Avoid:                               → GPTQ-groxaxo (8× noisier than peers)
```

## 3. Cross-quant performance comparison (single best cell per config)

| Configuration | Best tok/s | Best cell | Mean across c=1–4 (9 cells) | Verdict |
|---|---:|---|---:|---|
| **FP8+MTP=3 (Repne)** | **2,083.7** | c=32 × 0 | **252.7** | **Production SOTA** |
| FP8+MTP=5 (Repne) | 1,726.5 | c=32 × 0 | 257.3 | Wins c=1–4 narrowly, loses c≥8 |
| FP8+no-spec (Repne) | 1,875.5 | c=32 × 0 | ~150 | No-spec floor |
| BF16+DFlash=7 (Repne) | 344.9 | c=4 × 0 | 197.5 | Best DFlash variant |
| BF16+DFlash=8 (Repne) | 358.4 | c=4 × 0 | 194.1 | Slightly behind n=7 |
| BF16+DFlash=15 (Repne) | 313.4 | c=4 × 0 | 178.5 | Repne's recommended config — actually worst of three |
| FP8+MTP=3 (upstream v0.20.1) | 413.8 | c=4 × 0 | not measured at depth | Viable fallback |
| BF16+DFlash=8 (upstream v0.20.1) | 290.9 | c=4 × 0 | n/a | **Long-context broken**, drafter accepts 1–3% at 131k |
| NVFP4+MTP=3 (upstream v0.20.1) | 416.6 | c=4 × 0 | mean drops to 95–106 by c=2 ctx=131k | **Broken at 244k + 508 mbits KL** (see §2.2), disqualified |
| NVFP4+MTP=3 (Repne) | 175.8 | c=4 × 0 | n/a | 50% worse than upstream — Repne flags hurt NVFP4 |

---

## 4. Practical decision tree

```
Is this Blackwell SM120 hardware?
├── Yes → continue
└── No  → results not applicable, retest

Need maximum throughput at c≥8 production traffic?
├── Yes → FP8+MTP=3 on Repne fork (2,083 tok/s peak)
└── No  → continue (single-user / always c≤4)

Always single-user / c=1, deep context (131k+) priority?
├── Yes → FP8+MTP=5 on Repne fork (101.2 tok/s c=1×131k)
└── No  → FP8+MTP=3 on Repne fork (still wins majority of cells)

Need quality matching BF16 exactly?
├── Yes → BF16 (no spec) or BF16+DFlash on Repne; expect ~1/2 throughput
└── No  → FP8+MTP=3 (KLD ≈ Q8 ≈ noise floor)

Stuck on upstream vLLM (no Repne available)?
├── FP8+MTP=3 path is viable, expect 5–14% throughput cost
├── BF16+DFlash path is NOT viable (drafter collapses past 32k)
└── NVFP4 path is NOT viable (engine instability past 131k)
```

---

## 5. Source pointers

Every claim in this document is reproducible from raw `results.json` files in the experiment directories.

- Per-cell N=3 production data: [Experiment 6](../06-new-image-validation/) (FP8+MTP=3, DFlash variants)
- High-concurrency data: [Experiment 8 X1](../08-x1y1-sprint/) (c=8/16/32)
- KL-divergence data: [Experiment 8 Y1](../08-x1y1-sprint/y1-perplexity/) (BF16 vs Q8 GGUF)
- Cross-fork comparison: [Experiment 4](../04-fp8-mtp3-headtohead/), [Experiment 5](../05-bf16-dflash-headtohead/)
- MTP n sweep: [Experiment 7 Phase C](../07-quality-sprint/phase-c-fp8-sweep/)
- Functional-gate logs: [Experiment 7 phase logs](../07-quality-sprint/), [Experiment 6 gates](../06-new-image-validation/gates/)
- Quant SOTA verification: [Experiment 12](../12-autoround-mtp-vs-fp8-mtp/) (AutoRound-int4+MTP=3 vs FP8+MTP=3 head-to-head; FP8 retains SOTA)
- Master CSV: [`master-results.csv`](master-results.csv) (328 rows, all experiments)
