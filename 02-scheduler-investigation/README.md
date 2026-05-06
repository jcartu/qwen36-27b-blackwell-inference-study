# Experiment 1.5 — Scheduler pessimal-pocket investigation

**Goal:** explain the c=4 ctx=0 −5.7% regression from experiment 01 by perturbing scheduler parameters one at a time.

**Method:** N=3 per variant, c=4 ctx=0 only.

**Variants tested:**
| Variant | Knob | Result vs default (332.6) |
|---|---|---|
| V1a | max-num-seqs=64 | 345.2 ± 7.46 (+3.8%) |
| V1b | max-num-seqs=32 | 343.8 ± 9.93 (+3.4%) |
| V1c | max-num-seqs=256 | 345.2 ± 10.20 (+3.8%) |
| V3a | max-num-batched-tokens=16384 | 340.4 ± 2.02 (+2.3%) |
| V4a | spec-decode disabled | 211.2 ± 0.02 (raw decode floor) |

**Conclusion:** `max-num-seqs=128 + max-num-batched-tokens=32768` is in a specific pessimal scheduling pocket. Almost any deviation recovers throughput. The bimodal TTFT pattern (67-143ms vs 371-375ms split) seen at the default config disappears with seqs=32. **Scheduler-level interaction with spec-decode batching, not a kernel-level bug.**

**Files:** `runs/<variant>/run<N>/` containing 15 raw bench runs.
