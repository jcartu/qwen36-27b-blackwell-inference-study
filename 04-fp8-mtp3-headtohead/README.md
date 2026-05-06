# Experiment 4 — FP8+MTP=3 head-to-head

Repne fork vs upstream v0.20.1 on the production config.

| cell | Repne tok/s | Upstream tok/s | Δ |
|---|---|---|---|
| c=1 ctx=0 | **120.1** | 112.3 | +7.0% |
| c=1 ctx=131k | 93.7 | **99.0** | −5.3% |
| c=2 ctx=0 | **223.8** | 197.0 | +13.6% |
| c=2 ctx=131k | 183.4 | **183.9** | −0.3% |
| c=4 ctx=0 | **449.5** | 413.8 | +8.6% |
| c=4 ctx=131k | **347.4** | 345.0 | +0.7% |

Same FP8+MTP=3 config on both. Repne wins on short-context multi-stream (where most agent traffic sits); long context is a wash. Upstream forced to drop `gumbel` (Repne-only) — accepted with the default greedy sampler.

**Files:**
- `repne-fork/c{1,2,4}_ctx{0,131072}.json` + `.log`
- `upstream-v0.20.1/` same layout
