# Experiment 5 — BF16+DFlash head-to-head

Repne fork vs upstream v0.20.1 on speculative dflash decoding.

| cell | Repne tok/s | Upstream tok/s | Δ |
|---|---|---|---|
| c=1 ctx=0 | **98.9** | 85.5 | +15.6% |
| c=1 ctx=131k | **81.4** | 11.7 | **+598%** 🚨 |
| c=2 ctx=0 | **184.5** | 153.9 | +19.9% |
| c=2 ctx=131k | **162.7** | 22.4 | **+626%** 🚨 |
| c=4 ctx=0 | **358.4** | 290.9 | +23.2% |
| c=4 ctx=131k | **284.4** | 44.7 | **+537%** 🚨 |

**Upstream's dflash collapses at long context.** Drafter acceptance falls from 16-36% (Repne) to 1-3% (upstream) at 131K. ITL inflates 8× (11ms → 90ms per token).

**Why:** dflash drafter requires non-causal attention. flashinfer rejects this with `ValueError: ['non-causal attention not supported']`, forcing upstream onto flash_attn for the main path. Upstream also lacks Repne's `gumbel` draft sampler, `use_local_argmax_reduction`, and per-spec attention backend setting.

For BF16+DFlash production, **upstream is not a viable path.** Stay on Repne fork.

**Files:**
- `repne-fork/c{1,2,4}_ctx{0,131072}.json` + `.log`
- `upstream-v0.20.1/` same layout
