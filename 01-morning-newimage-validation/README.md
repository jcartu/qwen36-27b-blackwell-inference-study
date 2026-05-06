# Experiment 1 — Morning new-image validation

**Goal:** validate the new Repne `repne/vllm:latest` (`5e7583ca4df9`, May 5 2026) image against the prior baseline (`f0e7574`, May 3) for production use.

**Method:** N=5 randomized order across 5 representative cells (2σ verdict bands).

**Headline:**
- ✅ c=1 ctx=128k: +9.0% **WIN**
- ✅ c=4 ctx=128k: +6.6% **WIN**
- 🔴 c=4 ctx=0: −5.7% **REGRESSION** (triggered halt protocol)
- 🔴 c=2 ctx=250k: −16.3% **REGRESSION**
- ≈ c=2 ctx=244k: −9.0% but within noise band (NOISE)

The c=4 ctx=0 regression triggered the sprint-spec halt and pivoted to scheduler investigation (experiment 02). Later in the day, single-shot c=4 ctx=0 hit 449.5 tok/s — likely transient state.

**Files:**
- `RESULTS.md` — full verdict table with N=5 per cell, ±2σ bands
- `runs/<cell>_run<N>/results.json` + `bench.log` — 25 raw bench runs
- `run_plan.txt` — randomized run order (seed=42)
- `driver.log` — driver stdout

**Bench config:** BF16, dflash spec method, num_speculative_tokens=8, gumbel sampler, 30s + 10s warmup, --skip-prefill.
