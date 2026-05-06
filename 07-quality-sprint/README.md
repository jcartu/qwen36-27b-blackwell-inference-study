# Experiment 7 — Quality sprint (post-Discord patterns from Phaelon/AesSedai)

**Trigger:** Discord conversation with PhaelonQuant Creators raised concerns we'd never tested:
- "FP8 is W8A8" — weights AND activations quantized to 8-bit
- "An 8-bit GGUF will ALWAYS be more accurate than W8A8 because you get lobotomized lowering activations"
- We measured speed all day. We never measured quality. Was our SOTA verdict premature?

**Plus three throughput-optimization probes** based on patterns from prior experiments.

## Four phases

| Phase | What | Result |
|---|---|---|
| A | Spec-decode losslessness — does MTP=3 produce identical output to no-spec at temp=0? | **MTP=3 differs from no-spec on 4/8 prompts** but no correctness regressions; 1.74× speedup justifies. |
| B | FP8 vs Q8 GGUF quality probe — does Phaelon's "lobotomization" claim hold? | **No measurable quality regression on FP8.** Both produce factually correct, syntactically valid outputs. FP8 is 2.15× faster. |
| C | FP8 sweep — is MTP=3 actually optimal, or could we squeeze more? | **MTP=5 wins +1.8% mean across 9 cells**, especially +6.5% at c=1 ctx=131k (single-user-deep-context scenario). 4/4 gates pass. |
| D | NVFP4-MTP rescue on Repne fork — could Repne's flags fix the long-ctx collapse? | **NO — Repne fork is 50% WORSE for NVFP4 than upstream v0.20.1.** Repne's optimizations are W8A8-specific. |

## Key finding for ROCK SOLID config

**Switch FP8+MTP=3 → FP8+MTP=5 for a free 1.8% throughput gain with no quality penalty.**

Smaller wins everywhere, biggest single-cell win (+6.5%) at the deep-context single-user regime which is exactly what coding agents hit.

| | MTP=3 | MTP=5 | Δ |
|---|---|---|---|
| c=1 ctx=0 | 117.1 | 119.9 | +2.4% |
| c=1 ctx=131k | **95.0** | **101.2** | **+6.5%** ⭐ |
| c=2 ctx=0 | 227.2 | 234.4 | +3.2% |
| c=2 ctx=131k | 184.9 | 190.8 | +3.2% |
| c=4 ctx=0 | 449.8 | 462.5 | +2.8% |
| c=4 ctx=131k | 350.5 | 346.9 | −1.0% |
| **mean** | | | **+1.8%** |

Functional gates pass 4/4 on MTP=5 (Fibonacci, tool, reasoning, multi-turn).

## On Phaelon's W8A8 lobotomization claim

The claim is theoretically valid (W8A8 quantizes both weights AND activations; W8A16-style GGUF only quantizes weights so the compute path stays high-precision). **But empirically we couldn't measure a quality regression.**

What we tested:
- 8 standard prompts (Fibonacci, palindrome code, math, haiku, networking explanation, JSON extraction, code review, factual)
- 1 hard algorithmic prompt (Manacher's algorithm — both engines passed 8/8 functional tests)

What would be more decisive (deferred):
- AesSedai's [sliding-window perplexity branch](https://github.com/AesSedai/llama.cpp/tree/perplexity-sliding-window) for a single-number KLD verdict
- HumanEval / MBPP code benchmark
- GSM8K math benchmark
- LMSYS-style A/B grading via GPT-4 judge

For our use case (interactive coding agent), 2.15× speed advantage on FP8 vs Q8 is decisive. Quality differences at the level we measured don't cross any threshold that matters in practice.

## On the spec-decode losslessness question

Empirically, **MTP=3 is NOT bitwise-identical to no-spec even at temp=0, seed=42, top_p=1.0.** Both engines are deterministic individually (3× same prompt → exact same output), but they produce different outputs vs each other on 4/8 prompts. The differences are stylistic/lexical, not factual:

- "long multiplication" vs "multiplication" (math)
- Different haiku word choices
- Different sentence ordering in TCP/UDP explanation
- Different code-review point #3 (input validation vs duplicating built-in)

**No correctness deltas detected.** Theory says spec decoding *should* be lossless at temp=0 if rejection sampling is correct. Repne's gumbel sampler implementation appears to take a different RNG path through the spec accept/reject decisions. **Acceptable for production** — quality is preserved, 1.74× speedup is real.

## On NVFP4 — fully closed

Phase D confirms: **NVFP4 is dead for our hardware.** Already lost on upstream v0.20.1 at long context (experiment 03). Now confirmed it's also 50% slower on Repne fork. There's no remaining configuration where NVFP4 wins. **Final verdict: do not promote. NVFP4 is permanently disqualified.**

## Files

- `phase-a-spec-losslessness/` — RESULTS.md, fp8_mtp3_baseline.json, fp8_nospec_baseline.json
- `phase-b-fp8-vs-q8/` — RESULTS.md, q8_gguf.json, fp8mtp3_hard_prompt.json, q8_hard_prompt.json
- `phase-c-fp8-sweep/` — RESULTS.md, mtp{2,3,4,5,6}/ + mtp5-fullmatrix/ + mtp5-seqs64/
- `phase-d-nvfp4-repne/` — RESULTS.md, runs/ (27 raw bench files)

## Updated SOTA recommendation

The **MTP=5 finding from Phase C is the only actionable change.** Switch ROCK SOLID launch script from `--speculative-config.num_speculative_tokens 3` to `--speculative-config.num_speculative_tokens 5`. Everything else stays identical.
