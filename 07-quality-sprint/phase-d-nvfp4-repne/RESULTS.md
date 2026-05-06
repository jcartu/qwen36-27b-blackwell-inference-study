# Phase D — NVFP4-MTP rescue attempt on Repne fork

**Hypothesis:** Repne's flags (gumbel sampler, etc) rescued dflash from upstream's catastrophic long-ctx collapse — could they also rescue NVFP4-MTP?

**Result: NO. Repne fork is dramatically WORSE for NVFP4 than upstream v0.20.1.**

## Setup
- Image: `repne/vllm:latest` (`d0a200f77546`, May 5 evening)
- Model: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (same as exp 03)
- Engine: `v0.1.dev16400+g910d87a9d`
- Spec: `--speculative-config.method qwen3_5_mtp --speculative-config.num_speculative_tokens 3 --speculative-config.draft_sample_method gumbel`
- TP=2, max-num-seqs=4, kv-cache-dtype fp8, gpu-mem 0.9
- KV cache: 4,253,882 tokens (vs 4,268,776 on upstream v0.20.1 — within 1%)

## NVFP4-MTP=3 head-to-head

| cell | Upstream v0.20.1 | Repne fork | Δ |
|---|---|---|---|
| c=1 ctx=0 | 109.4 | 51.7 | **−52.7%** |
| c=1 ctx=32k | 101.2 | 50.9 | **−49.7%** |
| c=1 ctx=131k | 55.1 | 50.9 | **−7.6%** |
| c=2 ctx=0 | 206.8 | 93.6 | **−54.7%** |
| c=2 ctx=32k | 178.5 | 81.6 | **−54.3%** |
| c=2 ctx=131k | 105.6 | 42.7 | **−59.6%** |
| c=4 ctx=0 | 416.6 | 175.8 | **−57.8%** |
| c=4 ctx=32k | 277.0 | 158.3 | **−42.9%** |
| c=4 ctx=131k | 7.0 | 2.3 | **−66.9%** |

**Mean: −50.7% on Repne fork.** Spec acceptance dropped to 4-19% (vs 11-30% on upstream). ITL inflated 2-3× (8-25ms → 20-36ms typical, 1066ms outlier at c=4 ctx=131k).

## Why?

The Repne fork's optimizations (custom CUDA kernels, attention backend tuning, gumbel sampling integration) are tuned for **FP8 and BF16** paths. The NVFP4 modelopt path is a different code path that:
- Does NOT benefit from Repne's W8A8 fp8 kernel optimizations
- May be hitting code paths that haven't received the same optimization passes
- Possibly using a less-optimized fallback for nvfp4 matmuls

This is the OPPOSITE of what happened with dflash, where Repne's gumbel/argmax-reduction tweaks rescued the drafter at long ctx.

## Conclusion

**NVFP4-MTP rescue attempt: FAILED.** Stay with the prior verdict from experiment 03: do not promote NVFP4-MTP.

The lesson: Repne's optimizations are quant-specific. Same image, same flags, same model — but the W8A8 FP8 path is heavily optimized while the NVFP4 path is not. Upstream v0.20.1 is actually better for NVFP4.

If you ever wanted NVFP4 in production, you'd run it on upstream vLLM, not Repne. But since FP8+MTP=3 on Repne dominates everything else by a wide margin (Phase 1 of exp 06 confirmed 449.8 tok/s c=4 ctx=0), there's no scenario where NVFP4 wins.

## Caveats

- Single config tested (qwen3_5_mtp method, n=3, gumbel). Maybe a different combination of flags would help, but if Repne's primary optimizations don't apply to NVFP4, no flag tweak will close a 50% gap.
- KV cache size is comparable to upstream (4.25M vs 4.27M), so it's not a memory issue.
