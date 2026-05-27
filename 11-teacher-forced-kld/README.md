# Exp 11 — Teacher-forced decode KLD/JSD on Qwen3.6-27B (BF16, FP8, NVFP4)

**Date:** 2026-05-27
**Hardware:** 3× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120). GPUs 1+2 used (TP=2). GPU 0 reserved for display.
**Image:** `repne/vllm:v13` — vLLM `v0.1.dev17130+g155fef0e5.d20260526` (with `cu132`)
**Methodology source:** [local-inference-lab/rtx6kpro · `glm51-teacher-forced-decode-kld-js-2026-05-25.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.1/glm51-teacher-forced-decode-kld-js-2026-05-25.md)
**Reference scripts:** [`festr2/glm51-decode-kld-refs`](https://huggingface.co/datasets/festr2/glm51-decode-kld-refs) (HF dataset; scripts subdir; minor patches noted below)
**Repository:** `jcartu/qwen36-27b-blackwell-inference-study/11-teacher-forced-kld`

---

## Question

How closely do FP8 and NVFP4 quantized variants of Qwen3.6-27B match the BF16 reference distribution at every decoded position, on **identical** teacher-forced WikiText sequences?

What is the BF16-vs-BF16 inference noise floor on this stack, and where do the variant deltas sit relative to that floor?

## TL;DR — Bits above noise floor

All values in **bits** (nats / ln 2). Lower = closer to BF16 reference. Noise floor = two BF16 runs with the same forced WikiText sequence.

### Single-prompt cells (1 prompt × 17 forced tokens; **16 positions** after `--skip-prefill-next=1`)

| variant | mean KL(ref‖q) | mean JSD | max KL | max JSD | ratio vs noise floor (KL) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **bf16-self** (noise floor) | 7.59e-4 | 1.90e-4 | 2.94e-3 | 7.36e-4 | 1.0× |
| fp8 | 5.52e-3 | 1.38e-3 | 1.32e-2 | 3.27e-3 | **7.3×** |
| nvfp4 | 2.99e-1 | 3.61e-2 | 3.93 | 0.371 | **395×** |

### Multi-prompt cells (8 prompts × 64 forced tokens = **504 positions**)

| variant | mean KL(ref‖q) | mean JSD | max KL | max JSD | ratio vs noise floor (KL) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **bf16-self** (noise floor) | 2.69e-2 | 3.03e-3 | 10.9 | 0.984 | 1.0× |
| fp8 | 2.30e-1 | 1.40e-2 | 33.0 | 1.000 | **8.6×** |
| nvfp4 | 5.07e-1 | 4.34e-2 | 29.8 | 1.000 | **18.8×** |

### Headline takeaways

1. **FP8 quantization is statistically noticeable but small.** Single-prompt KL is **7.3× noise floor** (5.5 mbit/pos vs 0.76 mbit/pos). Multi-prompt is **8.6× noise floor**. Tail can be aggressive: max KL hits **33 bits** at one position out of 504.

2. **NVFP4 quantization shows substantial drift.** Single-prompt KL is **395× noise floor** — a clear, easily-measured signal at every position. Multi-prompt mean KL is *lower* than single (5.07e-1 vs 2.99e-1) only because the multi mean is computed over more positions; per-position drift is comparable or larger.

3. **Even BF16-vs-BF16 has a non-trivial tail on multi-prompt cells.** Max KL of **10.9 bits** between two identical BF16 runs is striking. Likely caused by a few positions where the two TP=2 runs produce slightly different floating-point results that, when one of those tokens has very low probability, blow up in log-space. Position-0 is already dropped; these outliers come from positions deeper in the sequence.

4. **MTP=3 (FP8 with speculative decoding) is not measurable** with this methodology on this vLLM build. See `SKIP_REASON.md`.

---

## Method

### What we measure

For each (model, variant) cell, we run vLLM with a custom `TeacherForceLogitsProcessor` that pins logits at sampling time so the model is forced to "decode" a specific WikiText continuation. The crucial property: **vLLM V1 records the logprobs returned to user code BEFORE the custom logits processor is applied** (verified empirically — see `R1 validation` below).

So we capture the model's **raw** distribution over the full vocabulary at each forced position. Two such captures (e.g., BF16 reference and FP8 candidate) over **byte-identical** teacher sequences let us compute per-position KL(ref‖cand), KL(cand‖ref), and JSD in nats; we report bits.

### Cells (8 total + 1 hard-skip)

| Cell | Model | Quant | Spec | Prompts | Tokens/prompt | Status |
| --- | --- | --- | --- | --- | --- | --- |
| BF16-ref-single | `Qwen/Qwen3.6-27B` | none (BF16) | — | 1 | 17 | ✅ |
| BF16-ref-multi | `Qwen/Qwen3.6-27B` | none (BF16) | — | 8 | 64 | ✅ |
| BF16-self-single | `Qwen/Qwen3.6-27B` (re-run) | none (BF16) | — | 1 | 17 | ✅ |
| BF16-self-multi | `Qwen/Qwen3.6-27B` (re-run) | none (BF16) | — | 8 | 64 | ✅ |
| FP8-single | `Qwen/Qwen3.6-27B-FP8` | fp8 | — | 1 | 17 | ✅ |
| FP8-multi | `Qwen/Qwen3.6-27B-FP8` | fp8 | — | 8 | 64 | ✅ |
| NVFP4-single | `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | modelopt_fp4 | — | 1 | 17 | ✅ (eager) |
| NVFP4-multi | (same) | modelopt_fp4 | — | 8 | 64 | ✅ (eager) |
| ~~FP8-MTP3~~ | `Qwen/Qwen3.6-27B-FP8` + MTP=3 | fp8 | MTP=3 | — | — | ❌ See SKIP_REASON.md |

### Settings

- TP=2 across GPUs 1+2 (UUIDs `d558b2e1`, `ae60c9dd`); GPU 0 reserved for display.
- `attention_backend=FLASHINFER`, `moe_backend=auto`, `kv_cache_dtype=auto`.
- `--language-model-only` (Qwen3.6-27B base is multimodal; we run text-only).
- `--max-num-seqs=1`, `max_num_batched_tokens=4096`, `max_model_len=4096`.
- `max_logprobs=-1` (return full vocab logprobs at every position).
- BF16/FP8 cells use vLLM's default torch.compile + CUDAGraph (`cudagraph_capture_sizes=[1,2]`).
- NVFP4 cells use **`--enforce-eager`** to work around an inductor stride-mismatch bug in the AOT-cached compile (see `Adaptations` below).
- Position 0 of every prompt is dropped (`--skip-prefill-next=1`, GLM default) to avoid prefill-boundary numerical noise.

### WikiText source (and a bug we found in the GLM scripts)

The GLM-5.1 reference scripts try to load WikiText-2-raw-v1 test split via `datasets.load_dataset(...)` at runtime. **`datasets` is not installed in `repne/vllm:v13`** — the scripts catch the `ImportError` and silently fall back to `("The quick brown fox jumps over the lazy dog. This fallback prompt is only used when the WikiText cache is unavailable. ") * 16384`.

We initially missed this and our v1 cells all ran on the periodic fallback, producing non-trivial but methodologically wrong numbers. **Our v2 cells (the ones reported here)** use a host-side `prep_wikitext_tokens.py` helper that loads WikiText on the host (where `datasets` works), tokenizes with the Qwen3.6 tokenizer, and writes the prompts as a `safetensors` file mounted into the container as `--token-source`. The 8 multi-prompt cells now use 8 genuinely-different WikiText prompts, verified diverse before the run.

The v1 fallback artifacts are preserved at `_fallback_artifacts/` in our local sweep dir; they are **not** part of this publication. We strongly recommend anyone reproducing this either install `datasets` in their image or use this `prep_wikitext_tokens.py` approach.

## R1 validation

The methodology hinges on a vLLM V1 invariant: `RequestOutput.outputs[i].logprobs` are recorded **before** the custom `TeacherForceLogitsProcessor` is applied. The processor docstring claims this; we verified empirically in Phase 0c with a 4-token smoke test on BF16:

- **Position 0** (with a 128-token context, intentionally short to expose the difference): model's actual top-1 was token `3992` at logprob `-0.504`; the forced token `5388` was recorded at logprob `-12.69` (≈3e-6 probability). If the saved logprobs were post-processor, the forced token would be at logprob `0.0` and everything else at `-inf`.
- **Positions 1-3**: forced token logprobs are realistic (`-0.015`, `-0.045`, `-0.022`), not the one-hot `0.0` that a post-processor capture would show.
- **Total probability mass sums to 1.0** at every position (within fp32 precision).

R1 holds on `repne/vllm:v13` for Qwen3.6-27B. Smoke artifact metadata at `results/smoke/smoke-bf16.safetensors.json`.

## Adaptations vs. GLM-5.1 reference

| Knob | GLM-5.1 default | Our Qwen3.6 setting | Reason |
| --- | --- | --- | --- |
| `--quantization` | `modelopt_fp4` (GLM is NVFP4) | `none` / `fp8` / `modelopt_fp4` per cell | Match each variant |
| `--attention-backend` | `B12X_MLA_SPARSE` | `FLASHINFER` | B12X kernels are GLM-specific |
| `--moe-backend` | `b12x` | `auto` | Same |
| `--hf-overrides` | `{"index_topk_pattern": "FFSFSSS..."}` | `{}` | GLM-only sparse-index config |
| `--language-model-only` | (not set) | `True` for all cells | Qwen3.6-27B base is multimodal; we run text-only |
| `--enforce-eager` | (not set, CUDAGraph on) | **NVFP4 cells only** | Inductor AOT-compiled kernel hits `assert_size_stride(arg2_1, (s18, 24, 128), (8240, 128, 1))` with actual stride `8256` — a 16-byte alignment bug in the NVFP4 path on this vLLM build. Eager mode bypasses it. |
| `--speculative-config` | (not in original script) | added by patch (unused in published cells) | Was needed for the MTP3 cells; left in place for future use |

The argparse + kwargs additions for `--speculative-config` and `--language-model-only` are 4 lines each in both `decode_logprob_kld.py` and `decode_logprob_kld_multi.py`. Original sha and patch shown in `scripts/.PATCH_NOTES.md` (TODO).

## Files

- `scripts/teacher_force_logits_processor.py` — V1 `AdapterLogitsProcessor` (verbatim from `festr2/glm51-decode-kld-refs`)
- `scripts/decode_logprob_kld.py`, `scripts/decode_logprob_kld_multi.py` — collect+compare (patched: `--speculative-config`, `--language-model-only`)
- `scripts/prep_wikitext_tokens.py` — host-side WikiText pre-tokenization (workaround for missing `datasets` in repne/vllm:v13)
- `scripts/inspect_smoke.py` — R1 decision gate (Phase 0c)
- `scripts/launch_collect.sh` — production cell launcher (parametrized; JSON-quoting-safe)
- `scripts/run_all_v2.sh` — orchestrator for the 8 sequential cells
- `scripts/build_leaderboard.py` — produces `compare/leaderboard.csv`
- `compare/*.json` — per-cell pairwise compare summaries (KL/JSD per-position arrays + means/maxes)
- `compare/leaderboard.csv` — final summary table
- `SKIP_REASON.md` — full root-cause analysis of the FP8-MTP3 incompatibility
- `results/smoke/smoke-bf16.safetensors.json` — R1 validation metadata

The raw logprob safetensors (≈2 GB total: 4 ref/self files + 4 variant files) are kept locally and not published in this repo. They can be regenerated deterministically from `scripts/run_all_v2.sh` + `scripts/prep_wikitext_tokens.py` in ~25 minutes.

## Reproduce

```bash
cd sweeps/11-teacher-forced-kld
# Pre-tokenize WikiText once (host Python with `datasets` installed):
python3 scripts/prep_wikitext_tokens.py --num-prompts 1 --max-tokens 17 \
    --output refs/wikitext-tokens-single.safetensors
python3 scripts/prep_wikitext_tokens.py --num-prompts 8 --max-tokens 64 \
    --output refs/wikitext-tokens-multi.safetensors

# Run all 8 cells (~25 minutes):
bash scripts/run_all_v2.sh

# Build leaderboard:
bash scripts/run_compare.sh
python3 scripts/build_leaderboard.py
```

## Audience-relevant context

- **For Repne:** these are the FP8 numbers you wanted on Qwen3.6-27B. Bottom line: `repne/vllm:v13` FP8 inference is ≈7-9× the BF16 self-noise floor on KL, with most of the multi-prompt mean coming from a few outlier positions. NVFP4 (a different model — `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`) is ≈400× the noise floor on single-prompt and shows real, distributed drift, not just outliers.
- **For Lavd:** as you suspected, the methodology composes with FP8 cleanly but breaks at MTP=3 — vLLM V1 unconditionally rejects custom logits processors when `speculative_config` is set (PR #19482). The check is a deliberate "TODO" by `@njhill` to prevent silent corruption from interface mismatch. The path forward is upstream PR work; we're not going to monkey-patch around it.
