# SKIP_REASON: `fp8-mtp3-single` and `fp8-mtp3-multi` cells

**Date:** 2026-05-27
**Status:** Not measured. Methodology incompatibility, not a configuration error.

## What we tried

Run Qwen3.6-27B-FP8 with MTP=3 speculative decoding (`--speculative-config={"method":"mtp","num_speculative_tokens":3}`) under the GLM-5.1 teacher-forced decode KLD methodology, which requires a custom `LogitsProcessor` (`TeacherForceLogitsProcessor`) registered via `vllm.LLM(logits_processors=[...])`.

## What failed

vLLM `v0.1.dev17130+g155fef0e5.d20260526` (inside `repne/vllm:v13`) rejects this combination at engine init:

```
File "/opt/venv/lib/python3.12/site-packages/vllm/v1/sample/logits_processor/__init__.py",
  line 203, in build_logitsprocs
    raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
ValueError: Custom logits processors are not supported when speculative decoding is enabled.
```

The check is unconditional:

```python
# vllm/v1/sample/logits_processor/__init__.py:200-205
if vllm_config.speculative_config:
    if custom_logitsprocs:
        raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
```

## Why this is fundamental in this build (not just a soft check we can flip)

Verified via vLLM upstream PR archaeology (PR [#19482](https://github.com/vllm-project/vllm/pull/19482), commit `6ebaf43e`, authors @southfreebird and @njhill):

1. **Built-in logits processors (penalties, `allowed_token_ids`, bad-words) are integrated** with the V1 spec decode rejection sampler via a special `apply_with_spec_decode(logits, num_draft_tokens)` interface that operates on a flattened `(sum(num_draft_tokens), vocab_size)` tensor.

2. **Custom logits processors only implement the standard `apply(logits)` interface** that expects a `(batch_size, vocab_size)` tensor. They have no `apply_with_spec_decode` method.

3. **Maintainer @njhill (PR #19482 review):** *"In particular we'll want to look at how this could also work with generic/custom logit processors. But we can get this merged first and address those as follow-on work. ... it would be good to add a startup check for spec decode + custom logits processors (if there is not one already) since it will still be silently 'incorrect' in that case otherwise."*

The check exists precisely **to prevent silent corruption**. Bypassing it would let our processor receive flattened draft logits and apply teacher forcing to wrong rows — corrupting results without crashing.

## Why workarounds are not viable

| Workaround | Verdict |
| --- | --- |
| Patch the `ValueError` to a no-op | Custom processor receives `(sum(num_draft_tokens), vocab_size)` instead of `(batch_size, vocab_size)`. Would silently apply teacher forcing to wrong draft positions. |
| Use `SamplingParams.allowed_token_ids` (which IS supported in spec decode) instead of a custom processor | `allowed_token_ids` is **per-request, static**. Forcing 17 different teacher tokens needs 17 separate single-token requests, but `max_tokens=1` disables spec decode entirely — defeating the point of measuring MTP=3. |
| Run target model only, drop spec decode | Equivalent to `fp8-single`/`fp8-multi` cells (already collected). Doesn't measure MTP=3. |
| Free-generation comparison (no teacher forcing) | Different metric — sampling variance dominates after first token disagreement. Not comparable to teacher-forced KLD numbers. |

## Consultation trail

Two parallel consultations confirmed the verdict:

- **Oracle** (high-IQ reasoning): Recommended Path E (skip + document). Identified the silent-corruption risk of patching.
- **Librarian** (vLLM upstream archaeology): Pinpointed PR #19482 as the source, confirmed the restriction is a TODO with no upstream timeline, confirmed `allowed_token_ids` viability for static cases but the per-step-different teacher sequence breaks that path.

Both consultations independently arrived at Path E.

## What we shipped instead

- 4 variant cells: `bf16-ref`, `fp8`, `nvfp4` (each in single + multi modes)
- 1 noise-floor cell: `bf16-self` (single + multi)
- This SKIP_REASON.md as the 5th cell record

The MTP=3 incompatibility is itself a publishable finding: vLLM V1's spec decode pipeline cannot currently support GLM-5.1-style teacher-forced decode KLD measurement on speculative decoding configurations. This is a methodological gap that future work (either vLLM upstream PR or an alternative measurement approach) would need to close.

## Upstream issue

Recommend filing an issue at `vllm-project/vllm` requesting either:
1. A `apply_with_spec_decode` hook on `AdapterLogitsProcessor` so custom processors can opt in to spec decode compatibility, OR
2. A "target-model logit observer" hook decoupled from logits modification (read-only, fires per accepted token in spec decode).

Reference: PR #19482, file `vllm/v1/sample/logits_processor/__init__.py:200-205`, error string `STR_SPEC_DEC_REJECTS_LOGITSPROCS`.
