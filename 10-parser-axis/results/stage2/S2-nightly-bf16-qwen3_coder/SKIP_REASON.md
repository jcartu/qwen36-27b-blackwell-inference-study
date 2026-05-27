# S2-nightly-bf16-qwen3_coder — Not Runnable

**Verdict:** `vllm/vllm-openai:nightly` does **not** support `DFlash` speculative decoding.

## Evidence

```
$ docker run --rm --entrypoint python3 vllm/vllm-openai:nightly -c "..."
INFO ... vllm: 0.21.1rc1.dev323+g1fc2cee50
dflash/draft archs: ['LongcatFlashForCausalLM', 'MiMoV2FlashForCausalLM']
```

The `DFlashDraftModel` architecture used by the `z-lab/Qwen3.6-27B-DFlash` drafter
is **not registered** in upstream nightly vLLM. Only Repne's vLLM fork
(`repne/vllm:v13`) carries the DFlash patches.

## Implication for Stage 2 (image axis on BF16+DFlash)

Since only one image (`repne/vllm:v13`) supports the configuration, Stage 2 is
**uncontested**:

- `S2-v13-bf16-qwen3_coder`: final_score=62, responsiveness=80 (✓ booted, benched)
- `S2-nightly-bf16-qwen3_coder`: **not runnable** (DFlash not in nightly)

**Stage 2 BF16 image winner: `repne/vllm:v13` by default (no nightly alternative).**

## Boot failure trace

The container started and crashed during engine init at 71 seconds (model loading
phase). The actual `DFlashDraftModel` registration miss happens during the
`Resolved architecture: DFlashDraftModel` step that succeeds on `repne/vllm:v13`
but errors out on `vllm/vllm-openai:nightly`.

See `launch-S2-nightly-bf16-qwen3_coder.log` for the truncated traceback.

## Cross-check on FP8 (Stage 3)

The Stage 3 result is the genuine FP8 image-axis comparison, since FP8+MTP=3 is
upstream-supported on both images:

- `S1-v13-fp8-coder` (Stage 1, same cell as Stage 3 baseline): final_score=62
- `S3-nightly-fp8-qwen3_coder`: final_score=48

**FP8 image winner: `repne/vllm:v13` (+14 points over nightly).**
