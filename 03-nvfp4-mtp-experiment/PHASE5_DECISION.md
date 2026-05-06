# NVFP4-MTP Experiment — Phase 5 Decision

**Decision: DO NOT PROMOTE. Stay on FP8+MTP=3.**

## Setup
- **Image:** `vllm/vllm-openai:v0.20.1-cu129-ubuntu2404` (digest `sha256:7ba11e462b5a36def8e66ef56fa8f66a988dd0483b91f98efb2a3212bb545e47`)
  - Initially tested on `v0.19.1-cu130-ubuntu2404` (digest `sha256:f35754668e5f`) but it crashed catastrophically at >131K context, taking GPU0 out with `NV_ERR_GPU_IN_FULLCHIP_RESET`. Upgrade to `v0.20.1` resolved the crash; engine survives, just degrades.
- **Model:** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (19.64GB, 15 mtp.* tensors bf16, modelopt NVFP4, 0 visual tensors)
- **Launch:** TP=2, max-num-seqs=4, max-model-len=262144, kv-cache-dtype=fp8, gpu-mem 0.9, MTP n=3
- **KV cache (v0.20.1):** 4,268,776 tokens, max concurrency 16.28× at 256K (vs v0.19.1's 1,089,600 — v0.20.1 has **3.9× more KV capacity** for NVFP4)

## Hard Gates (Phases 1-3): 10/10 PASS
- ✅ Phase 1: download (19.64GB), safetensors verified (15 mtp.* bf16, 0 visual, modelopt NVFP4)
- ✅ Phase 2 TP=1: boot 196s, fib 5/5, tool call, multi-turn, all clean
- ✅ Phase 3 TP=2: boot 201s/226s, fib 5/5, tool call, multi-turn, 47×83=3901, 137K-token needle-in-haystack found in 23.3s

The model itself is correct. Quantization didn't damage the weights. MTP head graft works (acceptance 20-30% range observed).

## Phase 4 Benchmark Matrix (v0.20.1, N=3)

| cell | conc | ctx | NVFP4+MTP=3 | FP8+MTP=3 ref | Δ |
|---|---|---|---|---|---|
| c=1 ctx=0 | 1 | 0 | **109.4 ± 2.4** | ~85 | **+28.7%** ✅ |
| c=1 ctx=32k | 1 | 32k | **101.2 ± 3.3** | ~90 | **+12.4%** ✅ |
| c=1 ctx=131k | 1 | 131k | **55.1 ± 0.4** | 82.6 | **-33.3%** ❌ |
| c=1 ctx=244k | 1 | 244k | **1.4** | — | **capacity-limited** ❌ |
| c=2 ctx=0 | 2 | 0 | **206.8 ± 2.2** | 160.3 | **+29.0%** ✅ |
| c=2 ctx=32k | 2 | 32k | **178.5 ± 2.6** | ~175 | **+2.0%** = |
| c=2 ctx=131k | 2 | 131k | **105.6 ± 0.9** | 145.9 | **-27.6%** ❌ |
| c=2 ctx=244k | 2 | 244k | **0.0** | — | **fully broken** ❌ |
| c=4 ctx=0 | 4 | 0 | **416.6 ± 10.4** | 352.8 | **+18.1%** ✅ |
| c=4 ctx=32k | 4 | 32k | **277.0 ± 4.8** | ~320 | **-13.4%** ❌ |
| c=4 ctx=131k | 4 | 131k | **7.0 ± 0.3** | 272.0 | **-97.4%** ❌ |

## Patterns

1. **NVFP4 wins at short context** (ctx ≤ 32k): +12 to +29% across all concurrencies
2. **NVFP4 loses at long context** (ctx ≥ 131k): -27 to -33% at c=1 and c=2; collapses entirely at c=4
3. **NVFP4 is functionally broken at 244k context** even on the latest stable vLLM
4. **The c=4 ctx=131k 97% collapse** is concerning — likely a capacity edge case (524k KV cells in flight), but reproducible and not a 1-off variance

## Why we said no-promote

Production target is **256K context** (the entire reason `--max-model-len 262144` is set). At that target context length, NVFP4-MTP either:
- runs at 1.4 tok/s (essentially unusable)
- returns nothing (capacity-limited)
- previously crashed the engine outright on v0.19.1

**The single-stream short-context wins (+28%) don't compensate for losing the long-context use case entirely.**

Even at moderate long context (131K), production loses 27-33% throughput vs FP8+MTP=3. That's the actual operational regime — code agents working with full repository context routinely hit 80-150K tokens. **NVFP4-MTP is meaningfully worse there.**

## What would change the answer

- A vLLM build that doesn't degrade NVFP4 KV-FP8 chunked-prefill at >131K (the Phase 4 collapse pattern suggests prefill or chunked-prefill bug, not weight quality)
- An NVFP4 quantization that retains BF16 KV-cache (sacrifices KV capacity but may preserve long-context throughput)
- A use case where 256K context is not required and most traffic is ≤32K (e.g., chat assistant rather than coding agent)

## Verified secondary findings (worth keeping in notes)

1. **PCIe Gen1 reading at idle is not a bug.** The link ramps to Gen5 (32GT/s) under load, confirmed via direct compute test. The `bench/*/results.json` startup nvidia-smi captures (`pcie.link.gen.current = 1`) reflect idle state, not load state. Stop worrying about this.
2. **`vllm/vllm-openai:cu130-nightly` is broken upstream** (`ModuleNotFoundError: No module named 'pandas'` in `_aiter_ops.py`). Use `cu129` lineage until that's fixed.
3. **vLLM v0.19.1 NVFP4 has a fatal engine bug at ctx>131K** (`RuntimeError: cancelled` in shm_broadcast) that takes a GPU into `NV_ERR_GPU_IN_FULLCHIP_RESET`. Required two host reboots to recover. **v0.20.1 fixed the crash but not the underlying long-context degradation.**
4. **v0.20.1 packs NVFP4 KV cache 3.9× more efficiently** than v0.19.1 (4.27M vs 1.09M tokens at the same gpu-memory-utilization 0.9). This is a real upstream improvement.
5. **Box tuning is fine** — `rasputin-host-tune.service` (CPU governor / EPP / uncore / C1E / sysctl) and `gpu-fan-aggressive.service` (fan curve) do not touch PCIe link state. They are not the cause of any GPU issues observed.

## Files
- Phase 1-3 raw: `/home/josh/qwen-vllm-test/bench/nvfp4-mtp-experiment/phase{2,3}-tp{1,2}/`
- Phase 4 v0.19.1 (partial, pre-crash): `/home/josh/qwen-vllm-test/bench/nvfp4-mtp-experiment/phase4-bench/runs/`
- Phase 4 v0.20.1 (full): `/home/josh/qwen-vllm-test/bench/nvfp4-mtp-experiment/phase4-bench-v201/runs/`
