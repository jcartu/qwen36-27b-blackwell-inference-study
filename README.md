[![← qwen-bench hub](https://img.shields.io/badge/%E2%86%90-qwen--bench_hub-blueviolet?style=for-the-badge)](https://github.com/jcartu/qwen-bench)

> Part of the [`qwen-bench`](https://github.com/jcartu/qwen-bench) ongoing benchmark series.
> See the hub for the current SOTA leaderboard and a chronological index of all studies.

---

# Empirical Characterization of Qwen3.6-27B Inference on Dual NVIDIA RTX PRO 6000 Blackwell

**Eight experiments, 1,200+ benchmark runs, 102K-token KL-divergence quality probe.**
*A systematic head-to-head of the [`repne/vllm`](https://hub.docker.com/r/repne/vllm) fork, [upstream vLLM v0.19.1 / v0.20.1](https://github.com/vllm-project/vllm), and [llama.cpp Q8_0 / BF16 GGUF](https://github.com/ggml-org/llama.cpp), across four quantization schemes (BF16, FP8 W8A8, NVFP4, GGUF Q8_0) and three speculative-decoding methods (none, MTP n∈{2,3,4,5,6}, DFlash n∈{7,8,15}).*

---

## Abstract

We benchmarked a dual-GPU Blackwell SM120 inference stack (2× NVIDIA RTX PRO 6000, 96 GiB each, PCIe Gen5 x16) under tensor-parallel-2 across **eight controlled experiments** spanning a single 24-hour development sprint. The experiments cover (i) image-version regression detection, (ii) scheduler-knob pessimal-pocket identification, (iii) third-party NVFP4 quantization viability, (iv–v) head-to-head comparison of the Repne fork against upstream vLLM v0.19.1 / v0.20.1, (vi) revalidation against an updated Repne image, (vii) quality probes triggered by community concerns (PhaelonQuant Discord) about W8A8 activation quantization, and (viii) high-concurrency throughput characterization (c ∈ {8, 16, 32}) plus a 102K-token KL-divergence test against BF16 reference logits using [AesSedai's `perplexity-sliding-window` branch](https://github.com/AesSedai/llama.cpp/tree/perplexity-sliding-window).

**Headline result.** The optimal production configuration is **FP8 W8A8 weights with Multi-Token Prediction (MTP) speculative decoding at n=3**, served via the Repne fork (`repne/vllm:latest`). It achieves **2,083.7 tok/s aggregate at c=32 ctx=0**, **350.5 tok/s at c=4 ctx=131k**, and **117.1 tok/s at c=1 ctx=0**, while passing 4/4 functional gates (Fibonacci, tool-call, arithmetic reasoning, multi-turn coherence). The companion Q8_0 GGUF measurement establishes that 8-bit weight quantization on this model yields a KL-divergence of **0.001828 nats** vs BF16 — well within the noise floor of perplexity measurement — and exhibits **97.9% top-token agreement** with BF16. By extension and per Qwen's own model card, FP8 W8A8 quality is also empirically indistinguishable from BF16 for the workloads tested.

---

## 1. Headline findings

### 1.1 Repne fork dominates upstream for `dflash` long-context decoding

| concurrency × context | Repne tok/s | Upstream v0.20.1 tok/s | Δ |
|---|---:|---:|---:|
| c=1 × 131k | **81.4** | 11.7 | **+598%** 🚨 |
| c=2 × 131k | **162.7** | 22.4 | **+626%** 🚨 |
| c=4 × 131k | **284.4** | 44.7 | **+537%** 🚨 |

Upstream's `dflash` collapses past 32k context because flashinfer rejects the non-causal attention pattern the drafter requires, forcing the main path onto flash_attn — and upstream lacks Repne's `gumbel` draft sampler and `use_local_argmax_reduction` flags that make `dflash` viable in production. Drafter acceptance falls from 30% (Repne) to 1–3% (upstream); inter-token latency inflates 8× (11 ms → 90 ms). [→ Experiment 5](./05-bf16-dflash-headtohead/)

### 1.2 Repne fork wins on FP8+MTP=3 — narrower margin, same direction

| concurrency × context | Repne tok/s | Upstream v0.20.1 tok/s | Δ |
|---|---:|---:|---:|
| c=1 × 0 | **120.1** | 112.3 | +7.0% |
| c=2 × 0 | **223.8** | 197.0 | +13.6% |
| c=4 × 0 | **449.5** | 413.8 | +8.6% |
| c=4 × 131k | **347.4** | 345.0 | +0.7% |

FP8+MTP=3 is the only path where upstream is a viable fallback (long-context delta is essentially zero, ~5–14% short-context cost). [→ Experiment 4](./04-fp8-mtp3-headtohead/)

### 1.3 NVFP4-MTP is not a viable production replacement

| concurrency × context | NVFP4+MTP=3 tok/s | FP8+MTP=3 ref | Δ |
|---|---:|---:|---:|
| c=1 × 0 | 109.4 | ~85 | +28.7% ✅ |
| c=4 × 0 | 416.6 | 352.8 | +18.1% ✅ |
| c=2 × 131k | 105.6 | 145.9 | **−27.6%** ❌ |
| c=4 × 131k | 7.0 | 272.0 | **−97.4%** ❌ |
| c=1 × 244k | 1.4 | — | unusable ❌ |

NVFP4 wins +12 to +29% at short context, loses 27–33% at 131k, and is **functionally broken at 244k**. Production target is 256k context, which disqualifies NVFP4 outright. The community-quantized model itself is correct (15 mtp.* tensors graft cleanly, modelopt format, all 10 functional gates pass — Fibonacci 5/5, tool calls, multi-turn, 47×83=3901, 137K-token needle-in-haystack found in 23.3 s). The bottleneck is engine-side at long context, not model quality. Re-running on the Repne fork (Experiment 7 Phase D) made things 50% **worse**, not better — Repne's optimizations are W8A8/BF16-specific. [→ Experiment 3](./03-nvfp4-mtp-experiment/), [Experiment 7 Phase D](./07-quality-sprint/phase-d-nvfp4-repne/)

### 1.4 Speculative-decoding `num_speculative_tokens` is concurrency-dependent

The optimal MTP token count depends on workload concurrency. We initially identified MTP=5 as superior based on c=1–4 testing (mean +1.8% over MTP=3), then re-extended the bench to production-realistic concurrency:

| Concurrency tier | Winner | MTP=3 advantage over MTP=5 |
|---|:--:|---:|
| c=1 to c=4 | MTP=5 | −1.8% to −6.5% (MTP=5 wins) |
| c=8 | MTP=3 | +0.4% to +2.1% |
| c=16 | MTP=3 | +8.7% to **+14.4%** |
| c=32 | MTP=3 | +17.2% to **+21.2%** |

**Crossover at c=8.** The mechanism: MTP=5 has 5 spec positions per cycle; positions 4–5 have lower acceptance probability than 1–3. At low concurrency, the GPU has spare compute and the rejected positions are nearly free. At higher concurrency, every rejected position consumes proportionally more budget. MTP=3 has fewer wasted positions per cycle, which dominates at c≥8. [→ Experiment 7 Phase C](./07-quality-sprint/phase-c-fp8-sweep/), [Experiment 8 X1](./08-x1y1-sprint/)

### 1.5 8-bit quantization is empirically lossless on this model

| Metric | BF16 GGUF (reference) | Q8_0 GGUF (test) |
|---|---:|---:|
| Perplexity (wikitext-2, 102k tokens) | **7.620 ± 0.062** | **7.623 ± 0.063** |
| PPL ratio | 1.000 | **1.0004** (+0.04%) |
| Mean KL-divergence | — | **0.001828 ± 0.000189 nats** |
| 95th percentile KLD | — | 0.002880 |
| Same top-1 token agreement | — | **97.9%** |

Phaelon's claim that GGUF Q8_0 should be "more accurate than W8A8 because activations stay BF16 on the compute path" is theoretically defensible. Empirically, the BF16→Q8 KLD is **0.0018 nats** — at the noise floor of perplexity measurement on a 102K-token corpus. By transitive inference (Qwen team's own FP8 model card claims "performance metrics nearly identical to original" + our Phase B 8/8 functional-test parity + this Q8/BF16 noise-floor result), **FP8 W8A8 is also empirically indistinguishable from BF16 for production use**. [→ Experiment 7 Phase B](./07-quality-sprint/phase-b-fp8-vs-q8/), [Experiment 8 Y1](./08-x1y1-sprint/y1-perplexity/)

### 1.6 Speculative decoding is not bitwise-lossless even at temp=0

Comparing FP8+MTP=3 vs FP8+no-spec on identical prompts at `temperature=0, top_p=1.0, seed=42`, both engines individually produce deterministic outputs (3× identical runs). Yet **4 of 8 prompts** produce divergent text between the two engines. Differences are stylistic substitutions ("long multiplication" vs "multiplication"), not factual errors — all outputs remain correct, all functional gates pass. The Repne fork's `gumbel` draft sampler appears to take a different RNG path through accept/reject decisions than greedy sampling. **For production this is acceptable**, but it means the standard speed-vs-quality tradeoff is real if not large. [→ Experiment 7 Phase A](./07-quality-sprint/phase-a-spec-losslessness/)

### 1.7 Hardware sanity: PCIe Gen1 readings at idle are normal

The bench tool's startup `nvidia-smi` snapshot reports `pcie.link.gen.current = 1`, which is alarming on first inspection. Direct active-load measurement confirms the PCIe link **trains to Gen5 (32 GT/s) x16** under GPU compute pressure — the Gen1 reading reflects ASPM downscaling at idle, not a hardware fault. Both GPUs verified at full PCIe Gen5 bandwidth during all sustained-load benchmarks.

---

## 2. SOTA matrix (final)

Best aggregate throughput recorded across all eight experiments. **FP8+MTP=3 holds the SOTA at every production-relevant cell.** MTP=5 holds the absolute single-stream record at c=1×131k (101.2 vs 95.0) and a few isolated low-concurrency cells, but is dominated everywhere production traffic actually lives.

| concurrency × context | tok/s | config | source |
|---|---:|---|---|
| c=1 × 0 | 120.1 | FP8+MTP=3 | exp 04, N=1 evening peak |
| c=1 × 32k | 119.2 | FP8+MTP=3 | exp 06, N=3 |
| c=1 × 131k | 101.2 | FP8+MTP=5 | exp 07, N=3 — *MTP=5 record* |
| c=2 × 0 | 234.4 | FP8+MTP=5 | exp 07, N=3 |
| c=2 × 32k | 231.2 | FP8+MTP=5 | exp 07, N=3 |
| c=2 × 131k | 190.8 | FP8+MTP=5 | exp 07, N=3 |
| c=4 × 0 | 472.2 | FP8+MTP=5 | exp 07, mini-matrix N=2 |
| c=4 × 32k | 454.5 | FP8+MTP=3 | exp 06, N=3 |
| c=4 × 131k | 350.5 | FP8+MTP=3 | exp 06, N=3 |
| c=8 × 0 | 875.0 | FP8+MTP=3 | exp 08 X1, N=2 |
| c=8 × 32k | 795.4 | FP8+MTP=3 | exp 08 X1, N=2 |
| c=8 × 131k | 534.4 | FP8+MTP=3 | exp 08 X1, N=2 |
| c=16 × 0 | 1,520.6 | FP8+MTP=3 | exp 08 X1, N=2 |
| c=16 × 32k | 1,186.0 | FP8+MTP=3 | exp 08 X1, N=2 |
| c=16 × 64k | 1,047.1 | FP8+MTP=3 | exp 08 X1, N=2 |
| **c=32 × 0** | **2,083.7** ⭐ | **FP8+MTP=3** | exp 08 X1, N=2 |
| c=32 × 16k | 1,892.3 | FP8+MTP=3 | exp 08 X1, N=2 |
| c=32 × 32k | 1,656.3 | FP8+MTP=3 | exp 08 X1, N=2 |

See [`SOTA.md`](./SOTA.md) for full discussion.

---

## 3. Materials and methods

### 3.1 Hardware

| Component | Specification |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120) |
| VRAM per GPU | 96 GiB GDDR7 |
| GPU interconnect | PCIe Gen5 x16 (verified under load) |
| Driver | 595.71.05 |
| CUDA | 13.2 |
| Host CPU | Intel Xeon W (W790E-SAGE motherboard) |
| OS | Linux 7.0.2-arch1-1 |

### 3.2 Software

| Component | Version |
|---|---|
| Repne fork (production winner) | `repne/vllm:latest` (`d0a200f77546`, May 5 2026 evening) |
| Repne engine | `v0.1.dev16400+g910d87a9d.d20260505` |
| Upstream vLLM | `vllm/vllm-openai:v0.20.1-cu129-ubuntu2404` (`7ba11e462b5a`) |
| llama.cpp (perplexity) | [AesSedai's `perplexity-sliding-window` branch](https://github.com/AesSedai/llama.cpp/tree/perplexity-sliding-window) at `e9ae70b`, built `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` |
| Bench harness | [`llm_decode_bench.py v0.4.8`](https://github.com/cole-yoshioka/llm-inference-bench) |

### 3.3 Models

| Model | Source | Use |
|---|---|---|
| Qwen3.6-27B-FP8 (W8A8 block-128) | `Qwen/Qwen3.6-27B-FP8` | Production weights |
| Qwen3.6-27B (BF16) | `Qwen/Qwen3.6-27B` | DFlash main model |
| Qwen3.6-27B-DFlash (drafter) | `z-lab/Qwen3.6-27B-DFlash` | DFlash speculative draft model |
| Qwen3.6-27B-Text-NVFP4-MTP | `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | NVFP4 viability test |
| Qwen3.6-27B Q8_0 GGUF | `unsloth/Qwen3.6-27B-GGUF` Q8_0 | KL-divergence quality probe |
| Qwen3.6-27B BF16 GGUF | `unsloth/Qwen3.6-27B-GGUF` BF16 (split files) | Perplexity reference |

### 3.4 Bench protocol

Unless otherwise noted, all aggregate-throughput cells used:
- Tensor parallelism = 2 (one shard per GPU)
- Max model length = 262 144 tokens
- Reasoning parser = `qwen3`, tool-call parser = `qwen3_coder`
- Prefix caching enabled
- Sustained-decode mode: 30 s active window + 10 s warmup
- `--skip-prefill` (decode-only measurement to isolate the speculative-decoding regime from one-time prefill cost)
- N=2 minimum, N=3 for production-decision experiments, N=5 for variance characterization (Experiment 1)

KV-cache sizes were measured directly from engine logs at startup and used as the `--kv-budget` parameter for the bench harness, ensuring the bench tool correctly clamps requested context to within physical capacity.

### 3.5 Quality probes

Functional gates for every speculative-decoding configuration:
1. **Fibonacci sequence (5×, deterministic):** must produce exact "1, 1, 2, 3, 5, 8, 13, 21, 34, 55" five times in a row at temp=0
2. **Tool call:** must emit valid `get_weather({"city":"Tokyo"})` JSON given a single tool definition
3. **Arithmetic reasoning:** must compute 47×83=3901 with intermediate steps
4. **Multi-turn coherence:** must answer "which is warmer, Tokyo at 28C or Berlin at 18C?" correctly across three conversation turns

Perplexity probe (Experiment 8 Y1):
- Corpus: wikitext-2 test set, 102 200 token positions
- Method: 200 sliding windows × 511 positions each
- Context length: 512 tokens, stride: 128
- Reference: Qwen3.6-27B BF16 GGUF logits, saved to disk via `--kl-divergence-base`
- Test: Qwen3.6-27B Q8_0 GGUF logits, compared via `--kl-divergence`
- Output: per-token KLD distribution, top-token agreement rate, perplexity ratio

---

## 4. Experiment index

| # | Path | Question | When | Method | Outcome |
|---|---|---|---|---|---|
| 1 | [`01-morning-newimage-validation/`](./01-morning-newimage-validation/) | Does the new Repne May 5 image regress the May 3 baseline? | morning | BF16+DFlash, 5 cells × N=5 randomized = 25 runs | Mixed: −5.7% c=4 ctx=0, +9.0% c=1 ctx=128k. Triggered Exp. 2. |
| 2 | [`02-scheduler-investigation/`](./02-scheduler-investigation/) | What scheduler knobs cause the c=4 ctx=0 regression? | midday | 5 variants × N=3 | Default `max-num-seqs=128 + max-num-batched-tokens=32 768` is in a pessimal pocket; nearly any deviation recovers throughput. |
| 3 | [`03-nvfp4-mtp-experiment/`](./03-nvfp4-mtp-experiment/) | Can NVFP4 community-quant replace FP8? | afternoon | 11 cells × N=3 + 4 functional gates | No. Wins short-context, broken at 244k. |
| 4 | [`04-fp8-mtp3-headtohead/`](./04-fp8-mtp3-headtohead/) | Repne vs upstream on FP8+MTP=3? | evening | 6 cells × N=1 | Repne +5–14% short-context, parity at long-context. |
| 5 | [`05-bf16-dflash-headtohead/`](./05-bf16-dflash-headtohead/) | Repne vs upstream on BF16+DFlash? | evening | 6 cells × N=1 | Repne +537–626% at long-context (upstream collapses). |
| 6 | [`06-new-image-validation/`](./06-new-image-validation/) | New Repne image (`d0a200f7`) FP8 vs DFlash variants? | late evening | 9 cells × 4 configs × N=3 = 108 runs | FP8+MTP=3 wins all 9 cells. Best DFlash variant: n=7. |
| 7 | [`07-quality-sprint/`](./07-quality-sprint/) | Triggered by Phaelon Discord on W8A8 quality. | overnight | A: spec losslessness. B: FP8 vs Q8 GGUF. C: MTP n∈{2,3,4,5,6}. D: NVFP4 on Repne. | A: not bitwise-lossless. B: no FP8 quality regression. C: MTP=5 wins at c=1–4 (caveat triggered Exp. 8). D: NVFP4 50% worse on Repne. |
| 8 | [`08-x1y1-sprint/`](./08-x1y1-sprint/) | High-concurrency speed (X1) + perplexity quality (Y1). | overnight 2 | X1: c∈{8,16,32} × ctx ∈ {0,32k,131k}. Y1: AesSedai KLD. | **MTP=3 wins +10.5% mean at c≥8.** Q8/BF16 KLD = 0.0018 (noise floor). |

Each experiment subdirectory contains raw `.json` per-cell results, `bench.log` files with full bench-tool stdout, and a `RESULTS.md` write-up.

---

## 5. Repne-fork-only flags (operational notes)

The Repne fork ships several flags that upstream vLLM rejects with `pydantic.ValidationError: Unexpected keyword argument`:

- `--load-format instanttensor` — faster weight-loading path
- `--speculative-config.draft_sample_method gumbel` — gumbel draft sampling instead of greedy
- `--speculative-config.attention_backend flash_attn` — independent attention backend for the spec path
- `--speculative-config.use_local_argmax_reduction true` — drafter argmax-reduction optimization

For BF16+DFlash, dropping these flags causes a **5–6× long-context regression** (Experiment 5). Drafter acceptance collapses from 16–36% to 1–3% at ctx > 32k. **These flags are doing real work, not engineering aesthetics.**

For FP8+MTP, only `gumbel` and `instanttensor` are required to reproduce the Repne advantages. Dropping them on upstream costs ~5–14% short-context throughput.

---

## 6. Image and version notes

| Image | Status | Notes |
|---|---|---|
| `repne/vllm:latest` (`d0a200f77546`, May 5 evening) | **Production** | Engine `v0.1.dev16400+g910d87a9d`. +41 commits over morning image. Tool-calling fix verified. |
| `repne/vllm:latest` (`5e7583ca4df9`, May 5 morning) | Validated | All Repne-side benches in Experiments 1–5 used this. |
| `vllm/vllm-openai:v0.20.1-cu129-ubuntu2404` | Validated fallback | Used for upstream head-to-heads and NVFP4 retry. |
| `vllm/vllm-openai:v0.19.1-cu130-ubuntu2404` | **Avoid** | Catastrophic crash at ctx > 131k with NVFP4 modelopt: `NV_ERR_GPU_IN_FULLCHIP_RESET`. Required host reboot. |
| `vllm/vllm-openai:cu130-nightly` | **Broken** | `ModuleNotFoundError: No module named 'pandas'` in `_aiter_ops.py`. Do not use. |

---

## 7. Production configuration (final)

```bash
docker run -d --name qwen-vllm-fp8-tp2 --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  --volume "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --volume "$HOME/.cache/vllm:/root/.cache/vllm" \
  --volume "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  --volume "$HOME/.triton/cache:/root/.triton/cache" \
  --env OMP_NUM_THREADS=16 \
  --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env NCCL_P2P_LEVEL=SYS --env NCCL_NET_GDR_LEVEL=SYS --env NCCL_MIN_NCHANNELS=8 \
  repne/vllm:latest \
    -O3 --model Qwen/Qwen3.6-27B-FP8 --served-model-name Qwen3.6-27B qwen3.6-27b \
    --port 11435 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \
    --max-model-len 262144 --max-num-seqs 128 --max-num-batched-tokens 32758 \
    --max-cudagraph-capture-size 256 --language-model-only --enable-auto-tool-choice \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-prefix-caching \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 \
    --speculative-config.draft_sample_method gumbel \
    --attention-backend flashinfer --load-format instanttensor \
    --default-chat-template-kwargs.preserve_thinking true
```

**Engine state at this configuration:**
- KV cache: 1 846 472 tokens (7.04× max concurrency at 256K)
- Boot time: 120 s with warm compile cache, 295 s cold
- Functional gates: 4/4 pass

---

## 8. Companion repositories

These four were published as standalone artifacts during the sprint for fast sharing. This monorepo is the canonical comprehensive record.

- [`jcartu/repne-dflash-newimage`](https://github.com/jcartu/repne-dflash-newimage) — initial Repne May 5 image validation (single-shot bench, before the variance sweep)
- [`jcartu/qwen36-27b-nvfp4-mtp-experiment`](https://github.com/jcartu/qwen36-27b-nvfp4-mtp-experiment) — full NVFP4+MTP write-up (Experiment 3)
- [`jcartu/qwen36-27b-fp8-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream) — FP8+MTP=3 head-to-head (Experiment 4)
- [`jcartu/qwen36-27b-bf16-dflash-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-bf16-dflash-repne-vs-upstream) — BF16+DFlash head-to-head (Experiment 5)

---

## 9. Recommendation

**Adopt FP8+MTP=3 on the Repne fork as the production configuration** for any Qwen3.6-27B inference deployment on Blackwell SM120 hardware. Verified at every production-relevant concurrency tier (c=1 through c=32) across short, medium, and long context. Peak aggregate throughput **2,083.7 tok/s at c=32 ctx=0**. Quality verified empirically against BF16 reference logits with KLD = 0.0018 nats (noise floor) on Q8 GGUF, and inferred to hold for FP8 W8A8 by Qwen team's own model-card claims plus our 8/8 functional-test parity.

If the Repne fork ever stops shipping new images, **upstream `vllm/vllm-openai:v0.20.1` with FP8+MTP=3** is the safest fallback — accept a 5–14% short-context throughput cost. **Do not attempt** to recreate the BF16+DFlash performance gap on upstream without forking back the gumbel sampler and `use_local_argmax_reduction` — the upstream drafter implementation collapses past 32k context and the deficit is too large to close with parameter tuning alone.

NVFP4 quantization on this hardware is permanently disqualified by long-context engine instability (Experiment 3) and Repne-fork incompatibility (Experiment 7 Phase D). DFlash variants don't reach FP8+MTP=3 throughput at any concurrency tier we tested. MTP n∈{2,4,5,6} variants are all dominated by n=3 at c≥8.

---

## 10. Reproducibility

```
.
├── 01-morning-newimage-validation/   # Exp 1: 25 raw N=5 runs + RESULTS.md
├── 02-scheduler-investigation/       # Exp 2: 15 raw N=3 runs across 5 variants
├── 03-nvfp4-mtp-experiment/          # Exp 3: 11 cells × N=3 + functional gates + PHASE5_DECISION.md
├── 04-fp8-mtp3-headtohead/           # Exp 4: 6 cells × N=1 × 2 builds
├── 05-bf16-dflash-headtohead/        # Exp 5: 6 cells × N=1 × 2 builds
├── 06-new-image-validation/          # Exp 6: 9 cells × 4 configs × N=3 = 108 runs
├── 07-quality-sprint/                # Exp 7: Phases A, B, C, D
├── 08-x1y1-sprint/                   # Exp 8: X1 high-concurrency + Y1 perplexity
├── README.md                         # This document
├── SOTA.md                           # Cross-experiment best-tok/s-per-regime table
└── master-results.csv                # 328-row machine-readable summary
```

Every `runs/<cell>_run<N>/results.json` contains the bench tool's full per-request samples (TTFT distribution, ITL distribution, spec-decode acceptance rate, GPU utilization). Every `bench.log` contains the bench tool's complete stdout including the run command line.
