[![← qwen-bench hub](https://img.shields.io/badge/%E2%86%90-qwen--bench_hub-blueviolet?style=for-the-badge)](https://github.com/jcartu/qwen-bench)

> Part of the [`qwen-bench`](https://github.com/jcartu/qwen-bench) ongoing benchmark series.
> See the hub for the current SOTA leaderboard and a chronological index of all studies.

---

# Qwen3.6-27B on RTX PRO 6000 Blackwell

## Quick Start

**117 tok/s** single user with MTP=3, **2,084 tok/s** at 32 concurrent users, **351 tok/s** at c=4 ctx=131k. Runs on **2× RTX PRO 6000** (96 GiB each) using FP8 W8A8 quantization on the Repne vLLM fork.

```bash
docker pull repne/vllm:v13
```

```bash
docker run -d --name qwen-vllm-fp8-tp2 --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  --volume "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --volume "$HOME/.cache/vllm:/root/.cache/vllm" \
  --volume "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  --volume "$HOME/.triton/cache:/root/.triton/cache" \
  --env OMP_NUM_THREADS=16 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env NCCL_P2P_LEVEL=SYS --env NCCL_NET_GDR_LEVEL=SYS --env NCCL_MIN_NCHANNELS=8 \
  repne/vllm:v13 \
    serve -O3 --model Qwen/Qwen3.6-27B-FP8 --served-model-name Qwen3.6-27B \
    --port 11435 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.88 \
    --max-model-len 134144 --max-num-seqs 128 --max-num-batched-tokens 32768 \
    --max-cudagraph-capture-size 256 --language-model-only \
    --enable-auto-tool-choice --enable-prefix-caching \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 \
    --attention-backend flashinfer
```

API endpoint: `http://localhost:11435/v1` (OpenAI-compatible). First run loads ~29 GB FP8 weights + JIT-compiles kernels (~120 s warm cache, ~295 s cold).

Without speculative decoding: remove the three `--speculative-config.*` flags. Expect ~30% lower single-stream throughput and ~10% lower aggregate throughput at c=32.

## Recommended Checkpoints

Three Qwen3.6-27B checkpoints are validated on 2× RTX PRO 6000:

| Checkpoint | Quant | Quality | Speed (c=1) | Best for |
|---|---|---|---|---|
| `Qwen/Qwen3.6-27B-FP8` | FP8 W8A8 (block-128) | mean KL 5.5e-3 bits vs BF16 (7.3× noise floor; Exp 11) | **117 tok/s** (MTP=3) | **Production** — fastest aggregate throughput |
| `Qwen/Qwen3.6-27B` | BF16 | reference | 91 tok/s (DFlash N=8) | Lossless quality with DFlash speculative decoding |
| `z-lab/Qwen3.6-27B-DFlash` | BF16 drafter | n/a | (drafter only) | Required by BF16+DFlash configuration |

> **NVFP4 disqualified.** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (the only community NVFP4 quant available) wins +12–29% at short context but is **broken at 244K context** and 50% worse on the Repne fork than on upstream. See [Known Issues §6](#known-issues-and-fixes).

## Benchmark Results

### FP8+MTP=3 (`repne/vllm:v13`, Qwen3.6-27B-FP8, TP=2)

Aggregate throughput (tok/s), N=5 reps per cell:

```
ctx\conc    1      2      4      8      16      32
   0      117.1  227.2  449.8  875.0  1520.6  2083.7
  16k      —      —      —      —    1186.0  1892.3
  32k     119.2  227.0  454.5  795.4  1186.0  1656.3
  64k      —      —      —      —    1047.1   —
 131k      95.0  184.9  350.5  534.4   661.3   —
```

### BF16+DFlash N=8 (`repne/vllm:v13`, Qwen3.6-27B + DFlash drafter, TP=2)

```
ctx\conc    1      2      4      8      16      32
   0       90.9  171.4  325.8  579.8   884.1  1122.9
  16k      89.2  174.9  327.9  568.1   859.7  1085.2
  32k      91.0  169.1  324.1  542.7   824.5  1015.5
  64k      88.5  165.9  320.4  529.3   768.1   929.6
 131k      82.3  161.5  290.9  471.8   661.3   787.0
```

### Best Configuration by Scenario

| Scenario | Setup | tok/s |
|---|---|---:|
| Single user, max speed | FP8+MTP=5 (Repne) | **119.9** (c=1×0) |
| Single user, deep context (131k) | FP8+MTP=5 (Repne) | **101.2** (c=1×131k) |
| Multi-user c=8 | FP8+MTP=3 (Repne) | **875** |
| Multi-user c=16 | FP8+MTP=3 (Repne) | **1,521** |
| **Multi-user c=32** | **FP8+MTP=3 (Repne)** | **2,084** ⭐ |
| BF16 quality, c=32 | BF16+DFlash N=8 (Repne) | 1,123 |

---

## Table of Contents

- [Overview](#overview)
- [All Checkpoints](#all-checkpoints)
- [Hardware Requirements](#hardware-requirements)
- [NCCL Environment Variables](#nccl-environment-variables)
- [Launch Commands -- Repne fork](#launch-commands----repne-fork)
- [Launch Commands -- Upstream vLLM](#launch-commands----upstream-vllm)
- [Docker Images](#docker-images)
- [MTP / Speculative Decoding](#mtp--speculative-decoding)
- [BF16 + DFlash Speculative Decoding](#bf16--dflash-speculative-decoding)
- [Quantization Details](#quantization-details)
- [Tool-call & Reasoning Parsers](#tool-call--reasoning-parsers)
- [Detailed Benchmark Tables](#detailed-benchmark-tables)
- [Memory Usage (VRAM)](#memory-usage-vram)
- [Known Issues and Fixes](#known-issues-and-fixes)
- [Methodology Archive (experiment index)](#methodology-archive-experiment-index)

---

## Overview

Qwen3.6-27B is a 27B-parameter dense decoder-only model from Qwen, released in early 2026.

| Parameter | Value |
|---|---|
| Total parameters | 27B (dense, not MoE) |
| Active parameters | 27B |
| Attention | Standard grouped-query attention |
| Max context (native) | 262 144 tokens |
| Tested context (this study) | up to 131 072 tokens |
| Tool-call format | XML or coder (vLLM parser choice — see §[Tool-call & Reasoning Parsers](#tool-call--reasoning-parsers)) |

This study characterizes Qwen3.6-27B inference on the dual-GPU **RTX PRO 6000 Blackwell** SM120 platform across the Repne vLLM fork, upstream vLLM, and llama.cpp. See [Methodology Archive](#methodology-archive-experiment-index) for the chronological experiment narrative.

---

## All Checkpoints

| Checkpoint | Quantization | KV Cache | Notes |
|---|---|---|---|
| **`Qwen/Qwen3.6-27B-FP8`** | **FP8 W8A8 block-128** | FP8 | **Recommended.** Official Qwen. Teacher-forced KLD vs BF16: **mean KL 5.5e-3 bits/pos** (single-prompt; 7.3× BF16 noise floor), **5.52e-3 bits** single / **0.230 bits** multi (Exp 11). 2× GPUs TP=2. |
| `Qwen/Qwen3.6-27B` | BF16 | BF16 | Lossless reference. Required for BF16+DFlash. 2× GPUs TP=2. |
| `z-lab/Qwen3.6-27B-DFlash` | BF16 drafter | — | Speculative draft model for BF16+DFlash. Loaded alongside main BF16 model. |
| `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | NVFP4 (modelopt) | FP8 | Community quant. 15 mtp.* tensors graft cleanly, functional gates pass. **Broken at 244K context.** Repne fork makes it 50% worse. Not production-viable. |
| `unsloth/Qwen3.6-27B-GGUF` Q8_0 | INT8 GGUF | FP16 | Quality probe only (llama.cpp). KLD vs BF16 = 0.0018 nats (noise floor). |
| `unsloth/Qwen3.6-27B-GGUF` BF16 | BF16 GGUF (split files) | FP16 | Perplexity reference for the KLD probe. |

---

## Hardware Requirements

The primary tested hardware is **RTX PRO 6000 Blackwell Workstation Edition** (96 GiB VRAM each, PCIe Gen5 x16, no NVLink).

| Configuration | Model / Quant | Notes |
|---|---|---|
| **2× RTX PRO 6000** | FP8, TP=2 | Most common setup. ~37 GiB per GPU at 134k max context. |
| **2× RTX PRO 6000** | BF16, TP=2 | ~73 GiB per GPU with BF16 + DFlash drafter. |
| **3× RTX PRO 6000** (1 display + 2 compute) | FP8 or BF16, TP=2 on GPUs 1+2 | This study's actual hardware. GPU 0 reserved for display, kept under 4 GiB. |

### GPU Topology (2× PCIe Gen5 x16, no NVLink)

```
        GPU0    GPU1    GPU2
GPU0     X      NODE    NODE
GPU1    NODE     X      NODE
GPU2    NODE    NODE     X
```

All GPUs connected via PCIe through NUMA node (no NVLink). Verified at full **PCIe Gen5 (32 GT/s) x16** under active GPU compute pressure; idle reads of `pcie.link.gen.current = 1` reflect ASPM downscaling, not a hardware fault.

### Driver / CUDA Versions

- NVIDIA Driver: **595.71.05**
- CUDA Version: **13.2** (host); **13.2.1** (container)
- VRAM per GPU: 97,887 MiB
- Power limit: 600 W per card (RTX PRO 6000 Workstation; the 280 W variant is the Max-Q SKU)
- Host OS: Linux 7.0.2-arch1-1, 251 GB system RAM, Intel Xeon W (W790E-SAGE motherboard)

---

## NCCL Environment Variables

### Required for 2× PCIe setup

```bash
NCCL_P2P_LEVEL=SYS         # Recommended on this study's hardware (Gen5 x16, PCIe through NUMA)
NCCL_NET_GDR_LEVEL=SYS     # GPU-direct RDMA level
NCCL_MIN_NCHANNELS=8       # More channels than default helps TP=2 throughput
```

### vLLM environment variables (all required)

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn   # Required for vLLM multi-GPU
VLLM_ALLREDUCE_USE_SYMM_MEM=0        # Disable symmetric-memory allreduce (incompatible with PCIe TP=2 on Blackwell)
OMP_NUM_THREADS=16                   # Limit OpenMP threads (cores per GPU rank)
```

### Diagnosing NCCL deadlocks

If GPUs hit 100% utilization at ~140 W with no VRAM growth, suspect an NCCL deadlock:

1. Check topology: `nvidia-smi topo -m`
2. Try `NCCL_P2P_LEVEL=PHB` instead of `SYS`
3. Or disable P2P entirely: `unset NCCL_P2P_LEVEL; NCCL_P2P_DISABLE=0`
4. Enable debug: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`

---

## Launch Commands -- Repne fork

The recommended FP8+MTP=3 launch command is in [Quick Start](#quick-start) above.

### BF16 + DFlash N=8, 2× GPUs (1,123 tok/s peak at c=32×0)

```bash
docker run -d --name qwen-vllm-bf16-tp2 --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  --volume "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --env OMP_NUM_THREADS=16 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env NCCL_P2P_LEVEL=SYS --env NCCL_NET_GDR_LEVEL=SYS \
  repne/vllm:v13 \
    serve -O3 --model Qwen/Qwen3.6-27B --served-model-name Qwen3.6-27B \
    --port 11435 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.88 \
    --max-model-len 134144 --max-num-seqs 128 \
    --enable-auto-tool-choice --enable-prefix-caching \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml \
    --speculative-config.method draft_model \
    --speculative-config.model z-lab/Qwen3.6-27B-DFlash \
    --speculative-config.num_speculative_tokens 8 \
    --speculative-config.draft_sample_method gumbel \
    --speculative-config.use_local_argmax_reduction true \
    --attention-backend flashinfer
```

> **DFlash note:** `num_speculative_tokens=8` is the throughput optimum on this hardware. `N=7` is 1.8% slower, `N=15` is 10.6% slower (the drafter struggles further into the chain; aggregate acceptance dilutes). See [Experiment 6](./06-new-image-validation/) and [Experiment 9](./09-v13-kitchen-sink/).

### Repne-fork-only flags (do not work on upstream)

| Flag | Effect | Required for |
|---|---|---|
| `--speculative-config.draft_sample_method gumbel` | Gumbel-noise draft sampling instead of greedy | BF16+DFlash (5–6× long-context speedup) |
| `--speculative-config.use_local_argmax_reduction true` | Drafter argmax-reduction optimization | BF16+DFlash |
| `--load-format instanttensor` | Faster weight-loading path | All configs (optional, +30 s boot speed) |
| `--speculative-config.attention_backend flash_attn` | Independent attention backend for spec path | BF16+DFlash on certain Blackwell kernels |
| `-O3` | Aggressive compile-time optimization flag | All configs |
| `--default-chat-template-kwargs.preserve_thinking true` | Preserve `<think>` tokens through chat templating | Reasoning-mode workloads |

Dropping `gumbel` and `use_local_argmax_reduction` on BF16+DFlash causes drafter acceptance to collapse from 23–30% to 1–3% at ctx > 32k, and inter-token latency inflates 8× (11 ms → 90 ms). **These flags are doing real work.**

---

## Launch Commands -- Upstream vLLM

Upstream is a viable fallback **only** for FP8+MTP=3 (5–14% short-context cost, parity long-context). **Do not** try to reproduce BF16+DFlash on upstream — the upstream drafter implementation collapses past 32k.

### FP8+MTP=3 on upstream vLLM v0.20.1

```bash
docker run -d --name qwen-vllm-upstream-fp8 --gpus all --ipc=host --shm-size=32g \
  --network host \
  --volume "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env NCCL_P2P_LEVEL=SYS \
  vllm/vllm-openai:v0.20.1-cu129-ubuntu2404 \
    --model Qwen/Qwen3.6-27B-FP8 --served-model-name Qwen3.6-27B \
    --port 11435 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.88 \
    --max-model-len 134144 --max-num-seqs 128 \
    --enable-auto-tool-choice --enable-prefix-caching \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### Upstream flags reference

| Flag | Notes |
|---|---|
| `--enable-expert-parallel` | **DO NOT USE.** This is a dense model, not MoE. Flag is irrelevant; included only because users from MoE workloads sometimes copy it over. |
| `--language-model-only` | Only available on Repne fork. Saves ~12 s TTFT on first request by skipping vision encoder load. |
| `VLLM_ALLREDUCE_USE_SYMM_MEM=0` | Required on Blackwell PCIe TP=2 — symmetric-memory allreduce hangs without NVLink. |

---

## Docker Images

| Image | Status | Notes |
|---|---|---|
| **`repne/vllm:v13`** | **Recommended** | Current production image. Combines fork-side MTP/DFlash optimizations with upstream's scheduler fixes. Engine `v0.1.dev16400+g910d87a9d`. |
| `repne/vllm:latest` (`d0a200f77546`, May 5 2026 evening) | Validated | Used in Experiments 6–8. +41 commits over the May 5 morning image. |
| `repne/vllm:latest` (`5e7583ca4df9`, May 5 2026 morning) | Validated | Used in Experiments 1–5. |
| `vllm/vllm-openai:v0.20.1-cu129-ubuntu2404` | Validated fallback | Used for upstream head-to-heads (Experiments 4, 5) and NVFP4 retry. |
| `vllm/vllm-openai:v0.19.1-cu130-ubuntu2404` | **Avoid** | Catastrophic crash at ctx > 131k with NVFP4 modelopt: `NV_ERR_GPU_IN_FULLCHIP_RESET`. Required host reboot. |
| `vllm/vllm-openai:cu130-nightly` | **Broken** (as of May 2026) | `ModuleNotFoundError: No module named 'pandas'` in `_aiter_ops.py`. |

---

## MTP / Speculative Decoding

### MTP flags -- Repne fork

```
--speculative-config.method mtp
--speculative-config.num_speculative_tokens 3
--speculative-config.draft_sample_method gumbel    # optional, +5-14% throughput
```

### MTP flags -- Upstream vLLM v0.20.1+

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### Key findings

- **MTP=3 is the production sweet spot** for FP8 on this dual-GPU setup. Wins at c≥8 by 0.4–21.2% over MTP=5, and 8.7–21.2% over MTP=5 at c=16+.
- **MTP=5 wins at c=1–4** by 1.8–6.5% (largest win: c=1×131k = 101.2 vs 95.0 tok/s). Holds isolated single-stream records.
- **Crossover at c=8.** Above c=8, MTP=3 dominates; below, MTP=5 has small advantage.
- **MTP=4 / MTP=6** are dominated by MTP=3 at every concurrency tier we tested.
- **Speculative decoding is not bitwise-lossless even at temp=0.** Comparing FP8+MTP=3 vs FP8+no-spec on identical prompts at `temperature=0, top_p=1.0, seed=42`, 4 of 8 prompts produce divergent text. Differences are stylistic, not factual — all functional gates pass. The Repne fork's `gumbel` draft sampler takes a different RNG path through accept/reject than greedy. [→ Experiment 7 Phase A](./07-quality-sprint/)

### MTP=3 vs no-spec speedup (Repne fork, FP8, ctx=0)

| Concurrency | no-spec | MTP=3 | MTP=3 speedup |
|---|---:|---:|---:|
| c=8 | 573.6 | 875.0 | **1.53×** |
| c=16 | 1,126.7 | 1,520.6 | **1.35×** |
| c=32 | 1,875.5 | 2,083.7 | **1.11×** |

The MTP=3 advantage compresses as concurrency rises (less spare GPU compute to amortize spec misses) but remains positive everywhere.

### MTP acceptance rates (FP8+MTP=3, Repne fork, averaged across 30-cell matrix)

- Mean spec acceptance rate: **56.6%**
- ITL at c=1: 10.6 ms (BF16+DFlash) / 8.5 ms (FP8+MTP=3)
- ITL at c=32 ctx=131k: 38 ms (BF16+DFlash) — comfortably under 50 ms streaming threshold

---

## BF16 + DFlash Speculative Decoding

DFlash is Repne's speculative-decoding scheme using a separate small drafter model (`z-lab/Qwen3.6-27B-DFlash`) instead of MTP heads grafted onto the main model. It targets BF16 workloads where MTP isn't available.

### DFlash flags -- Repne fork

```
--speculative-config.method draft_model
--speculative-config.model z-lab/Qwen3.6-27B-DFlash
--speculative-config.num_speculative_tokens 8
--speculative-config.draft_sample_method gumbel             # critical
--speculative-config.use_local_argmax_reduction true        # critical
--attention-backend flashinfer
```

### Key findings

- **DFlash N=8 is the throughput optimum** (197.5 tok/s mean across 9 cells in Exp 06; 1,122.9 peak at c=32×0 in Exp 09).
- **N=7 is 1.8% slower** than N=8 across the 9-cell c×ctx matrix.
- **N=15 is 10.6% slower than N=7 / N=8.** Repne's "0.931→0.063 acceptance distribution per position" claim doesn't translate to aggregate throughput wins — the aggregate `server_spec_accept_rate` dilutes when most positions reject.
- **Mean spec acceptance rate at N=8: 23.1%** (across Exp 09's 30-cell matrix).
- **DFlash dominates upstream long-context.** Upstream vLLM's DFlash collapses past 32k context (drafter acceptance falls from 30% → 1–3%, ITL inflates 8×). Repne fork sustains 81–284 tok/s at c=1–4×131k vs upstream's 12–45 tok/s — a **+537–626% gap**. See [Experiment 5](./05-bf16-dflash-headtohead/).

---

## Quantization Details

### FP8 W8A8 (recommended)

- Uses `Qwen/Qwen3.6-27B-FP8` checkpoint (official Qwen)
- Block-128 W8A8 (weights and activations both FP8 e4m3, blocks of 128)
- FP8 KV cache with calibrated scales
- ~37 GiB per GPU at 134k max context (TP=2)
- **Quality:** teacher-forced KLD vs BF16 = **5.5e-3 bits/pos** single-prompt (7.3× BF16 noise floor of 7.6e-4 bits/pos); **0.230 bits/pos** multi-prompt (8.6× noise floor). See [Experiment 11](./11-teacher-forced-kld/). Q8 GGUF proxy was 0.0018 nats ≈ 0.0026 bits — a lower bound; the real FP8 vLLM number is ~2× higher in single-prompt mode.
- 8/8 functional gates pass (Fibonacci 5x deterministic, tool-calling, multi-turn coherence, 47×83=3901 arithmetic)

### BF16

- Uses `Qwen/Qwen3.6-27B` checkpoint
- ~73 GiB per GPU with DFlash drafter loaded (TP=2)
- Reference quality — perplexity ratio = 1.000
- Required if exact BF16 quality is needed (e.g., regulated environments)

### NVFP4 (disqualified)

- Uses `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (only community NVFP4 quant; `--quantization modelopt`)
- Fits comfortably on 2× GPUs (~16 GiB per GPU)
- **Wins +12–29% at short context, loses 27–33% at 131k context, broken at 244k.**
- Production target is 256k context → NVFP4 disqualified.
- On Repne fork: 50% **worse** than upstream — Repne's `gumbel`/`use_local_argmax_reduction` optimizations are W8A8/BF16-specific.
- The community-quantized model itself is correct (15 mtp.* tensors graft cleanly, all functional gates pass). Bottleneck is engine-side, not model quality.
- See [Experiment 3](./03-nvfp4-mtp-experiment/) and [Experiment 7 Phase D](./07-quality-sprint/).

### Q8_0 GGUF (quality probe only, llama.cpp)

- Used only for the KL-divergence quality probe ([Experiment 8 Y1](./08-x1y1-sprint/))
- Wikitext-2 102 200-token perplexity: **7.623 ± 0.063** vs BF16 reference **7.620 ± 0.062** (ratio 1.0004, +0.04%)
- Mean KLD vs BF16: **0.001828 ± 0.000189 nats**
- 97.9% top-token agreement with BF16
- Establishes that 8-bit weight quantization on this model is empirically at the noise floor

---

## Tool-call & Reasoning Parsers

vLLM ships two tool-call parsers for Qwen3-family models. Both are valid; they differ in output format.

| Parser | Output format | When to use |
|---|---|---|
| `qwen3_xml` | XML-tagged: `<tool_call>{"name": "get_weather", "arguments": {...}}</tool_call>` | vLLM v13 default for Qwen3-family. Required if your client expects XML tool calls. |
| `qwen3_coder` | JSON in a coder-shaped wrapper | Used in Experiments 01–08 of this study. Required if your client expects JSON. |

> **Experiment 10** ran a head-to-head of `qwen3_xml` vs `qwen3_coder` on `tool-eval-bench`'s full 69-scenario suite, plus a v13-vs-nightly image-axis tournament and 7 frontier API yardsticks. **Result: tie** — both parsers score 62/100 on the full 69-scenario suite (responsiveness 80 each). Either is production-suitable; pick by client expectation. See [`10-parser-axis/`](./10-parser-axis/) for the full tournament, including the surprising **+14 point image-axis result** (`repne/vllm:v13` beats upstream nightly even on FP8+MTP=3 where DFlash is not a factor) and the frontier comparison.

Reasoning parser is unambiguous:

```
--reasoning-parser qwen3
```

To disable thinking-mode output entirely:

```
--default-chat-template-kwargs '{"enable_thinking": false}'
```

---

## Detailed Benchmark Tables

### FP8+MTP=3 (Repne fork, N=5)

Aggregate throughput in tok/s, mean ± std across 5 reps:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 117.1 ±2.8 | 227.2 ±2.4 | 449.8 ±4.6 | 875.0 ±11.5 | 1520.6 ±2.5 | **2083.7** ±12.6 |
| 16k | — | — | — | 795.4 ±0.9 | 1186.0 ±7.5 | 1892.3 ±4.1 |
| 32k | 119.2 ±4.7 | 227.0 ±5.1 | 454.5 ±3.3 | — | — | 1656.3 ±13.6 |
| 64k | — | — | — | — | 1047.1 ±4.8 | — |
| 131k | 95.0 ±2.5 | 184.9 ±0.9 | 350.5 ±4.9 | 534.4 ±7.4 | 661.3 ±14.5 | — |

### BF16+DFlash N=8 (Repne fork, N=5)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 90.9 ±2.2 | 171.4 ±4.2 | 325.8 ±2.8 | 579.8 ±5.3 | 884.1 ±8.4 | 1122.9 ±4.8 |
| 16k | 89.2 ±2.1 | 174.9 ±2.7 | 327.9 ±6.7 | 568.1 ±8.3 | 859.7 ±16.1 | 1085.2 ±12.4 |
| 32k | 91.0 ±4.4 | 169.1 ±3.0 | 324.1 ±6.8 | 542.7 ±8.4 | 824.5 ±7.6 | 1015.5 ±8.3 |
| 64k | 88.5 ±2.3 | 165.9 ±4.0 | 320.4 ±6.8 | 529.3 ±6.5 | 768.1 ±8.8 | 929.6 ±11.1 |
| 131k | 82.3 ±1.5 | 161.5 ±4.4 | 290.9 ±7.4 | 471.8 ±6.0 | 661.3 ±14.5 | 787.0 ±7.5 |

### MTP=3 vs MTP=5 (FP8, Repne fork) — concurrency crossover

| Concurrency × ctx | MTP=3 tok/s | MTP=5 tok/s | MTP=5 advantage |
|---|---:|---:|---:|
| c=1 × 0 | 117.1 | 119.9 | +2.4% |
| c=1 × 131k | 95.0 | **101.2** ⭐ | **+6.5%** |
| c=2 × 0 | 227.2 | 234.4 | +3.2% |
| c=4 × 0 | 449.8 | 462.5 | +2.8% |
| c=8 × 0 | 875.0 | 865.8 | −1.1% (MTP=3 retakes) |
| c=16 × 0 | 1,520.6 | 1,329.7 | **−12.6%** |
| c=32 × 0 | 2,083.7 | 1,726.5 | **−17.1%** |

### Repne fork vs Upstream vLLM v0.20.1

**BF16+DFlash N=8 (long-context):**

| concurrency × ctx | Repne tok/s | Upstream tok/s | Δ |
|---|---:|---:|---:|
| c=1 × 131k | **81.4** | 11.7 | **+598%** 🚨 |
| c=2 × 131k | **162.7** | 22.4 | **+626%** 🚨 |
| c=4 × 131k | **284.4** | 44.7 | **+537%** 🚨 |

Upstream collapses because flashinfer rejects the non-causal attention pattern the drafter requires, forcing flash_attn. Upstream lacks Repne's `gumbel` sampler and `use_local_argmax_reduction`.

**FP8+MTP=3 (much narrower margin):**

| concurrency × ctx | Repne tok/s | Upstream tok/s | Δ |
|---|---:|---:|---:|
| c=1 × 0 | **120.1** | 112.3 | +7.0% |
| c=2 × 0 | **223.8** | 197.0 | +13.6% |
| c=4 × 0 | **449.5** | 413.8 | +8.6% |
| c=4 × 131k | **347.4** | 345.0 | +0.7% |

FP8+MTP=3 on upstream is a viable fallback (5–14% short-context cost, parity long-context).

---

## Memory Usage (VRAM)

- **FP8 on 2× GPUs (TP=2):** ~37 GiB per GPU at 134k max context
- **BF16 on 2× GPUs (TP=2) with DFlash drafter:** ~73 GiB per GPU
- `--gpu-memory-utilization 0.85–0.88` typical range (this study used 0.88; raise to 0.91 if you need more KV)
- Each RTX PRO 6000 Blackwell: **97,887 MiB** total VRAM
- `--language-model-only` (Repne only) reduces TTFT from 12 s to <1 s on first request by skipping vision encoder load — required since Qwen3.6-27B is text-only on this checkpoint

### Engine state at production config (FP8+MTP=3)

- KV cache: **1,846,472 tokens** (7.04× max concurrency at 256k context)
- Boot time: **120 s** warm compile cache, **295 s** cold
- All functional gates: **4/4 pass**

---

## Known Issues and Fixes

### 1. NVFP4 long-context engine failure

**Symptom:** NVFP4+MTP=3 throughput collapses to 1.4 tok/s at c=1×244k on `vllm/vllm-openai:v0.20.1`. Effectively unusable.

**Cause:** Engine-side, not model quality. The community-quantized weights pass all functional gates (Fibonacci 5x, tool-calls, 47×83=3901, 137K-token needle-in-haystack found in 23.3 s on the Repne fork).

**Workaround:** Use FP8 W8A8 instead. NVFP4 is permanently disqualified for production until either the upstream NVFP4 long-context path is fixed or the Repne fork adds W8A8-equivalent NVFP4 optimizations.

[→ Experiment 3](./03-nvfp4-mtp-experiment/), [Experiment 7 Phase D](./07-quality-sprint/)

### 2. `vllm/vllm-openai:v0.19.1` GPU reset crash

**Symptom:** `NV_ERR_GPU_IN_FULLCHIP_RESET` at ctx > 131k with NVFP4 modelopt. Required host reboot.

**Fix:** Upgrade to `vllm/vllm-openai:v0.20.1-cu129-ubuntu2404` or use Repne fork.

### 3. `vllm/vllm-openai:cu130-nightly` pandas import error

**Symptom:** `ModuleNotFoundError: No module named 'pandas'` in `_aiter_ops.py` on container start.

**Fix:** Avoid this image entirely as of May 2026. Use `repne/vllm:v13` or `vllm/vllm-openai:v0.20.1`.

### 4. Scheduler pessimal pocket on Repne morning image

**Symptom:** Default `max-num-seqs=128 + max-num-batched-tokens=32 768` produced a −5.7% c=4 ctx=0 regression on the May 5 morning image vs prior baselines.

**Fix:** Nearly any deviation from the exact default pair recovers throughput. The evening image (`d0a200f77546`) and v13 do not exhibit this. [→ Experiment 2](./02-scheduler-investigation/)

### 5. Tool-call format changes with MTP enabled (upstream)

**Symptom:** On upstream vLLM, model outputs XML tool calls when `tool_choice='required'` and MTP is on; 50–70% of tool calls fail JSON parsing.

**Fix:** Use `tool_choice='auto'` (handles both XML and JSON), or disable thinking mode (`--default-chat-template-kwargs '{"enable_thinking": false}'`). On the Repne fork this is not observed at the same rate; thinking-mode + MTP is the trigger upstream.

### 6. NCCL deadlock with `VLLM_ALLREDUCE_USE_SYMM_MEM=1`

**Symptom:** Hang at startup with GPUs at 100% utilization, ~140 W power, no VRAM growth.

**Fix:** Always set `VLLM_ALLREDUCE_USE_SYMM_MEM=0` on Blackwell PCIe TP=2 (no NVLink). Symmetric-memory allreduce requires NVLink topology.

### 7. PCIe Gen1 reading at idle (not a hardware fault)

**Symptom:** `nvidia-smi` startup snapshot reports `pcie.link.gen.current = 1`.

**Cause:** ASPM (Active State Power Management) downscaling at idle. Verified: PCIe link trains to **Gen5 (32 GT/s) x16** under active GPU compute pressure.

---

## Methodology Archive (experiment index)

This study was a 24-hour development sprint of **eight controlled experiments** (Experiments 1–8), followed by three later experiments (Experiment 9: v13 image kitchen sink; Experiment 10: tool-call parser axis; Experiment 11: teacher-forced decode KLD/JSD). Each experiment has its own subdirectory with raw `.json` per-cell results, `bench.log` files with full bench-tool stdout, and a `RESULTS.md` write-up.

| # | Path | Question | Method | Outcome |
|---|---|---|---|---|
| 1 | [`01-morning-newimage-validation/`](./01-morning-newimage-validation/) | Does the new Repne May 5 image regress the May 3 baseline? | BF16+DFlash, 5 cells × N=5 randomized = 25 runs | Mixed: −5.7% c=4 ctx=0, +9.0% c=1 ctx=128k. Triggered Exp 2. |
| 2 | [`02-scheduler-investigation/`](./02-scheduler-investigation/) | What scheduler knobs cause the c=4 ctx=0 regression? | 5 variants × N=3 | Default `max-num-seqs=128 + max-num-batched-tokens=32 768` is in a pessimal pocket; nearly any deviation recovers throughput. |
| 3 | [`03-nvfp4-mtp-experiment/`](./03-nvfp4-mtp-experiment/) | Can NVFP4 community-quant replace FP8? | 11 cells × N=3 + 4 functional gates | No. Wins short-context, broken at 244k. |
| 4 | [`04-fp8-mtp3-headtohead/`](./04-fp8-mtp3-headtohead/) | Repne vs upstream on FP8+MTP=3? | 6 cells × N=1 | Repne +5–14% short-context, parity at long-context. |
| 5 | [`05-bf16-dflash-headtohead/`](./05-bf16-dflash-headtohead/) | Repne vs upstream on BF16+DFlash? | 6 cells × N=1 | Repne +537–626% at long-context (upstream collapses). |
| 6 | [`06-new-image-validation/`](./06-new-image-validation/) | New Repne image (`d0a200f7`) FP8 vs DFlash variants? | 9 cells × 4 configs × N=3 = 108 runs | FP8+MTP=3 wins all 9 cells. Best DFlash variant: N=7 (but N=8 in v13). |
| 7 | [`07-quality-sprint/`](./07-quality-sprint/) | Triggered by Phaelon Discord on W8A8 quality. | A: spec losslessness. B: FP8 vs Q8 GGUF. C: MTP n∈{2,3,4,5,6}. D: NVFP4 on Repne. | A: not bitwise-lossless. B: no FP8 quality regression. C: MTP=5 wins at c=1–4 (caveat triggered Exp 8). D: NVFP4 50% worse on Repne. |
| 8 | [`08-x1y1-sprint/`](./08-x1y1-sprint/) | High-concurrency speed (X1) + perplexity quality (Y1). | X1: c∈{8,16,32} × ctx ∈ {0,32k,131k}. Y1: AesSedai KLD. | **MTP=3 wins +10.5% mean at c≥8.** Q8/BF16 KLD = 0.0018 (noise floor). |
| 9 | [`09-v13-kitchen-sink/`](./09-v13-kitchen-sink/) | Characterize the new `repne/vllm:v13` image end-to-end. | 5×6 matrix × 2 configs × N=5 = 300 cells + 16 method-validation + 4 quality probes | BF16+DFlash N=8 peak 1,123 tok/s; FP8+MTP=3 peak 2,146 tok/s; tool-calling **93/100 on the 15-scenario `--short` subset** of tool-eval-bench (using `qwen3_xml`). Experiment 10 measures the same config on the **full 69-scenario suite** and scores 62/100 — the two numbers are not comparable (different scoring denominators / different scenario distributions). |
| 10 | [`10-parser-axis/`](./10-parser-axis/) | `qwen3_xml` vs `qwen3_coder` parser head-to-head + image axis + frontier yardsticks. | Staged tournament: parser axis on v13+FP8 → image axis on BF16 → FP8 generalization. 7 frontier API endpoints as yardsticks. | Parser axis: **tie** (both 62/100). BF16 image axis: **v13 uncontested** (nightly does not register `DFlashDraftModel`). FP8 image axis: **v13 wins +14 points** over nightly. Frontier yardsticks: Claude Sonnet 4.6 = 86, Gemini 3.5 Flash = 86, GPT-5.5 = 83, Claude Haiku 4.5 = 82, Qwen-235B (Cerebras) = 81, GPT-5-mini = 72; GPT-5-nano timed out. Local Qwen3.6-27B at 62 sits ~27% behind frontier on quality but faster than every API except Cerebras Qwen-235B. |
| 11 | [`11-teacher-forced-kld/`](./11-teacher-forced-kld/) | How closely do FP8 and NVFP4 match BF16 at every decoded position (teacher-forced decode KLD/JSD)? | 8 cells (BF16-ref × 2 + BF16-self noise floor × 2 + FP8 × 2 + NVFP4 × 2) on WikiText-2-raw-v1 via GLM-5.1 methodology. TP=2, GPUs 1+2. | **FP8: 7.3× noise floor** (mean KL 5.5e-3 bits, single-prompt). **NVFP4: 395× noise floor** (mean KL 2.99e-1 bits). FP8-MTP=3 hard-skipped: vLLM V1 rejects custom logits processors with spec decode (`ValueError`; see `SKIP_REASON.md`). |
| 13 | [`13-v17-decode-matrix/`](./13-v17-decode-matrix/) | Baseline decode/prefill sweep on the new `repne/vllm:v17` image. | Single-pass (N=1) 41-cell decode matrix + prefill, BF16+DFlash N=8, TP=2, `--gpu-memory-utilization 0.80`, `--kv-budget 854152`. Harness `llm-inference-bench v0.4.24`. | v17 build `0.1.dev17236+g50272be4a`. Peak **1,138 tok/s @ c=32×0**; prefill 7,803→5,994 tok/s (8k→128k). Not a SOTA attempt — single run at 0.80 mem fraction; does not beat Exp 08 (2,084 tok/s). Power avg 813 W / max 1,117 W. |
| 14 | [`14-v17-tp1-decode-matrix/`](./14-v17-tp1-decode-matrix/) | Same v17 sweep as Exp 13 but **TP=1 (single GPU)** — tensor-parallel A/B. | Single-pass decode + prefill, BF16+DFlash N=8, **TP=1**, `--gpu-memory-utilization 0.80`, `--max-model-len 131072` (single-card KV cap), `--kv-budget 134606`. | Peak **407 tok/s @ c=16×0** (vs 1,138 on TP=2 — **2.8×** gap). Single-card KV holds only 134,606 tokens → **26/41 cells KV-skipped**; full 262k context does not fit at 0.80 mem. TP=2 wins on both throughput and capacity. |

Companion documents:

- [`SOTA.md`](./SOTA.md) — full SOTA leaderboard, cross-quant comparison, decision tree
- [`PERFORMANCE_CHART.md`](./PERFORMANCE_CHART.md) — text-mode bar charts of throughput scaling
- [`master-results.csv`](./master-results.csv) — 328-row machine-readable summary of every cell × run

### Companion standalone repositories

Published as standalone artifacts during the original sprint for fast sharing:

- [`jcartu/repne-dflash-newimage`](https://github.com/jcartu/repne-dflash-newimage) — initial Repne May 5 image validation
- [`jcartu/qwen36-27b-nvfp4-mtp-experiment`](https://github.com/jcartu/qwen36-27b-nvfp4-mtp-experiment) — NVFP4+MTP write-up
- [`jcartu/qwen36-27b-fp8-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-fp8-repne-vs-upstream) — FP8+MTP=3 head-to-head
- [`jcartu/qwen36-27b-bf16-dflash-repne-vs-upstream`](https://github.com/jcartu/qwen36-27b-bf16-dflash-repne-vs-upstream) — BF16+DFlash head-to-head

This monorepo is the canonical comprehensive record.

---

## Recommendation

**Adopt FP8+MTP=3 on `repne/vllm:v13` as the production configuration** for any Qwen3.6-27B inference deployment on RTX PRO 6000 Blackwell SM120 hardware. Verified at every production-relevant concurrency tier (c=1 through c=32) across short, medium, and long context. Peak aggregate throughput **2,083.7 tok/s at c=32 ctx=0**. Quality now directly measured via teacher-forced decode KLD (Exp 11): FP8 mean KL = **5.5e-3 bits/pos** (single-prompt), 7.3× the BF16 self-noise floor. NVFP4 is 395× the noise floor and disqualified. 8/8 functional gates pass for FP8.

If the Repne fork ever stops shipping new images, **upstream `vllm/vllm-openai:v0.20.1` with FP8+MTP=3** is the safest fallback — accept 5–14% short-context throughput cost. **Do not attempt** to recreate the BF16+DFlash performance gap on upstream without forking back `gumbel` and `use_local_argmax_reduction`.

NVFP4 quantization is permanently disqualified by long-context engine instability and Repne-fork incompatibility. DFlash variants don't reach FP8+MTP=3 throughput at any tested concurrency tier. MTP n∈{2,4,5,6} are dominated by n=3 at c≥8.
