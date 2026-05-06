# NVFP4-MTP Experiment — PARTIAL RESULTS

**Status:** Halted at Phase 4. GPU0 hung after c=1 ctx=250k cell crash. Host reboot required.

## Setup

- Image: `vllm/vllm-openai:v0.19.1-cu130-ubuntu2404` (digest `sha256:f35754668e5f...`)
- Model: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (19.64GB, 15 mtp.* tensors bf16, modelopt NVFP4)
- Launch flags: `--quantization modelopt --tensor-parallel-size 2 --max-model-len 262144 --max-num-seqs 4 --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`
- KV cache (TP=2): 1,089,600 tokens, max concurrency 15.48× at 256K

## Phase 2 (TP=1) Hard Gates: 4/4 PASS
- Boot: 196s (≤360s)
- Fibonacci coherence 5/5: PASS
- Tool call (Tokyo weather): PASS
- Multi-turn (Tokyo→Berlin→which warmer): PASS

## Phase 3 (TP=2) Hard Gates: 6/6 PASS
- Boot: 201s (≤360s)
- Fibonacci 5/5: PASS
- Tool call: PASS
- Multi-turn: PASS
- Reasoning gate (47×83=3901): PASS
- Long-context coherence at 137K tokens (true 131K+ test): PASS — needle found in 23.3s

## Phase 4 Benchmark Matrix — PARTIAL (3/11 cells)

Cells completed BEFORE GPU0 crash:

| cell | conc | ctx | mean tok/s | std | n |
|---|---|---|---|---|---|
| c1_ctx0 | 1 | 0 | 105.8 | 3.02 | 3 |
| c1_ctx32000 | 1 | 32000 | 99.2 | 3.95 | 3 |
| c1_ctx131072 | 1 | 131072 | 51.4 | 0.87 | 3 |

**Comparison vs FP8+MTP=3 baseline (from EXP-1 N=5):**

| cell | NVFP4+MTP=3 | FP8+MTP=3 (OLD) | Δ |
|---|---|---|---|
| c=1 ctx=0 | 105.8 | ~85 (extrapolated) | **+24%** |
| c=1 ctx=32k | 99.2 | not measured | — |
| c=1 ctx=131k | 51.4 | 82.6 (c=1 ctx=128k baseline) | **-38%** |

**Concerning: -38% at long context.** Single-stream short context wins +24%, but 131K loses big.

## Failure Mode

c=1 ctx=250k cell aborted (bench tool clamped 250000→244k, then engine errored "RuntimeError: cancelled" in shm_broadcast.dequeue). Container exited. GPU0 (the second RTX Pro 6000) hung in NV_ERR_GPU_IN_FULLCHIP_RESET state with `_deviceTeardown: Disable of Cuda limit activation failed`.

`nvidia-smi -i 0 -r` failed: GPU is gone from device list. `modprobe -r nvidia_uvm` failed: module not present (loaded as built-in or different name).

Host reboot required to recover GPU0.

## Hard Gates Summary

- Phases 1-3 (download, verify, smoke, TP=2 gates): **all 10 hard gates PASS**
- Phase 4 (benchmark): 3/11 cells completed, GPU crash blocked the rest
- Promote-decision blocked by missing data on c=2, c=4 cells and long-context confirmation
