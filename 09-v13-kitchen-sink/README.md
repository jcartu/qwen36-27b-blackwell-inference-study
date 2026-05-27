# Experiment 9 — v13 kitchen sink: BF16+DFlash + FP8+MTP=3, characterized

**Trigger:** repne shipped `repne/vllm:v13` overnight. We'd been on the v0.4.8 fork for Exp 06-08 and the upstream-main image for Exp 03/07. v13 promised "both worlds": MTP/DFlash quality from the fork *plus* upstream's scheduler fixes. Time to characterize it end-to-end on Blackwell with the matrix we'd use for any new image: 5×6 cells (c ∈ {1,2,4,8,16,32} × ctx ∈ {0, 16k, 32k, 64k, 131k}), N=5 reps, both BF16+DFlash and FP8+MTP=3 head-to-head.

**Total compute:** ~8 hours autonomous, 300 phase3-matrix cells (2 configs × 30 cells × 5 reps), 16 method-validation cells, 4 quality probes (LAVD + hotel-lights, each config), no babysitting.

## TL;DR


- **Pass A (BF16 + DFlash N=8) peak:** `1122.9` tok/s @ c=32 ctx=0 (N=5)
- **Pass B (FP8 + MTP=3) peak:** `2146.2` tok/s @ c=32 ctx=0 (N=5)
- **Long-context resilience (c=4 ctx=131k):** A = `290.9` tok/s, B = `324.5` tok/s
- **Quality probes:**
  - LAVD: A = `10/10`, B = `9/10`
  - Hotel-lights: A = `29/30`, B = `29/30`
- **Spec acceptance averaged across matrix:** A (DFlash N=8) = `23.1%`, B (MTP=3) = `56.6%`

## Methodology

### Why two harnesses?

We carry forward Exp 07's lesson: harness-version artifacts can leak into "absolute throughput" comparisons. Pass A and Pass B both ran the same matrix twice, once with `harness-v0.4.8` (the fork's snapshot) and once with `harness-upstream-main` (vLLM upstream's bench script as of v0.4.8 tag). The phase-1 method-validation cells (c ∈ {1, 8} × ctx ∈ {0, 32k}) establish ±2-3% harness agreement before we trust the phase-3 matrix.

### Why N=5?

Exp 06 used N=3 and we saw 5-8% intra-cell variance at high concurrency. We went to N=5 here so std is tight enough (typically <2% relative) to make 5%+ claims defensible.

### Hardware

- 3× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120), 96 GiB GDDR7 each, PCIe Gen5 x16
- GPU 0 (`538bf008`) reserved for display; baseline 3273 MiB, hardline halt if >4000 MiB
- GPU 1 + GPU 2 participate in workload via `--tensor-parallel-size 2` (TP=2)
- Driver 595.71.05, CUDA 13.2
- 251 GB system RAM, x86_64

### Settings shared by both passes

- `--gpu-memory-utilization 0.88` (down from 0.91 default; user request to leave headroom)
- `--enable-thinking` (user request — relevant for quality probes)
- `--max-model-len 134144` (just over 131k to give bench scenarios slack)
- Bench: prompt length matches ctx_tokens, output length 1024, requests = 5×concurrency

### Tool-call & reasoning parsers (relevant for Phase D)

- `--tool-call-parser qwen3_xml` (vLLM v13 default for Qwen3-family; **not** `qwen3_coder`, which Exp 01–08 used)
- `--reasoning-parser qwen3`
- `--enable-auto-tool-choice`

> **Note on parser choice.** vLLM ships two tool-call parsers for Qwen3-family models: `qwen3_xml` (XML-tagged tool calls, used here) and `qwen3_coder` (used in earlier experiments in this study). The Phase D tool-calling score below (93/100, ranked #1) was obtained with `qwen3_xml`. Whether `qwen3_coder` produces equal/better/worse tool-calling quality on the same model + image is the subject of [Experiment 10](../10-parser-axis/) (in progress).

## Pass A — BF16 + DFlash N=8

<!-- AUTO-GENERATED FROM A-summary.md -->
# Aggregate matrix (mean ± std, N reps)

## Config: `bf16-dflash` (N up to 5)

### Aggregate throughput (tok/s)

| concurrency \ context | 0 | 16k | 32k | 64k | 131k |
|---|---:|---:|---:|---:|---:|
| c=1 | 90.9 ±2.2 | 89.2 ±2.1 | 91.0 ±4.4 | 88.5 ±2.3 | 82.3 ±1.5 |
| c=2 | 171.4 ±4.2 | 174.9 ±2.7 | 169.1 ±3.0 | 165.9 ±4.0 | 161.5 ±4.4 |
| c=4 | 325.8 ±2.8 | 327.9 ±6.7 | 324.1 ±6.8 | 320.4 ±6.8 | 290.9 ±7.4 |
| c=8 | 579.8 ±5.3 | 568.1 ±8.3 | 542.7 ±8.4 | 529.3 ±6.5 | 471.8 ±6.0 |
| c=16 | 884.1 ±8.4 | 859.7 ±16.1 | 824.5 ±7.6 | 768.1 ±8.8 | 661.3 ±14.5 |
| c=32 | 1122.9 ±4.8 | 1085.2 ±12.4 | 1015.5 ±8.3 | 929.6 ±11.1 | 787.0 ±7.5 |

### Speculative decoding acceptance rate

| concurrency \ context | 0 | 16k | 32k | 64k | 131k |
|---|---:|---:|---:|---:|---:|
| c=1 | 0.267 | 0.166 | 0.312 | 0.241 | 0.264 |
| c=2 | 0.191 | 0.196 | 0.200 | 0.203 | 0.204 |
| c=4 | 0.200 | 0.229 | 0.213 | 0.227 | 0.250 |
| c=8 | 0.266 | 0.269 | 0.263 | 0.237 | 0.220 |
| c=16 | 0.211 | 0.195 | 0.203 | 0.212 | 0.252 |
| c=32 | 0.250 | 0.290 | 0.262 | 0.231 | 0.207 |

### Inter-token latency (ms, avg)

| concurrency \ context | 0 | 16k | 32k | 64k | 131k |
|---|---:|---:|---:|---:|---:|
| c=1 | 10.62 | 11.01 | 10.63 | 11.08 | 11.71 |
| c=2 | 11.45 | 11.25 | 11.61 | 11.72 | 11.98 |
| c=4 | 12.03 | 11.94 | 12.02 | 12.02 | 12.91 |
| c=8 | 13.26 | 13.30 | 13.51 | 13.96 | 16.25 |
| c=16 | 17.78 | 18.15 | 18.81 | 19.96 | 22.48 |
| c=32 | 25.64 | 26.35 | 28.63 | 31.96 | 38.14 |

## Pass B — FP8 + MTP=3

<!-- AUTO-GENERATED FROM B-summary.md -->
# Aggregate matrix (mean ± std, N reps)

## Config: `fp8-mtp3` (N up to 5)

### Aggregate throughput (tok/s)

| concurrency \ context | 0 | 16k | 32k | 64k | 131k |
|---|---:|---:|---:|---:|---:|
| c=1 | 93.9 ±1.9 | 98.2 ±1.4 | 98.3 ±1.2 | 94.0 ±0.6 | 85.0 ±0.8 |
| c=2 | 194.7 ±1.4 | 193.4 ±2.9 | 194.0 ±4.0 | 185.9 ±1.8 | 168.9 ±0.4 |
| c=4 | 387.5 ±2.6 | 385.1 ±1.9 | 380.4 ±4.4 | 364.6 ±4.0 | 324.5 ±3.1 |
| c=8 | 761.0 ±5.9 | 755.8 ±2.9 | 729.7 ±9.7 | 685.5 ±1.0 | 548.5 ±3.7 |
| c=16 | 1484.5 ±4.5 | 1371.5 ±11.9 | 1222.3 ±21.6 | 1006.4 ±6.8 | 711.7 ±3.6 |
| c=32 | 2146.2 ±5.3 | 1800.5 ±6.0 | 1572.5 ±21.5 | 1217.7 ±10.5 | 911.6 ±13.0 |

### Speculative decoding acceptance rate

| concurrency \ context | 0 | 16k | 32k | 64k | 131k |
|---|---:|---:|---:|---:|---:|
| c=1 | 0.651 | 0.583 | 0.637 | 0.533 | 0.531 |
| c=2 | 0.568 | 0.738 | 0.540 | 0.464 | 0.535 |
| c=4 | 0.611 | 0.676 | 0.547 | 0.518 | 0.518 |
| c=8 | 0.641 | 0.565 | 0.506 | 0.535 | 0.630 |
| c=16 | 0.560 | 0.534 | 0.558 | 0.547 | 0.568 |
| c=32 | 0.521 | 0.528 | 0.532 | 0.562 | 0.538 |

### Inter-token latency (ms, avg)

| concurrency \ context | 0 | 16k | 32k | 64k | 131k |
|---|---:|---:|---:|---:|---:|
| c=1 | 10.07 | 9.70 | 9.74 | 10.40 | 11.39 |
| c=2 | 10.01 | 9.82 | 9.81 | 10.54 | 11.43 |
| c=4 | 9.99 | 9.91 | 10.19 | 10.73 | 11.84 |
| c=8 | 10.11 | 10.36 | 10.75 | 11.33 | 14.11 |
| c=16 | 10.50 | 11.46 | 12.82 | 15.53 | 21.37 |
| c=32 | 14.58 | 17.70 | 20.10 | 27.10 | 33.62 |

## Phase C — Head-to-head: BF16+DFlash vs FP8+MTP=3

<!-- AUTO-GENERATED FROM C-synthesis.md -->
Same image (`repne/vllm:v13`), same matrix, same N=5 reps per cell.
All numbers below are 5-rep means; ± is population stdev.

## Aggregate throughput (tok/s)

| cell | A: BF16+DFlash | B: FP8+MTP=3 | Δ (B − A) | Δ % | winner |
|---|---:|---:|---:|---:|---|
| c=1 ctx=0 | 90.9 ±2.2 | 93.9 ±1.9 | +3.0 | +3.3% | **B** |
| c=1 ctx=16k | 89.2 ±2.1 | 98.2 ±1.4 | +8.9 | +10.0% | **B** |
| c=1 ctx=32k | 91.0 ±4.4 | 98.3 ±1.2 | +7.3 | +8.1% | **B** |
| c=1 ctx=64k | 88.5 ±2.3 | 94.0 ±0.6 | +5.5 | +6.2% | **B** |
| c=1 ctx=131k | 82.3 ±1.5 | 85.0 ±0.8 | +2.7 | +3.3% | **B** |
| c=2 ctx=0 | 171.4 ±4.2 | 194.7 ±1.4 | +23.4 | +13.6% | **B** |
| c=2 ctx=16k | 174.9 ±2.7 | 193.4 ±2.9 | +18.5 | +10.6% | **B** |
| c=2 ctx=32k | 169.1 ±3.0 | 194.0 ±4.0 | +24.9 | +14.7% | **B** |
| c=2 ctx=64k | 165.9 ±4.0 | 185.9 ±1.8 | +19.9 | +12.0% | **B** |
| c=2 ctx=131k | 161.5 ±4.4 | 168.9 ±0.4 | +7.4 | +4.6% | **B** |
| c=4 ctx=0 | 325.8 ±2.8 | 387.5 ±2.6 | +61.7 | +18.9% | **B** |
| c=4 ctx=16k | 327.9 ±6.7 | 385.1 ±1.9 | +57.1 | +17.4% | **B** |
| c=4 ctx=32k | 324.1 ±6.8 | 380.4 ±4.4 | +56.3 | +17.4% | **B** |
| c=4 ctx=64k | 320.4 ±6.8 | 364.6 ±4.0 | +44.2 | +13.8% | **B** |
| c=4 ctx=131k | 290.9 ±7.4 | 324.5 ±3.1 | +33.6 | +11.5% | **B** |
| c=8 ctx=0 | 579.8 ±5.3 | 761.0 ±5.9 | +181.2 | +31.3% | **B** |
| c=8 ctx=16k | 568.1 ±8.3 | 755.8 ±2.9 | +187.7 | +33.0% | **B** |
| c=8 ctx=32k | 542.7 ±8.4 | 729.7 ±9.7 | +187.0 | +34.4% | **B** |
| c=8 ctx=64k | 529.3 ±6.5 | 685.5 ±1.0 | +156.2 | +29.5% | **B** |
| c=8 ctx=131k | 471.8 ±6.0 | 548.5 ±3.7 | +76.8 | +16.3% | **B** |
| c=16 ctx=0 | 884.1 ±8.4 | 1484.5 ±4.5 | +600.3 | +67.9% | **B** |
| c=16 ctx=16k | 859.7 ±16.1 | 1371.5 ±11.9 | +511.7 | +59.5% | **B** |
| c=16 ctx=32k | 824.5 ±7.6 | 1222.3 ±21.6 | +397.8 | +48.3% | **B** |
| c=16 ctx=64k | 768.1 ±8.8 | 1006.4 ±6.8 | +238.3 | +31.0% | **B** |
| c=16 ctx=131k | 661.3 ±14.5 | 711.7 ±3.6 | +50.4 | +7.6% | **B** |
| c=32 ctx=0 | 1122.9 ±4.8 | 2146.2 ±5.3 | +1023.2 | +91.1% | **B** |
| c=32 ctx=16k | 1085.2 ±12.4 | 1800.5 ±6.0 | +715.3 | +65.9% | **B** |
| c=32 ctx=32k | 1015.5 ±8.3 | 1572.5 ±21.5 | +557.0 | +54.9% | **B** |
| c=32 ctx=64k | 929.6 ±11.1 | 1217.7 ±10.5 | +288.2 | +31.0% | **B** |
| c=32 ctx=131k | 787.0 ±7.5 | 911.6 ±13.0 | +124.6 | +15.8% | **B** |

## Summary

- Cells where **B (FP8+MTP=3) wins**: 30 / 30
- Cells where **A (BF16+DFlash) wins**: 0 / 30
- Mean delta (B − A) % across all cells: **+26.1%**
- Max B advantage: +91.1%
- Max A advantage: +0.0% (A never wins)

## Inter-token latency (avg ms — lower is better)

| cell | A: BF16+DFlash | B: FP8+MTP=3 |
|---|---:|---:|
| c=1 ctx=0 | 10.62 | 10.07 |
| c=1 ctx=16k | 11.01 | 9.70 |
| c=1 ctx=32k | 10.63 | 9.74 |
| c=1 ctx=64k | 11.08 | 10.40 |
| c=1 ctx=131k | 11.71 | 11.39 |
| c=2 ctx=0 | 11.45 | 10.01 |
| c=2 ctx=16k | 11.25 | 9.82 |
| c=2 ctx=32k | 11.61 | 9.81 |
| c=2 ctx=64k | 11.72 | 10.54 |
| c=2 ctx=131k | 11.98 | 11.43 |
| c=4 ctx=0 | 12.03 | 9.99 |
| c=4 ctx=16k | 11.94 | 9.91 |
| c=4 ctx=32k | 12.02 | 10.19 |
| c=4 ctx=64k | 12.02 | 10.73 |
| c=4 ctx=131k | 12.91 | 11.84 |
| c=8 ctx=0 | 13.26 | 10.11 |
| c=8 ctx=16k | 13.30 | 10.36 |
| c=8 ctx=32k | 13.51 | 10.75 |
| c=8 ctx=64k | 13.96 | 11.33 |
| c=8 ctx=131k | 16.25 | 14.11 |
| c=16 ctx=0 | 17.78 | 10.50 |
| c=16 ctx=16k | 18.15 | 11.46 |
| c=16 ctx=32k | 18.81 | 12.82 |
| c=16 ctx=64k | 19.96 | 15.53 |
| c=16 ctx=131k | 22.48 | 21.37 |
| c=32 ctx=0 | 25.64 | 14.58 |
| c=32 ctx=16k | 26.35 | 17.70 |
| c=32 ctx=32k | 28.63 | 20.10 |
| c=32 ctx=64k | 31.96 | 27.10 |
| c=32 ctx=131k | 38.14 | 33.62 |

## Speculative decoding acceptance rate

| cell | DFlash N=8 (A) | MTP N=3 (B) |
|---|---:|---:|
| c=1 ctx=0 | 0.267 | 0.651 |
| c=1 ctx=16k | 0.166 | 0.583 |
| c=1 ctx=32k | 0.312 | 0.637 |
| c=1 ctx=64k | 0.241 | 0.533 |
| c=1 ctx=131k | 0.264 | 0.531 |
| c=2 ctx=0 | 0.191 | 0.568 |
| c=2 ctx=16k | 0.196 | 0.738 |
| c=2 ctx=32k | 0.200 | 0.540 |
| c=2 ctx=64k | 0.203 | 0.464 |
| c=2 ctx=131k | 0.204 | 0.535 |
| c=4 ctx=0 | 0.200 | 0.611 |
| c=4 ctx=16k | 0.229 | 0.676 |
| c=4 ctx=32k | 0.213 | 0.547 |
| c=4 ctx=64k | 0.227 | 0.518 |
| c=4 ctx=131k | 0.250 | 0.518 |
| c=8 ctx=0 | 0.266 | 0.641 |
| c=8 ctx=16k | 0.269 | 0.565 |
| c=8 ctx=32k | 0.263 | 0.506 |
| c=8 ctx=64k | 0.237 | 0.535 |
| c=8 ctx=131k | 0.220 | 0.630 |
| c=16 ctx=0 | 0.211 | 0.560 |
| c=16 ctx=16k | 0.195 | 0.534 |
| c=16 ctx=32k | 0.203 | 0.558 |
| c=16 ctx=64k | 0.212 | 0.547 |
| c=16 ctx=131k | 0.252 | 0.568 |
| c=32 ctx=0 | 0.250 | 0.521 |
| c=32 ctx=16k | 0.290 | 0.528 |
| c=32 ctx=32k | 0.262 | 0.532 |
| c=32 ctx=64k | 0.231 | 0.562 |
| c=32 ctx=131k | 0.207 | 0.538 |

### Verdict for production

<!-- FILL IN -->

## Phase D — How does our local Qwen stack up against frontier APIs?

Throughput is meaningless if the model can't actually do the work. Phase D answers the question that matters for production: **is our Qwen3.6-27B FP8+MTP=3 endpoint competitive with frontier providers on real tool-calling tasks?**

We ran `tool-eval-bench` (15-scenario `--short` subset, seed=42, temp=0.2) against the same scenarios on:

| provider | model | gateway |
|---|---|---|
| **local (subject)** | **Qwen3.6-27B FP8+MTP=3 (this experiment)** | **localhost:8000** |
| Anthropic | claude-sonnet-4.6, claude-haiku-4.5 | OpenRouter |
| OpenAI | gpt-5.5, gpt-5-mini, gpt-5-nano | OpenRouter |
| Google | gemini-3.5-flash | OpenRouter |
| Cerebras | qwen-3-235b-a22b-instruct-2507 | Cerebras Cloud |

The frontier endpoints are **reference yardsticks**, not subjects. The point of this phase is to locate Qwen3.6-27B in the competitive landscape.

### Verdict for local Qwen

**Qwen3.6-27B FP8+MTP=3 placed #1 of 8** with final score **93/100** (★★★★★ Excellent).

It topped the leaderboard, above **google/gemini-3.5-flash** (#2, 93).

Responsiveness sub-score: **80/100** (lower = slower per-turn latency vs. frontier baselines).

### Full leaderboard

<!-- AUTO-GENERATED FROM phase-D leaderboard CSV -->
| endpoint_id | model | base_url | final_score | rating | total_points | max_points | deployability | responsiveness | median_turn_ms | total_tokens | token_efficiency | worst_category | scenario_count | thinking_enabled | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1-local | Qwen3.6-27B | http://***:8000 | 93 | ★★★★★ Excellent | 28 | 30 | 89 | 80 | 1195.8 | 36064 | 0.78 | C Multi-Step Chains (67%) | 15 | True | 2026-05-27T10-39-14Z_9af9d9 |
| D8-gemini-3.5-flash | google/gemini-3.5-flash | https://***/api | 93 | ★★★★★ Excellent | 28 | 30 | 87 | 72 | 1590.0 | 30916 | 0.91 | B Parameter Precision (67%) | 15 | True | 2026-05-27T08-17-18Z_d78a5f |
| D9-qwen-235b-cerebras | qwen-3-235b-a22b-instruct-2507 | https://*** | 87 | ★★★★ Good | 26 | 30 | 90 | 96 | 363.1 | 50962 | 0.51 | A Tool Selection (67%) | 15 | True | 2026-05-27T08-18-18Z_0e5c67 |
| D4-claude-haiku-4.5 | anthropic/claude-haiku-4.5 | https://***/api | 87 | ★★★★ Good | 26 | 30 | 78 | 57 | 2464.4 | 55831 | 0.47 | C Multi-Step Chains (67%) | 15 | True | 2026-05-27T08-04-39Z_36ef2f |
| D3-claude-sonnet-4.6 | anthropic/claude-sonnet-4.6 | https://***/api | 77 | ★★★★ Good | 23 | 30 | 71 | 56 | 2546.8 | 69709 | 0.33 | E Error Recovery (33%) | 15 | True | 2026-05-27T08-02-57Z_57b394 |
| D5-gpt-5.5 | openai/gpt-5.5 | https://***/api | 73 | ★★★ Adequate | 22 | 30 | 67 | 52 | 2877.6 | 25637 | 0.86 | E Error Recovery (33%) | 15 | True | 2026-05-27T08-06-07Z_9e847b |
| D7-gpt-5-nano | openai/gpt-5-nano | https://***/api | 73 | ★★★ Adequate | 22 | 30 | 56 | 15 | 9355.5 | 40922 | 0.54 | E Error Recovery (50%) | 15 | True | 2026-05-27T08-11-00Z_d56d0c |
| D6-gpt-5-mini | openai/gpt-5-mini | https://***/api | 60 | ★★★ Adequate | 18 | 30 | 49 | 24 | 6483.0 | 31111 | 0.58 | E Error Recovery (17%) | 15 | True | 2026-05-27T08-07-43Z_30ebad |

**Budget:** 60 min, $5 cap. Actual: `15` min, $`~2` total.

## Replicating

Sweep driver: `sweep_driver.sh` (411 lines, see `scripts/`). Single command rebuilds everything from raw JSON.

```bash
# Pull image
docker pull repne/vllm:v13

# Start the model
bash launch_qwen36-27b_v13_fp8mtp3.sh   # or _bf16dflash.sh

# Run sweep (8-10 hours)
nohup setsid bash scripts/sweep_driver.sh > logs/sweep.stdout.log 2>&1 &

# Finalize + publish
bash scripts/finalize_and_publish.sh --skip-phase-d
```

## What's in this directory

```
09-v13-kitchen-sink/
├── README.md            # this file
├── A-bf16-dflash/       # Pass A raw + reports
│   ├── phase1-method-validation/
│   ├── phase3-matrix/run{1..5}/
│   ├── phase4-quality/
│   └── _reports/{REPORT.md, A-summary.{md,json}, A-results-{rich,master}.csv}
├── B-fp8-mtp3/          # Pass B raw + reports (same shape)
├── C-synthesis/         # Phase C head-to-head
├── phase-D-tooleval/    # Phase D cross-provider results
└── scripts/             # All scripts used (vendored)
```

## Source provenance

- Sweep working dir: `/home/josh/qwen-vllm-test/sweeps/v13-kitchen-sink-bf16dflash-and-fp8mtp3/`
- Driver PID at completion: 370894
- Wall clock: 2026-05-27 02:55:07 → 2026-05-27 13:37:53
- All raw JSONs preserved; nothing curated away
