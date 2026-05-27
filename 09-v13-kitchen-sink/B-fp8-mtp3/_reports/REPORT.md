# Experiment 9 — v13 image kitchen-sink: BF16+DFlash vs FP8+MTP=3, plus tool-eval-bench cross-provider

**Date**: 2026-05-27
**Hardware**: 2× NVIDIA RTX PRO 6000 Blackwell (96 GiB each), TP=2, GPU 0 reserved for display
**Image**: `repne/vllm:v13`
**Harnesses**: `llm-inference-bench` v0.4.8 (pinned) + upstream main (v0.4.23, 12 commits ahead)
**Total runs**: 300 matrix cells (150 per config × 2 configs) + 8 method-validation cells + 1 P2P + quality probes + Phase D cross-provider

---

## Abstract

This experiment performs an exhaustive kitchen-sink characterization of two production-relevant configurations on the freshly-released `repne/vllm:v13` image: **(A) BF16 + DFlash N=8** and **(B) FP8 + MTP=3**. Each config is swept across a 5×6 matrix (concurrency ∈ {1,2,4,8,16,32}, context ∈ {0, 16k, 32k, 64k, 131k}) with 5 repetitions for statistical power, validated against both the pinned v0.4.8 harness and the upstream main branch (12 commits ahead) for methodology integrity, and topped off with a **Phase D tool-eval-bench cross-provider showdown** comparing our local endpoints head-to-head with Anthropic Claude, OpenAI GPT-5, Google Gemini, and Cerebras Qwen-3-235B.

**Headline result**: TBD after Pass B + Phase D completion.

---

## 0. Why this experiment exists

Exp 06 + Exp 08 established Repne-fork FP8+MTP=3 as production SOTA at **2,083.7 tok/s** (c=32 ctx=0) but on the previous `repne/vllm:latest` image. The `v13` image was released with `--decode-warmup-seconds` support and several scheduler fixes. We needed to:

1. **Validate v13 doesn't regress** vs the prior image
2. **Run a higher-rep matrix** (N=5 instead of Exp 08's N=2) to tighten error bars
3. **Cross-validate the harness itself** by running every cell through both the pinned v0.4.8 (the reference) and upstream main
4. **Compare against the broader frontier** — not just upstream vLLM, but the actual frontier APIs (Claude, GPT-5, Gemini) on a deterministic tool-call task to ground our Qwen3.6-27B performance in the real-world LLM landscape

This is **Experiment 9** in the [`qwen36-27b-blackwell-inference-study`](https://github.com/jcartu/qwen36-27b-blackwell-inference-study/) monorepo.

---

## 1. Matrix design

### 1.1 Configurations

| | Pass A | Pass B |
|---|---|---|
| **Weights** | BF16 (`Qwen/Qwen3.6-27B`) | FP8 W8A8 (`Qwen/Qwen3.6-27B-FP8`) |
| **Speculative decoding** | DFlash N=8 | MTP N=3 |
| **GMU** | 0.88 | 0.88 |
| **TP** | 2 | 2 |
| **Max model len** | 262,144 | 262,144 |
| **Extra flags** | `--gumbel-draft-sampler --use-local-argmax-reduction` | (none) |

### 1.2 Phase structure (per config)

| Phase | Cells | Purpose | Wall-time |
|---|---:|---|---:|
| 0 — Preflight | — | Container reset, KV/CG verify, P2P sanity, display-GPU baseline | 5 min |
| 1 — Method validation | 8 | (c1, c8) × (ctx0, ctx32k) × {v0.4.8, upstream main} → harness cross-validation | 15 min |
| 2 — P2P intra-image stability | 1 | nccl-tests `all_reduce_perf` for TP=2 bandwidth baseline | 2 min |
| 3 — Matrix | 150 | (c ∈ {1,2,4,8,16,32}) × (ctx ∈ {0, 16k, 32k, 65k, 131k}) × N=5 reps | ~3 h |
| 4 — Quality | 2 JSONs | (LAVD long-context summarization) + (hotel-lights tool-call), seed-pinned | 5 min |
| 5 — CJK + display-GPU watchdog | 1 | end-of-pass CJK degenerate-output check + display GPU mem audit | 2 min |

**Total per pass**: ~3h 30min. **Both passes**: ~7h. With Phase D bolt-on (60 min cap): ~8h.

### 1.3 Display-GPU isolation

GPU 0 (`PCI 538bf008`) is the **physical display GPU**. Hardline rules enforced by `sweep_driver.sh`:
- vLLM container launched with `CUDA_VISIBLE_DEVICES=1,2` (skipping 0)
- Pre-cell watchdog: if `nvidia-smi --query-gpu=memory.used --id=0` returns >4000 MiB → abort sweep, alert user
- Tolerance: display GPU baseline is 3273-3400 MiB (Sway + Firefox tabs)
- **Total cells across whole sweep with zero violations**: target TBD on completion

---

## 2. Methodology

### 2.1 Per-cell command shape

```
benchy --base-url http://localhost:8000 \
       --model Qwen3.6-27B[-FP8] \
       --concurrency $C \
       --context-size $CTX \
       --duration 60 \
       --decode-warmup-seconds 20 \
       --enable-thinking \
       --seed $((42 + REP))
```

### 2.2 Idempotency

`run_cell.sh` skips any cell whose JSON exists *and* contains a valid numeric `aggregate_tps`. Lets us resume after any crash/OOM/restart with zero re-work.

### 2.3 Endpoint health gate

Before each cell:
1. `curl -fsS http://localhost:8000/v1/models` → must 200
2. `nvidia-smi --query-gpu=memory.used --id=0,1,2` → display GPU < 4 GiB
3. If endpoint absent: auto-relaunch via known-good launcher snapshot

### 2.4 A → B handoff

Pass A ends → driver:
1. Stops Pass A container cleanly
2. Rewrites `launch-qwen36-27b-tp2-sota.sh`: BF16 model→FP8 model, DFlash 5-line block → MTP 2-line block
3. Waits for endpoint at T+120s with KV ≥ 1,000,000 tokens
4. **No fallback flag set** → Pass B continues on v13. If endpoint never comes up → drops `_fallback.flag` and re-runs Pass B on v12 image (known-good for FP8).

---

## 3. Results — Pass A: BF16 + DFlash N=8

### 3.1 Method validation (phase 1)

[Pass A phase 1 cross-validation table — to be filled]

### 3.2 Matrix (phase 3, N=5)

[5×6 matrix mean ± std table — to be filled by analyze.py]

### 3.3 Quality probes (phase 4)

| Probe | Status | Notes |
|---|---|---|
| LAVD long-context summarization | TBD | 18 KB JSON, seed=42 |
| hotel-lights tool-call | TBD | 35 KB JSON, seed=42 |

### 3.4 CJK + display-GPU audit (phase 5)

| Check | Result |
|---|---|
| CJK degenerate output | TBD |
| Max display GPU mem during pass | TBD MiB (baseline 3273) |

---

## 4. Results — Pass B: FP8 + MTP=3

[Same structure as §3]

---

## 5. Cross-config synthesis (A vs B head-to-head, same image, same matrix)

[Side-by-side table: tok/s per cell, with Δ % and bias toward MTP for high-concurrency / DFlash for long-context]

---

## 6. Phase D — tool-eval-bench cross-provider

A bolt-on comparison using the open-source [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (69 deterministic tool-call scenarios, OpenAI-compatible wire format, pass/partial/fail scoring) to ground our Qwen3.6-27B performance in the broader LLM landscape.

### 6.1 Endpoint matrix

| ID | Provider | Model | Why |
|---|---|---|---|
| D1 | localhost vLLM | Qwen3.6-27B (FP8+MTP=3) | Our config under test |
| D3 | OpenRouter | anthropic/claude-sonnet-4.6 | Anthropic flagship |
| D4 | OpenRouter | anthropic/claude-haiku-4.5 | Anthropic cheap-tier |
| D5 | OpenRouter | openai/gpt-5.5 | OpenAI flagship |
| D6 | OpenRouter | openai/gpt-5-mini | OpenAI cheap-tier |
| D7 | OpenRouter | openai/gpt-5-nano | OpenAI ultra-cheap baseline |
| D8 | OpenRouter | google/gemini-3.5-flash | Google flash-tier |
| D9 | Cerebras | qwen-3-235b-a22b-instruct-2507 | Best-possible Qwen reference |

### 6.2 Protocol

- 15 scenarios via `--short` selector, seed 42, temperature 0.2
- Per-endpoint timeout 90s, max-turns 8 (tool-eval-bench defaults)
- Cost cap **$5**, wall-clock cap **60 minutes**

### 6.3 Leaderboard

[Auto-generated by `aggregate.py` → `leaderboard.csv`]

| Rank | Endpoint | Model | Score | Deployability | Median Turn (ms) |
|---|---|---|---|---|---|
| TBD | | | | | |

---

## 7. Reproducibility

Every artifact in this experiment is reproducible from the contents of this directory:

```bash
# Pull the image
docker pull repne/vllm:v13

# Launch endpoint (FP8+MTP=3 variant)
bash scripts/launch-fp8-mtp3.sh   # or launch-bf16-dflash.sh

# Run sweep
bash scripts/sweep_driver.sh

# Phase D
bash phase-D-tooleval/run_phase_d.sh

# Aggregate
python3 scripts/extract_csv.py exp09-results.csv {A,B}-*/phase3-matrix
python3 scripts/to_master_schema.py exp09-results.csv master-rows.csv
```

The 411-line `sweep_driver.sh` is a PPID=1-detached daemon that runs unattended for ~8 hours, auto-restarts the endpoint if it disappears, and emits per-cell JSON to a hierarchical directory tree. It is **idempotent** — re-running it after a crash skips completed cells.

---

## 8. Files

```
09-v13-kitchen-sink/
├── README.md                          ← this file
├── exp09-results.csv                  ← 300 cells, 35-col rich schema
├── exp09-master-rows.csv              ← 300 cells, 10-col master schema
├── PLAN.md                            ← original plan
├── A-bf16-dflash/
│   ├── phase1-method-validation/      ← 8 cross-validated cells
│   ├── phase3-matrix/run{1..5}/       ← 150 matrix cells
│   ├── phase4-quality/                ← LAVD + hotel-lights JSONs
│   └── phase5-cjk/                    ← end-of-pass audits
├── B-fp8-mtp3/                        ← same layout
├── phase-D-tooleval/
│   ├── run_phase_d.sh                 ← orchestrator
│   ├── aggregate.py                   ← leaderboard builder
│   ├── leaderboard.csv                ← final cross-provider scores
│   └── results/D{1,3,..9}-*/          ← per-endpoint outputs
├── scripts/
│   ├── sweep_driver.sh
│   ├── run_cell.sh
│   ├── extract_csv.py
│   ├── to_master_schema.py
│   ├── harness-v0.4.8/                ← pinned harness snapshot
│   └── harness-upstream/              ← upstream main snapshot
└── logs/
    └── driver.log                     ← event log, ~8 hr of activity
```

---

## 9. Acknowledgements

- The Repne fork team for the `v13` image
- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) for the cross-provider scoring framework
- AesSedai's `perplexity-sliding-window` branch (used in Exp 08, retained as reference)
- The PhaelonQuant Discord community for the quality concerns that motivated Exp 07 and shaped this experiment's quality-probe scope
