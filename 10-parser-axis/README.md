# Experiment 10 — Staged parser × image tournament (Qwen3.6-27B on Blackwell)

> **TL;DR.** A three-stage tournament settles three questions on the same 69-scenario
> `tool-eval-bench` suite: (1) which tool parser is better on the v13 + FP8 + MTP=3
> stack — `qwen3_xml` or `qwen3_coder`? (2) on BF16 + DFlash, which vLLM image
> ships more efficient inference — `repne/vllm:v13` or `vllm/vllm-openai:nightly`?
> (3) does that image winner generalize to FP8 + MTP=3? Seven frontier endpoints
> (Claude Sonnet 4.6, Claude Haiku 4.5, GPT-5.5, GPT-5-mini, GPT-5-nano,
> Gemini 3.5 Flash, Qwen-235B on Cerebras) are run on the same suite as
> yardsticks.

## Methodology note (read first)

**This experiment uses the full 69-scenario `tool-eval-bench` suite.** Experiment 09
(`09-v13-kitchen-sink/`) used the 15-scenario `--short` subset, where Qwen3.6-27B on
v13 + FP8 + MTP=3 + `qwen3_xml` scored **93/100**. On the **full 69-scenario suite**,
the same configuration scores **62/100**.

The 62 and the 93 are **not comparable numbers**. They are different scoring
denominators measuring different scenario distributions. The 69-scenario suite
includes harder categories (E Error Recovery, I Multi-Turn Coherence, M Goal
Planning, N Cross-Tool Synthesis) that are underrepresented in the 15-scenario
`--short` subset. The full suite is the correct number for cross-model comparisons
with the frontier yardsticks reported below.

## Key findings

1. **Parser axis: tie.** `qwen3_xml` and `qwen3_coder` both score 62 on the full
   suite (responsiveness 80 each). Either parser is production-suitable. Pick
   `qwen3_xml` for clients that expect XML-tagged tool calls, `qwen3_coder` for
   JSON-wrapper clients.

2. **BF16 image axis: `repne/vllm:v13` wins uncontested.** The upstream
   `vllm/vllm-openai:nightly` image does **not** support DFlash speculative
   decoding — the `DFlashDraftModel` architecture is registered only in Repne's
   vLLM fork. Verified by introspecting the nightly image's `ModelRegistry`
   (only `LongcatFlashForCausalLM` and `MiMoV2FlashForCausalLM` are present). For
   BF16 + DFlash on Qwen3.6-27B, Repne v13 is the only viable image. The
   `S2-nightly-bf16-qwen3_coder/SKIP_REASON.md` artifact documents the evidence.

3. **FP8 image axis: `repne/vllm:v13` wins by +14 points.** Same model
   (Qwen3.6-27B-FP8), same speculative method (MTP=3), same parser (`qwen3_coder`),
   same hardware: v13 scores 62, nightly scores 48. The Repne fork's
   tool-calling pipeline is materially better on this stack — not a noise gap.

4. **Frontier positioning.** Local Qwen3.6-27B on v13 scores 62/100. The frontier
   models score 72–86. Qwen3.6-27B is roughly **27% behind frontier quality** but
   its `responsiveness=80` puts it **faster than every API except Cerebras Qwen-235B**
   (`responsiveness=96`). Median turn 1.15 s for the local 27B model vs 1.99 s
   (Gemini Flash) to 7.17 s (gpt-5-mini).
## Hardware

3× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120). GPU 0 is reserved
for display. All vLLM cells run TP=2 on GPUs 1 and 2.

## Methodology

| | |
|---|---|
| Bench | `tool-eval-bench` (full 69-scenario suite) |
| Bench flags | `--no-warmup --no-live --no-probe-engine --skip-coherence --timeout 120 --seed 42 --temperature 0.0 --trials 1 --json` |
| Local backend | OpenAI-compatible (`/v1/chat/completions`) |
| Frontier backend | OpenAI-compatible adapter, routed via OpenRouter and Cerebras |
| Repro | `bash scripts/run_exp10.sh` (orchestrator) |

### Stage 1 — Parser axis

Both cells run on the **Exp 09 v13 + FP8 + MTP=3** stack. The only difference
is the vLLM `--tool-call-parser`.

| Cell | Image | Quant | Spec | Tool parser |
|---|---|---|---|---|
| `S1-v13-fp8-xml` | `repne/vllm:v13` | FP8 | MTP / N=3 | `qwen3_xml` |
| `S1-v13-fp8-coder` | `repne/vllm:v13` | FP8 | MTP / N=3 | `qwen3_coder` |

Winner (by `final_score`, tiebreak `responsiveness` then `median_turn_ms`) is
referred to as **P\***.

### Stage 2 — Image axis (BF16 + DFlash)

Both cells use the **Stage 1 winning parser P\***.

| Cell | Image | Quant | Spec | Tool parser |
|---|---|---|---|---|
| `S2-v13-bf16-P*` | `repne/vllm:v13` | BF16 | DFlash / N=8 | P\* |
| `S2-nightly-bf16-P*` | `vllm/vllm-openai:nightly` | BF16 | DFlash / N=8 | P\* |

Winner by **efficiency** (`responsiveness` first, then lower `median_turn_ms`,
then `final_score`).

### Stage 3 — FP8 image generalization

Does the Stage 2 image preference hold under FP8?

| Cell | Image | Quant | Spec | Tool parser |
|---|---|---|---|---|
| `S1-v13-fp8-P*` (reused from Stage 1) | `repne/vllm:v13` | FP8 | MTP / N=3 | P\* |
| `S3-nightly-fp8-P*` | `vllm/vllm-openai:nightly` | FP8 | MTP / N=3 | P\* |

Winner by the same efficiency rule as Stage 2.

### Frontier yardsticks

All seven run the same 69-scenario suite with identical flags.

| Yardstick | Provider | Model id |
|---|---|---|
| Y1 | OpenRouter | `anthropic/claude-sonnet-4.6` |
| Y2 | OpenRouter | `anthropic/claude-haiku-4.5` |
| Y3 | OpenRouter | `openai/gpt-5.5` |
| Y4 | OpenRouter | `openai/gpt-5-mini` |
| Y5 | OpenRouter | `openai/gpt-5-nano` |
| Y6 | OpenRouter | `google/gemini-3.5-flash` |
| Y7 | Cerebras | `qwen-3-235b-a22b-instruct-2507` |

## Results

### Stage 1 — Parser axis

| Cell | `final_score` | `responsiveness` | `median_turn_ms` | `tool_call_quality` | `compliance` |
|---|---:|---:|---:|---:|---:|
| `S1-v13-fp8-xml`   | 62.0   | 80.000   | 1192   | —   | —   |
| `S1-v13-fp8-coder` | 62.0 | 80.000 | 1168 | — | — |

**Winner P\***: ``qwen3_coder``.

### Stage 2 — Image axis (BF16 + DFlash, parser=P\*)

| Cell | `final_score` | `responsiveness` | `median_turn_ms` |
|---|---:|---:|---:|
| `S2-v13-bf16-P*`     | 62.0     | 81.000     | 1146     |
| `S2-nightly-bf16-P*` | **not runnable** | **not runnable** | **not runnable** |

> `vllm/vllm-openai:nightly` (build `0.21.1rc1.dev323+g1fc2cee50`) does not register the `DFlashDraftModel` architecture used by `z-lab/Qwen3.6-27B-DFlash`. DFlash is a Repne-fork-exclusive speculative decoding method. See `results/stage2/S2-nightly-bf16-qwen3_coder/SKIP_REASON.md` for the registry introspection.

**BF16 image winner**: ``repne/vllm:v13``.

### Stage 3 — FP8 image generalization (parser=P\*)

| Cell | `final_score` | `responsiveness` | `median_turn_ms` |
|---|---:|---:|---:|
| `S1-v13-fp8-P*` (reused)     | 62.0     | 80.000     | 1168     |
| `S3-nightly-fp8-P*`          | 48.0 | 70.000 | 1685 |

**FP8 image winner**: ``repne/vllm:v13``.

### Frontier yardsticks (full 69-suite, same flags)

| Yardstick | `final_score` | `responsiveness` | `median_turn_ms` |
|---|---:|---:|---:|
| Y1 — claude-sonnet-4.6   | 86.0 | 47.000 | 3248 |
| Y2 — claude-haiku-4.5    | 82.0 | 43.000 | 3617 |
| Y3 — gpt-5.5             | 83.0 | 56.000 | 2587 |
| Y4 — gpt-5-mini          | 72.0 | 21.000 | 7172 |
| Y5 — gpt-5-nano          | **timeout** | **timeout** | **timeout** |
| Y6 — gemini-3.5-flash    | 86.0 | 65.000 | 1986 |
| Y7 — qwen-235b (Cerebras)| 81.0 | 96.000 | 378 |

> Y5 gpt-5-nano hit the 30-minute orchestrator timeout at scenario 55/69. The model averages 60–100 s per turn on hard categories (E/I/M/N), exceeding the per-cell budget. The other six yardsticks completed in 70 s (Cerebras) to 1249 s (gpt-5-mini).

## Verdict

- **Parser axis (Stage 1)**: Tie within noise. `qwen3_xml` = 62, `qwen3_coder` = 62. Picked `qwen3_coder` as P\* on tiebreak (responsiveness, then median_turn_ms).
- **BF16 image axis (Stage 2)**: `repne/vllm:v13` wins **uncontested**. Upstream `vllm/vllm-openai:nightly` does not support DFlash (see `results/stage2/S2-nightly-bf16-qwen3_coder/SKIP_REASON.md`).
- **FP8 image axis (Stage 3)**: `repne/vllm:v13` wins on both quality and efficiency (final_score +14, responsiveness +10, median turn 1168 ms vs 1685 ms). Confirms the BF16 image-axis result — the Repne fork's tool-calling pipeline is materially better, not just DFlash-only.
- **Frontier (yardsticks)**: 6/7 yardsticks completed. **Y5 gpt-5-nano timed out** at scenario 55/69 (30-minute orchestrator cap; the model's reasoning loops on harder categories take 60–100 s per turn). Top frontier scores: claude-sonnet-4.6 = 86, gemini-3.5-flash = 86, gpt-5.5 = 83, claude-haiku-4.5 = 82, qwen-235b on Cerebras = 81, gpt-5-mini = 72.

## Reproducing

```bash
cd sweeps/10-parser-axis
bash scripts/run_exp10.sh     # ~115 min wall-clock
python3 scripts/build_leaderboard.py
python3 scripts/render_exp10_readme.py EXP10_README_TEMPLATE.md README.md
```

## Layout

```
10-parser-axis/
  configs/matrix.tsv                    # cell definitions
  scripts/
    launch_cell.sh                      # per-cell container launcher
    run_exp10.sh                        # orchestrator (3 stages + frontier)
    build_leaderboard.py                # results/*/teb-results.json → leaderboard.csv
    render_exp10_readme.py              # template + JSONs → README.md
  results/
    stage1/{S1-v13-fp8-xml,S1-v13-fp8-coder}/{teb-results.json,console.log}
    stage1/winner.txt                   # P*
    stage2/{S2-v13-bf16-P*,S2-nightly-bf16-P*}/{teb-results.json,console.log}
    stage2/winner.txt                   # image winner
    stage3/S3-nightly-fp8-P*/{teb-results.json,console.log}
    stage3/winner.txt                   # FP8 image winner
    frontier/Y{1..7}-*/{teb-results.json,console.log}
  logs/
    exp10-driver.log                    # orchestrator event log
    frontier.log                        # background frontier log
  leaderboard.csv                       # unified output of build_leaderboard.py
  README.md                             # rendered final report
```
