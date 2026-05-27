# Experiment 10 — Staged parser × image tournament (Qwen3.6-27B on Blackwell)

> **TL;DR.** A three-stage tournament settles three questions on the same 69-scenario
> `tool-eval-bench` suite: (1) which tool parser is better on the v13 + FP8 + MTP=3
> stack — `qwen3_xml` or `qwen3_coder`? (2) on BF16 + DFlash, which vLLM image
> ships more efficient inference — `repne/vllm:v13` or `vllm/vllm-openai:nightly`?
> (3) does that image winner generalize to FP8 + MTP=3? Seven frontier endpoints
> (Claude Sonnet 4.6, Claude Haiku 4.5, GPT-5.5, GPT-5-mini, GPT-5-nano,
> Gemini 3.5 Flash, Qwen-235B on Cerebras) are run on the same suite as
> yardsticks.

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
| `S1-v13-fp8-xml`   | <S1_XML_FINAL>   | <S1_XML_RESP>   | <S1_XML_MTM>   | <S1_XML_TCQ>   | <S1_XML_CMP>   |
| `S1-v13-fp8-coder` | <S1_CODER_FINAL> | <S1_CODER_RESP> | <S1_CODER_MTM> | <S1_CODER_TCQ> | <S1_CODER_CMP> |

**Winner P\***: `<P_STAR>`.

### Stage 2 — Image axis (BF16 + DFlash, parser=P\*)

| Cell | `final_score` | `responsiveness` | `median_turn_ms` |
|---|---:|---:|---:|
| `S2-v13-bf16-P*`     | <S2_V13_FINAL>     | <S2_V13_RESP>     | <S2_V13_MTM>     |
| `S2-nightly-bf16-P*` | <S2_NIGHTLY_FINAL> | <S2_NIGHTLY_RESP> | <S2_NIGHTLY_MTM> |

**BF16 image winner**: `<S2_WINNER>`.

### Stage 3 — FP8 image generalization (parser=P\*)

| Cell | `final_score` | `responsiveness` | `median_turn_ms` |
|---|---:|---:|---:|
| `S1-v13-fp8-P*` (reused)     | <S3_V13_FINAL>     | <S3_V13_RESP>     | <S3_V13_MTM>     |
| `S3-nightly-fp8-P*`          | <S3_NIGHTLY_FINAL> | <S3_NIGHTLY_RESP> | <S3_NIGHTLY_MTM> |

**FP8 image winner**: `<S3_WINNER>`.

### Frontier yardsticks (full 69-suite, same flags)

| Yardstick | `final_score` | `responsiveness` | `median_turn_ms` |
|---|---:|---:|---:|
| Y1 — claude-sonnet-4.6   | <Y1_FINAL> | <Y1_RESP> | <Y1_MTM> |
| Y2 — claude-haiku-4.5    | <Y2_FINAL> | <Y2_RESP> | <Y2_MTM> |
| Y3 — gpt-5.5             | <Y3_FINAL> | <Y3_RESP> | <Y3_MTM> |
| Y4 — gpt-5-mini          | <Y4_FINAL> | <Y4_RESP> | <Y4_MTM> |
| Y5 — gpt-5-nano          | <Y5_FINAL> | <Y5_RESP> | <Y5_MTM> |
| Y6 — gemini-3.5-flash    | <Y6_FINAL> | <Y6_RESP> | <Y6_MTM> |
| Y7 — qwen-235b (Cerebras)| <Y7_FINAL> | <Y7_RESP> | <Y7_MTM> |

## Verdict

- **Parser axis (Stage 1)**: <VERDICT_PARSER>
- **BF16 image axis (Stage 2)**: <VERDICT_BF16_IMAGE>
- **FP8 image axis (Stage 3)**: <VERDICT_FP8_IMAGE>

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
