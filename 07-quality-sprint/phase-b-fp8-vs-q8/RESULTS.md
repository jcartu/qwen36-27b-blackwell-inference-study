# Phase B — FP8 (W8A8) vs Q8_0 GGUF (closer to W8A16) quality probe

**Phaelon's claim:** "an 8-bit GGUF will ALWAYS be more accurate than a W8A8 because you get lobotomized lowering activations"

**Empirical test:** Run identical prompts on both engines at temp=0, seed=42, compare outputs. Then run a hard algorithmic test and verify functional correctness.

## Setup

| | FP8+MTP=3 (vLLM) | Q8_0 GGUF (llama.cpp) |
|---|---|---|
| Engine | repne/vllm:latest (`d0a200f7`) v0.1.dev16400 | llama-server 8467 |
| Weights | Qwen3.6-27B-FP8 W8A8 fp8 block-128 | unsloth/Qwen3.6-27B-GGUF Q8_0 |
| Activations | FP8 (block-quantized) | BF16 (compute path) |
| Spec decoding | MTP=3 with gumbel sampler | None (n/a in llama-server) |
| TP / split | TP=2 vLLM | --tensor-split 0.5,0.5 |
| Sampling | temp=0, top_p=1.0, seed=42 | identical |

## Results — 8 standard prompts

| prompt | FP8 chars | Q8 chars | identical? | both correct? |
|---|---|---|---|---|
| fib (Fibonacci 20) | 85 | 85 | ✅ | ✅ |
| code_python (palindrome) | 136 | 143 | ❌ | ✅ both |
| math (47×83=3901) | 381 | 297 | ❌ | ✅ both 3901 |
| haiku | 261 | 246 | ❌ | ✅ both valid |
| explain_diff (TCP/UDP) | 457 | 507 | ❌ | ✅ both factual |
| json_extract | 61 | 61 | ✅ | ✅ |
| code_review | 427 | 469 | ❌ | ✅ both valid |
| factual (Eiffel) | 66 | 66 | ✅ | ✅ |

**3/8 identical content**, 5/8 differ in phrasing/style. **All Q8 outputs are correct, all FP8 outputs are correct.** No quality regression detected on FP8.

## Hard algorithmic test — Manacher's algorithm

Both engines were given: *"Implement Python `longest_palindromic_substring(s) -> str` using Manacher's algorithm."*

| | result |
|---|---|
| FP8+MTP=3 | ✅ Correct, single-pass max tracking, idiomatic |
| Q8_0 GGUF | ✅ Correct, separate `max(P)` + `P.index(max_len)` (slightly less efficient) |

Both pass 8/8 functional tests including edge cases (empty string, single char, all-same-char, multi-palindrome). **No quality difference at hard-coding tasks.**

## Reasoning length analysis

**Q8 GGUF reasons 2.06× more on average** than FP8+MTP=3, with extreme cases:
- code_python: Q8 = 17,887 reasoning chars vs FP8 = 2,597 (**6.89× more**)
- factual: Q8 = 3,393 vs FP8 = 1,253 (2.71× more)

**This may reflect Q8's slightly less aggressive verbosity-suppression.** Either Q8 is genuinely "thinking harder" (potentially better quality on edge cases) or it's just less calibrated and meandering. Without a harder benchmark we can't tell.

## Speed

| Engine | wall time | total tokens | effective tok/s |
|---|---|---|---|
| FP8+MTP=3 | 52.1s | 7,812 | **149.9** |
| Q8 GGUF | 188.3s | 13,127 | 69.7 |

**FP8+MTP=3 is 2.15× faster on this prompt suite** (and would be more on equal-token-count comparisons since Q8 generated more reasoning tokens).

## Conclusion on Phaelon's claim

**Phaelon's "W8A8 lobotomization" doesn't manifest in our tests at this scale.** Both engines:
- produce factually correct outputs
- pass functional code tests with 100% correctness
- generate equivalent-quality prose

The differences are real but don't cross any quality threshold I can measure. Q8 reasons more (2x avg, 7x extreme), but that doesn't translate to better answers on the prompts I tested.

**For production: FP8+MTP=3 is 2.15× faster with no measurable quality penalty.** If we cared about marginal token-level distribution matching (e.g., for replay testing or distillation), Q8 might be preferable. For an interactive coding agent, FP8 wins.

## Caveats

- **Small N (8 prompts + 1 hard).** A real perplexity sweep on FineWeb or similar would be more decisive.
- **Subjective quality judgments.** "Both correct" is my eyeball assessment; a more rigorous test would use an LLM judge (e.g., GPT-4 grading both side by side).
- **GGUF is not "pure" W8A16.** Q8_0 stores weights as 8-bit + scale, dequantized to FP16/BF16 on-the-fly. The activations path is BF16, so it's W8A16 in spirit.
- We didn't test Aes Sedai's [sliding-window perplexity branch](https://github.com/AesSedai/llama.cpp/tree/perplexity-sliding-window) for KLD. That would give a single-number quality verdict.

## Worth doing next (deferred)

If quality becomes contested, run:
1. AesSedai's sliding-window KLD test on a 100K-token corpus
2. HumanEval / MBPP code benchmark on both
3. GSM8K math benchmark
4. LMSYS-style A/B ratings via GPT-4 judge

For now: **FP8+MTP=3 stays SOTA. Phaelon's concern is theoretically valid but not empirically dominant on this hardware/model combo.**
