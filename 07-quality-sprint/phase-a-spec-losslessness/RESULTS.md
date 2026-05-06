# Phase A — Spec-decode losslessness check

**Setup:** Identical prompts (8 diverse), temperature=0, top_p=1.0, seed=42 on:
- FP8+MTP=3 (current production)
- FP8 no-spec (same image, same model, just dropped `--speculative-config`)

**Both engines verified deterministic** at temp=0 (3x same prompt → identical output, seed-independent at greedy).

## Findings

| prompt | content match | mtp3 chars | nospec chars |
|---|---|---|---|
| fib | ✅ identical | 85 | 85 |
| code_python | ✅ identical | 136 | 136 |
| math | ❌ differs | 381 | 362 |
| haiku | ❌ differs | 261 | 264 |
| explain_diff | ❌ differs | 457 | 510 |
| json_extract | ✅ identical | 61 | 61 |
| code_review | ❌ differs | 427 | 451 |
| factual | ✅ identical | 66 | 66 |

**4/8 prompts produce different outputs at temp=0.** This means **MTP=3 is not lossless** in the Repne implementation.

## Are the differences harmful?

Looking at each diverging case:
- **math**: MTP=3 says "long multiplication" vs no-spec "multiplication". Both arrive at 3901. **Equivalent.**
- **haiku**: Different haikus. Both 5-7-5, both valid English, both about thunderstorms. **Equivalent quality, different content.**
- **explain_diff**: Same factual content, different sentence ordering. Both correct. **Equivalent.**
- **code_review**: Different point #3 (input validation vs duplicating built-in). Both valid critiques. **Equivalent quality.**

**No correctness regressions detected.** All MTP=3 outputs are factually accurate, code is syntactically valid, math is correct. The differences are stylistic/lexical, not quality.

## Speed

- MTP=3: 149.9 effective tok/s across 8 prompts
- No-spec: 86.0 effective tok/s
- **MTP=3 speedup: 1.74×** (real, sustained, on actual user-style prompts)

## Conclusion

**MTP=3 is empirically not bitwise-lossless** at temp=0 in this implementation. **But for our use case it doesn't matter:**
1. No correctness regressions on diverse prompts
2. 1.74× speedup is significant
3. The "non-lossless" behavior is at the level of synonymous-token substitutions, not factual errors

This is acceptable for production. **Keep FP8+MTP=3.** But it's worth knowing: if you ever need pure determinism (e.g., for replay-based testing), you'd want to disable spec decoding.

Phaelon's implicit concern about W8A8 lobotomization is independent of the MTP=3 question — the next step (Phase B) is to compare FP8 vs Q8 GGUF outputs on the same prompts.
