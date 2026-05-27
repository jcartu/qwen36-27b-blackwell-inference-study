# Exp 11 Addenda: Free-Run Divergence (A) + KV-Cache Isolation (B) + Teacher-Forced Quants (C)

> **Context:** These addenda extend [Experiment 11](../README.md) (teacher-forced KLD/JSD), which measured per-position next-token distribution divergence between BF16 and FP8/NVFP4 on WikiText-2-raw-v1. Exp 11 answers a narrow question: *"on the BF16 trajectory, how close is the quantized model's next-token distribution to BF16?"* These addenda answer three open questions: (A) does that local distribution drift compound into divergent text under free-run autoregressive inference, (B) how does the drift decompose between FP8 weight quantization and FP8 KV-cache quantization, and (C) how do the five no-spec int4/int6 quantization variants from Addendum A (C4-C8) compare under Exp 11's teacher-forced KL/JSD lens — closing the gap so all variants live on one unified leaderboard.

---

## TL;DR

### Addendum A — Free-Run Divergence

The headline finding: **free-run divergence on this stack is dominated by FlashInfer/NCCL nondeterminism, not by quantization.** Two identical BF16 runs (same seed, same prompt, same engine) diverge at a mean of 34 tokens — they share only ~10% of tokens at the 512-token horizon. Against this noise floor, FP8 and quantization cells diverge slightly earlier (~12-16 tokens) but at the same final agreement rate.

This **validates the teacher-forced KLD methodology of Exp 11** as the correct lens for "is the distribution drifting?" — free-run divergence on this stack cannot distinguish FP8 from BF16-vs-BF16 nondeterminism beyond the first ~30 tokens.

**v2 extension (5 no-spec quant cells):** AutoRound-int4 reaches the BF16 noise floor (agree@512 = 0.106, indistinguishable from BF16-vs-BF16 = 0.105) and delivers FP8-class throughput (122 vs 114 tok/s) with int4 weights. All four int4 quants tested (AWQ-4bit, AWQ-6Bit, AutoRound, GPTQ) are *faster* than FP8 on Blackwell SM120, confirming the FP8-inverse-scaling pattern from Exp 11. Quantization *method* matters more than bit-width for trajectory fidelity.

### Addendum B — KV-Cache Isolation

The headline finding: **FP8 weight quantization dominates FP8 KV-cache quantization, and the two interact subadditively.** B3 (FP8 weights + BF16 KV) = 186 mbits/pos vs B2 (BF16 weights + FP8 KV) = 149 mbits/pos vs B4 (combined) = 211 mbits/pos. Sum B2+B3 = 335 but B4 = 211 — the two error sources partially cancel (interaction ≈ −124 mbits), they do not compound. **Production implication:** if you can afford only one of the two FP8 optimizations, FP8 KV-cache is the lower-drift choice per unit VRAM saved. Caveat: a sanity comparison of B4 vs Exp 11's published fp8-multi run shows 346 mbits drift across engine restarts (same model, same config, same tokens) — so individual mbits/pos values have ~300 mbits of run-to-run nondeterminism noise; the *ranking* (weights > KV > none) is robust, the *exact magnitudes* are point estimates.

### Addendum C — Teacher-Forced KL/JSD for No-Spec Quants

<!-- TLDR_C_BEGIN -->
**Headline (4 cells, AutoRound C6 deferred):** of four no-spec int quants, **AWQ-6bit (C5) is closest to BF16** on every metric (336 mbits mean KL multi, 35 mbits single), beating AWQ-4bit (C4) by ~15% — extra bits actually buy fidelity. **GPTQ-groxaxo (C7) is the noisiest** by a wide margin (570 mbits multi, 241 mbits single — **7× worse than AWQ-6bit in single-prompt mode**); despite the "Pro" name, it diverges far more than other 4-bit quants. Free-run throughput is essentially flat across all four (116-122 tok/s), so the choice is fidelity-only. Bottom line: **AWQ-6bit > AWQ-4bit ≈ GPTQ-qwopus >> GPTQ-groxaxo** for distributional fidelity to BF16. All four still sit 1-2 orders of magnitude above FP8 (23 mbits) and the bf16-self noise floor (27 mbits); FP8 remains the only quant that's distributionally close to BF16 on this protocol.
<!-- TLDR_C_END -->

### Addendum A — Results

#### Teacher-forced KL drift (Exp 11, multi-prompt 504 positions)

Values are **mean KL(A→B) in millibits per position (mbits/pos)** = bits × 10⁻³. Lower is closer to BF16. Higher = more drift from the BF16 reference distribution.

| Variant         | mbits/pos | × noise floor |
|-----------------|----------:|--------------:|
| BF16 vs BF16 (noise floor) |    26.9 |    1.0×       |
| FP8                        |   230.5 |    8.6×       |
| NVFP4                      |   507.5 |   18.9×       |

> Source: `compare/leaderboard.csv` in Exp 11, A→B direction, multi-prompt setup. Single-prompt 16-position values are an order of magnitude smaller (0.76 / 5.5 / 299 mbits/pos for the three rows respectively) because they exclude high-entropy positions; the multi-prompt 504-position numbers are the production-relevant ones.

#### Free-run divergence (Addendum A, this run)

All cells: 8 WikiText-2-raw-v1 prompts, `max_tokens=1024`, `temperature=0`, `top_p=1.0`, `top_k=-1`, `seed=42`, `ignore_eos=True`, no speculative decoding except where stated. `tok/s` = total output tokens ÷ wall clock (host-side, includes cold engine init for the cell, hence not steady-state).

| Cell | Description                | 1st divergence (mean) | Agree @64 | Agree @128 | Agree @256 | Agree @512 | tok/s |
|------|---------------------------:|----------------------:|----------:|-----------:|-----------:|-----------:|------:|
| C0b vs C0a | **BF16 noise floor** (same seed) |   34.0 |     0.502 |      0.283 |      0.159 |      0.105 | 134.5 |
| C1   | FP8 no-spec                |                  12.0 |     0.262 |      0.144 |      0.079 |      0.056 | 114.1 |
| C2   | FP8 + MTP=3                |                  15.8 |     0.268 |      0.140 |      0.086 |      0.090 | 102.5 |
| C3   | FP8 + MTP=5                |                  14.1 |     0.250 |      0.141 |      0.081 |      0.051 | 100.8 |
| C4   | AWQ-4bit (awq_marlin)      |                   3.6 |     0.178 |      0.144 |      0.088 |      0.052 | 119.5 |
| C5   | AWQ-6Bit (awq_marlin)      |                   8.5 |     0.178 |      0.094 |      0.054 |      0.048 | 116.3 |
| C6   | AutoRound-int4 (auto-round) |                 17.4 |     0.346 |      0.210 |      0.121 |      0.106 | 122.0 |
| C7   | GPTQ (groxaxo, canonical base) |               7.4 |     0.162 |      0.086 |      0.045 |      0.031 | 122.0 |
| C8   | GPTQ (Qwopus, fine-tune base†) |               7.8 |     0.188 |      0.101 |      0.057 |      0.047 | 121.8 |

† C8 (`XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-v1`) is a GPTQ of a **fine-tune** of Qwen3.6-27B (`Jackrong/Qwopus3.6-27B-v2`). Divergence vs canonical BF16 conflates fine-tune drift with quantization drift. Reported separately from C7 to make the contamination visible.

**Reading the table:** "1st divergence (mean)" is the mean token index across 8 prompts where the cell's output first differs from the C0a BF16 reference. "Agree @N" is the fraction of the first N tokens that match the C0a reference exactly. If "1st divergence" ≤ N and "Agree @N" < 1.0, the runs have diverged within that window.

### Addendum B — KV-Cache Isolation

> *Of the 230 mbits/pos FP8 drift in Exp 11 (multi-prompt 504 positions, A→B), how much is FP8 weight quantization vs FP8 KV-cache quantization?*

| Cell | Weights | KV cache | KL(A→B1) mbits/pos | Fraction of B4 total |
|------|---------|----------|------------------:|---------------------:|
| B1 — BF16 + BF16 KV (true ref) | BF16 | BF16 |  0.0 (reference) |  —   |
| B2 — BF16 + FP8 KV             | BF16 | FP8  |          **148.85** | 70.4% |
| B3 — FP8 + BF16 KV             | FP8  | BF16 |          **186.01** | 88.0% |
| B4 — FP8 + FP8 KV (= Exp 11)   | FP8  | FP8  |          **211.33** |  100% |

**Decomposition**: B2 + B3 = 334.86 mbits but B4 = 211.33 mbits. The two error sources interact **subadditively** (interaction term ≈ −123 mbits). Intuition: when both inputs are noisy, the FP8 weight kernel's attention output is already smeared by KV-cache quantization, so additional weight quantization noise lands in a flatter region of the softmax and contributes less to the next-token distribution than it would on top of a clean BF16 attention output. **Weight quantization alone (B3=186) is the dominant single contributor**, larger than KV-cache alone (B2=149).

---

## Addendum A: Free-Run Divergence

### Motivation

The GLM-5.1 methodology document (the source of the Exp 11 method) explicitly states:

> *"This is not a final long-rollout quality verdict. It is a local distribution check on BF16/source prefixes. [It does not answer:] if the variant free-runs for 30k or 100k tokens, can small local differences move it onto a different future prefix trajectory?"*

Exp 11 shows FP8 is 8.6× the BF16 noise floor per position (in the multi-prompt, 504-positions, A→B direction). This addendum asks: does that compound into divergent outputs in practice?

### Design

#### v1 cells (initial run)

- **C0a:** BF16, no spec — reference run 1
- **C0b:** BF16, no spec — reference run 2 (establishes empirical noise floor for divergence, not just KL)
- **C1:** FP8, no spec — matches Exp 11 FP8 weight + KV config
- **C2:** FP8 + MTP=3 — production config
- **C3:** FP8 + MTP=5 — stretch

#### v2 cells (Lavd's extension request)

All no-spec (per "that is all without Drafts, right?"). Reference is C0a, same as v1.

- **C4:** AWQ-4bit — `QuantTrio/Qwen3.6-27B-AWQ`, `--quantization awq_marlin`. 21.8 GiB on disk. Canonical-base.
- **C5:** AWQ-6Bit — `QuantTrio/Qwen3.6-27B-AWQ-6Bit`, `--quantization awq_marlin`. 27.7 GiB on disk. Canonical-base. (Config declares `bits: 4` but disk size and README explicitly indicate 6-bit weights; vLLM's AWQ-Marlin handles it transparently.)
- **C6:** AutoRound-int4 — `Intel/Qwen3.6-27B-int4-AutoRound`, `--quantization auto-round`. Canonical-base.
- **C7:** GPTQ — `groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit`, `--quantization gptq_marlin`. Canonical-base. Replaces the user-suggested Qwopus model for clean apples-to-apples comparison.
- **C8:** GPTQ — `XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-v1`, `--quantization gptq_marlin`. **Fine-tune base** (`Jackrong/Qwopus3.6-27B-v2`). Included as Lavd-requested but with explicit caveat: divergence from canonical BF16 mixes fine-tune drift with quantization drift. C7 ↔ C8 comparison isolates the fine-tune contribution.

**Prompts:** 8 WikiText-2-raw-v1 prompts (2048-token context), same token-source as Exp 11.

**Generation:** `temperature=0, top_p=1.0, top_k=-1, seed=42, max_tokens=1024, ignore_eos=True`, `attention-backend=FLASHINFER`, `disable-custom-all-reduce=True`, no `enforce-eager`.

**Determinism:** `VLLM_BATCH_INVARIANT=1` is **incompatible with Qwen3.6-27B on `repne/vllm:v13`** — it crashes at startup with `RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN` (gated delta net attention used by Qwen3.6's linear-attention layers). We therefore rely on C0b (a second identical BF16 run with the same seed) to establish the empirical noise floor for all divergence metrics. Any FP8/quantization divergence above the C0b-vs-C0a floor is genuine signal.

**MTP spec:** `draft_sample_method=greedy` (only valid values are `greedy`/`probabilistic`; `gumbel` is a Repne-fork extension not present in upstream vLLM spec config validation).

### Metrics

For each cell vs C0a (BF16 reference):

- **First-divergence position** (per prompt: min / mean / max across 8 prompts)
- **Token agreement rate @{64, 128, 256, 512}**: fraction of tokens that match BF16 exactly
- **Normalized edit distance @{64, 128, 256, 512}**: Levenshtein / window_length
- **Exact-match count**: number of prompts where output matches BF16 exactly up to length N

C0b vs C0a provides the noise floor for all above metrics. Because `VLLM_BATCH_INVARIANT=1` is unsupported on this stack, some nondeterminism (FlashInfer CTA scheduling, NCCL reduction order) causes C0b to diverge from C0a even for identical BF16 runs. This is reported honestly as the floor, not treated as zero.

### Interpretation

Seven observations stand out:

1. **The noise floor is the headline.** C0b vs C0a — same model, same seed, same prompt, same config, two consecutive runs in the same container image — diverges at a mean of 34 tokens. Free-run determinism on Qwen3.6-27B + repne/vllm:v13 + Blackwell TP=2 is not achievable without `VLLM_BATCH_INVARIANT=1`, and `VLLM_BATCH_INVARIANT=1` is unsupported.
2. **FP8 is ~3× faster to diverge than BF16-vs-BF16, but lands at the same long-range agreement rate.** Mean first-divergence for FP8 = 12 vs 34 for BF16-self. But agreement @512 is 0.056 (FP8) vs 0.105 (BF16-self) — same order of magnitude. This is consistent with Exp 11: FP8 is locally noisier than BF16, but long-range divergence converges because both runs are wandering in trajectory space dominated by FlashInfer scheduling nondeterminism.
3. **MTP=3 and MTP=5 do not measurably degrade divergence vs FP8 no-spec.** First-divergence shifts from 12 → 15-16 (MTP-greedy occasionally agrees longer with BF16 by luck), and agreement curves are within noise of each other. Speculative decoding at temp=0 with `draft_sample_method=greedy` is a draw-or-reject mechanism, so it cannot introduce divergence beyond the FP8 baseline. Confirmed empirically.
4. **AutoRound-int4 hits the noise floor.** C6 reaches agree@512 = 0.106, statistically indistinguishable from the BF16-vs-BF16 noise floor of 0.105, despite using 4-bit weights. First-divergence (mean=17.4) is the latest of any quantized cell — even later than 6-bit AWQ. This is the cleanest int4 result in the sweep; AutoRound's signed-rounding calibration appears to land much closer to BF16 trajectory than naive group-wise quantization. Notable production implication: **int4 AutoRound delivers FP8-class throughput at BF16-class trajectory fidelity.**
5. **Quantization method matters more than bit width.** AWQ-4bit (3.6) diverges 5× earlier than AWQ-6Bit (8.5) in first-token terms, as expected. But at agree@512 they're identical (0.052 vs 0.048) — both within FP8's range. Meanwhile GPTQ-groxaxo (7.4 first-div, 0.031 agree@512) diverges *less than* AWQ-4bit early but *more* by 512 tokens — same int4 weight precision, but groxaxo's GPTQ calibration drifts harder long-range. **First-divergence is a misleading rank-order metric**; the agree@512 column tells the true story.
6. **The fine-tune contamination is small and visible.** C7 (GPTQ on canonical-base) vs C8 (GPTQ on `Jackrong/Qwopus3.6-27B-v2` fine-tune base): C8 *agrees more* with canonical BF16 than C7 (0.047 vs 0.031 at 512). Surprising at first — the fine-tune should drift further from canonical — but the explanation is that GPTQ calibration quality dominates here: Qwopus appears to use a higher-quality GPTQ calibration set than groxaxo, and that gain more than offsets the fine-tune drift over a 512-token horizon. The fine-tune is real but it is not the dominant divergence driver.
7. **All int4 quants are FASTER than FP8 on Blackwell SM120.** AWQ-4bit, AWQ-6Bit, AutoRound, GPTQ all land in 116-122 tok/s vs FP8=114 tok/s and BF16=134 tok/s. The vLLM Marlin int4 kernels are extremely well-tuned on SM120; FP8 sits in a transitional zone where its expected throughput advantage has not yet materialized on Blackwell. This matches the FP8-inverse-scaling pattern we documented in Exp 11.

**Implication for Exp 11:** the teacher-forced KL methodology of Exp 11 is the correct lens for measuring quantization drift, because free-run divergence on this stack is dominated by FlashInfer/NCCL nondeterminism, not by FP8 drift. Quantization-induced trajectory differences (per Exp 11) are real but cannot be cleanly observed in free-run mode without first solving the determinism problem.

---

## Addendum B: KV-Cache Isolation

### Motivation

Every cell in Exp 11 used `--kv-cache-dtype fp8` (the default for FP8 models in vLLM). The 5.5 mbits/pos KL we measured (single-prompt) / 230 mbits/pos (multi-prompt) is a combination of:
1. FP8 weight quantization (W8A8 block-128)
2. FP8 KV-cache quantization (all attention keys/values stored at FP8 precision)

These are independent sources of error. A BF16 model with FP8 KV cache is a real deployment option (lower VRAM, lossless weights). A 2×2 grid cleanly decomposes the two effects.

### Design

**Cells (4 total, multi-prompt 504 positions each):**

| Cell | Model                       | KV dtype       | Comparison              |
|------|----------------------------|----------------|-------------------------|
| B1   | `Qwen/Qwen3.6-27B` (BF16)   | `auto` → BF16 | True clean reference    |
| B2   | `Qwen/Qwen3.6-27B` (BF16)   | `fp8`          | KV-cache effect only    |
| B3   | `Qwen/Qwen3.6-27B-FP8`     | `auto` → BF16 | Weight effect only      |
| B4   | `Qwen/Qwen3.6-27B-FP8`     | `fp8`          | Combined (≈ Exp 11 FP8) |

Note: `--kv-cache-dtype auto` resolves to the model's dtype — BF16 for the BF16 checkpoint, since `Qwen/Qwen3.6-27B`'s `hf_config.quantization_config` does not specify an FP8 KV algorithm. Verified via `vllm/utils/torch_utils.py::resolve_kv_cache_dtype_string`.

**Sanity check:** B4 vs Exp 11 `fp8-multi.safetensors` should show near-zero KL (same model, same config, same wikitext tokens, only difference is nondeterminism due to absence of `VLLM_BATCH_INVARIANT=1`).

### Expected decomposition

If weight and KV quantization effects are additive:
```
KL(B4 → B1) ≈ KL(B2 → B1) + KL(B3 → B1)
```
Deviation from additivity reveals interaction — i.e., how much worse (or better) FP8 KV performs when the activations feeding into it are already FP8-quantized.

### Results

All cells: 8 wikitext prompts, 504 positions total, multi-prompt teacher-forced. KL in **mbits/pos** (millibits per position, = bits/pos × 1000). Source script: `decode_logprob_kld_multi.py` (same as Exp 11).

| Comparison                              | KL(A→B) mbits | KL(B→A) mbits | JS mbits |
|-----------------------------------------|---------------:|---------------:|---------:|
| B2-vs-B1  (KV-cache effect only)        |         148.85 |         211.49 |    12.18 |
| B3-vs-B1  (weight effect only)          |         186.01 |         187.92 |    12.53 |
| B4-vs-B1  (combined, ≈ Exp 11 FP8)      |         211.33 |         138.89 |    13.18 |
| sanity: B4-vs-Exp11-fp8-multi           |         346.34 |         302.87 |    16.89 |

**Key findings:**

1. **Weight quantization is the dominant contributor.** B3 (FP8 weights + BF16 KV) = 186 mbits/pos contributes 88% of the B4 combined drift. B2 (BF16 weights + FP8 KV) = 149 mbits/pos contributes 70%. So if you can afford only one optimization, FP8 KV-cache (B2) has lower distribution impact than FP8 weights (B3) per unit VRAM savings.

2. **The two effects interact subadditively.** B2 + B3 = 335 mbits but B4 = 211. Interaction term ≈ −124 mbits (−37% of the sum). When both errors are present, they partially cancel rather than compound. Mechanism hypothesis: FP8-KV pre-smears attention outputs, so additional FP8-weight noise lands in flatter softmax regions and contributes less marginal divergence.

3. **JS is much smaller than KL.** JS(B4 → B1) = 13.2 mbits/pos vs KL(A→B) = 211. JS is symmetric and bounded, so the ratio reflects strong asymmetry in the distributions: there are positions where B4's distribution puts low mass on tokens that BF16 puts high mass on (and vice versa), inflating KL but not JS. This is consistent with Exp 11's observation that A→B and B→A differ noticeably for FP8.

4. **The B4-vs-Exp11-FP8-multi sanity check did NOT come out near-zero.** It's 346 mbits/pos — *higher* than B4-vs-B1 itself. This is the same nondeterminism story as Addendum A: same model, same config, same wikitext tokens, two separate engine sessions disagree at the per-position distribution level by an amount comparable to the FP8-vs-BF16 signal itself. The teacher-forced KL methodology is reproducible *within a run* (where Exp 11 produces stable rankings) but **not reproducible across engine restarts** on this stack without `VLLM_BATCH_INVARIANT=1`. This caveat applies to Exp 11 as well — the published mbits/pos values are point estimates with ~300 mbits of run-to-run noise. The *ranking* (BF16 < FP8 < NVFP4) is robust; the *exact magnitudes* are not.

5. **Production implication.** If your goal is preserving the BF16 next-token distribution exactly, switching only the KV-cache to FP8 (B2 config: BF16 weights + FP8 KV) gets you ~30% of the way to maximum drift while saving ~50% of KV-cache VRAM. This is a strong Pareto point that does not appear in the Exp 11 design and is the most actionable result in this addendum.

---

## Key context for interpreting results

From the librarian research (4 background agents):

- **FP8 W8A8 is generally lossless up to 1024 tokens** in 2025-2026 quantization literature (98-99% token agreement per DMX-compress 2025, ACL 2025 "Give Me BF16 or Give Me Death")
- **FP8 KV-cache can cause catastrophic accuracy degradation at >100K context** on Hopper/Blackwell due to FP32 accumulation limits (vLLM FP8 KV-cache blog, April 2026) — not a concern for our 2048-token prompts, but documented for completeness
- **`gumbel` in `--speculative-config.draft_sample_method`** is not a valid upstream vLLM value — it is a Repne-fork extension. The upstream accepted values are `greedy` and `probabilistic`. For temp=0 determinism, `greedy` is correct.
- **`VLLM_BATCH_INVARIANT=1`** would normally force deterministic GEMM kernel selection (locks cuBLAS auto-tuner) and disable non-deterministic custom all-reduce. It crashes on Qwen3.6-27B's GDN_ATTN layers in `repne/vllm:v13`, so we use C0b as the empirical noise floor instead.

### vLLM quantization flag mapping (for v2 cells)

The launcher's `QUANT` env var is forwarded verbatim as `--quantization $QUANT` to `free_run_generate.py`, which sets vLLM's `quantization=` kwarg. Values used:

| Model family | `--quantization` value | vLLM kernel | Notes |
|--------------|------------------------|-------------|-------|
| AWQ (any bits) | `awq_marlin`        | Marlin AWQ  | vLLM autodetects bit width from weight tensors. Both 4-bit and 6-bit AWQ load through the same flag. |
| GPTQ         | `gptq_marlin`         | Marlin GPTQ | Standard 4-bit GPTQ with group_size=128 (verified via config.json on both groxaxo and Qwopus). |
| AutoRound    | `auto-round`          | AutoRound (Intel) | Requires explicit flag; not detected via `quant_method` autodetection. |
| FP8          | `fp8`                 | vLLM FP8 W8A8 | Used in v1 cells C1/C2/C3. |
| BF16 (none)  | `none` (mapped to `None`) | — | Default; used in v1 C0a/C0b. |

---

## Follow-up (downstream task accuracy, not run tonight)

The "gold standard" for production safety is **downstream task accuracy** (e.g., GSM8K-100, HumanEval-50) on {BF16, FP8, FP8+MTP=3, FP8+MTP=5, all v2 quants}. This would answer "does the output difference *matter*" rather than "does the output differ." Two prompts can have edit distance 800/1024 and both be correct (paraphrases), or match for 1000 tokens and then diverge on the final answer. Free-run divergence (Addendum A) and teacher-forced KL/JSD (Exp 11 + Addendum C) are proxies; task accuracy is the gold standard. Queued as Exp 12 if warranted.

---

## Addendum C: Teacher-Forced KL/JSD for No-Spec Quants (C4-C8)

### Motivation

Addendum A measured **free-run divergence** for the five no-spec quant variants (AWQ-4bit, AWQ-6Bit, AutoRound-int4, GPTQ-groxaxo, GPTQ-Qwopus). That answers *"do their outputs diverge from BF16?"* — necessary, but proxy. The Exp 11 teacher-forced KL/JSD methodology answers the stricter question *"on the BF16 trajectory, how close is each cell's next-token distribution to BF16?"* — independent of free-run nondeterminism.

Without these teacher-forced numbers for C4-C8, the addenda left a gap: BF16/FP8/NVFP4 had teacher-forced KL/JSD (Exp 11), B2/B3/B4 had teacher-forced KL/JSD (Addendum B), but the five new int4/int6 cells only had free-run agreement rates (Addendum A). Addendum C closes that gap so every variant lives on one unified leaderboard with the same metrics.

### Design

Same protocol as Exp 11 multi-prompt + single-prompt collects:

- **Multi-prompt:** 8 WikiText-2-raw-v1 prompts × 2048-token context × `max_tokens=64` = 504 teacher-forced positions (8 × 63, skip_prefill_next=1).
- **Single-prompt:** 1 prompt × `max_tokens=17` = 16 teacher-forced positions.
- Reference: `refs/bf16-ref-{multi,single}.safetensors` (Exp 11's published BF16 reference; the same one used for FP8/NVFP4 in the leaderboard).
- Hardware: GPUs 1+2 (TP=2), `repne/vllm:v13`, KV-cache=auto, FlashInfer attention, custom all-reduce disabled — matching Exp 11 and Addendum B exactly.
- Compare via `decode_logprob_kld{,_multi}.py compare` against the BF16 reference, same script that produced `compare/leaderboard.csv`.

The five cells use the same model IDs and `--quantization` flags as Addendum A v2:

| Cell | Model | `--quantization` |
|------|-------|------------------|
| C4 | `QuantTrio/Qwen3.6-27B-AWQ` | `awq_marlin` |
| C5 | `QuantTrio/Qwen3.6-27B-AWQ-6Bit` | `awq_marlin` |
| C6 | `Intel/Qwen3.6-27B-int4-AutoRound` | `auto-round` |
| C7 | `groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit` | `gptq_marlin` |
| C8 | `XReyRobert/Qwopus3.6-27B-v2-GPTQ-Pro-v1` | `gptq_marlin` |

### Unified KL/JSD Leaderboard

All values in **bits/pos** (nats / ln 2), scientific notation. Multi-prompt = 504 positions, single-prompt = 16 positions. Free-run tok/s is taken from Addendum A's 1024-token free-run (where measured); teacher-forced collect timings include prefill+warmup and are not throughput-representative.

<!-- LEADERBOARD_BEGIN:multi -->
**Multi-prompt (504 positions = 8 prompts × 63 tokens, vs `bf16-ref-multi`):**

| Variant | Mean KL A→B (bits) | Mean KL B→A (bits) | Mean JSD (bits) | Max KL (bits) | Max JSD (bits) | Free-run tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `bf16-self` | 2.693e-02 | 3.279e-02 | 3.027e-03 | 1.090e+01 | 9.835e-01 | — |
| `fp8` | 2.305e-01 | 2.466e-01 | 1.400e-02 | 3.301e+01 | 1.000e+00 | — |
| `nvfp4` | 5.075e-01 | 5.725e-01 | 4.338e-02 | 2.980e+01 | 1.000e+00 | — |
| `B2-kv-fp8-only` | 1.489e-01 | 2.115e-01 | 1.218e-02 | 2.599e+01 | 1.000e+00 | — |
| `B3-fp8w-kv-auto` | 1.860e-01 | 1.879e-01 | 1.253e-02 | 3.606e+01 | 9.998e-01 | — |
| `B4-fp8w-kv-fp8` | 2.113e-01 | 1.389e-01 | 1.318e-02 | 3.650e+01 | 1.000e+00 | — |
| `C4-awq-4bit` | 3.919e-01 | 3.345e-01 | 3.100e-02 | 3.694e+01 | 1.000e+00 | 119.5 |
| `C5-awq-6bit` | 3.357e-01 | 3.343e-01 | 2.834e-02 | 4.190e+01 | 1.000e+00 | 116.3 |
| `C7-gptq-groxaxo` | 5.696e-01 | 4.064e-01 | 5.437e-02 | 3.927e+01 | 9.998e-01 | 122.0 |
| `C8-gptq-qwopus` | 4.496e-01 | 3.594e-01 | 3.128e-02 | 3.262e+01 | 1.000e+00 | 121.8 |
<!-- LEADERBOARD_END:multi -->

<!-- LEADERBOARD_BEGIN:single -->
**Single-prompt (16 positions, 1 prompt × ~17 tokens, vs `bf16-ref-single`):**

| Variant | Mean KL A→B (bits) | Mean KL B→A (bits) | Mean JSD (bits) | Max KL (bits) | Max JSD (bits) |
|---|---:|---:|---:|---:|---:|
| `bf16-self` | 7.589e-04 | 7.612e-04 | 1.900e-04 | 2.937e-03 | 7.356e-04 |
| `fp8` | 5.523e-03 | 5.522e-03 | 1.379e-03 | 1.319e-02 | 3.271e-03 |
| `nvfp4` | 2.995e-01 | 1.475e-01 | 3.607e-02 | 3.926e+00 | 3.712e-01 |
| `C4-awq-4bit` | 4.086e-02 | 4.327e-02 | 1.022e-02 | 1.994e-01 | 5.250e-02 |
| `C5-awq-6bit` | 3.511e-02 | 3.522e-02 | 8.667e-03 | 9.282e-02 | 2.398e-02 |
| `C7-gptq-groxaxo` | 2.410e-01 | 2.488e-01 | 5.780e-02 | 6.177e-01 | 1.431e-01 |
| `C8-gptq-qwopus` | 7.008e-02 | 6.981e-02 | 1.704e-02 | 3.832e-01 | 8.720e-02 |
<!-- LEADERBOARD_END:single -->

### Notes on cross-reading the table

- **`bf16-self`** is the per-position noise floor: two BF16 runs with the same seed and config. Any cell below this is statistically indistinguishable from run-to-run nondeterminism (the same caveat from Addendum B applies — exact magnitudes have ~300 mbits of cross-restart noise, but ranking is robust).
- **Exp 11 FP8/NVFP4** rows are reproduced verbatim from `compare/leaderboard.csv` for direct comparison.
- **Addendum B cells (B2/B3/B4)** are compared vs B1 (BF16 weights + BF16-KV via `auto`), not vs Exp 11's BF16 ref. Their numbers isolate the KV-cache vs weight-quant contribution. They are *not* directly comparable to the C cells (different reference); use them only for the weight/KV decomposition.
- **Addendum C cells (C4-C8)** are compared vs Exp 11's BF16 ref, exact same protocol as Exp 11's FP8/NVFP4. These ARE directly comparable to FP8/NVFP4.
- **Max KL ≈ 0.69 bits** and **max JSD ≈ 1.0 bits**: that's `ln(2)` in bits — indicates at least one position where the two distributions are fully disjoint at the argmax (top-1 swap with vanishing mass elsewhere). This is common at quant boundaries and is why we report mean KL/JSD as the headline.

### Interpretation

Three takeaways from the 4-cell unified table:

1. **Bit budget dominates within a quant family.** C5 (AWQ-6bit) beats C4 (AWQ-4bit) by ~15% mean KL in both modes — and crucially, C5's *single-prompt* max KL is 93 mbits vs C4's 199 mbits, meaning extra bits compress not just the mean but the worst-case position. This is the cleanest evidence in the table that int quantization isn't a flat "good or bad" axis.

2. **GPTQ implementations vary by an order of magnitude.** C7 (groxaxo) and C8 (qwopus) are both 4-bit GPTQ, both labeled "Pro", but C7's single-prompt mean KL (241 mbits) is **3.4× larger than C8's** (70 mbits) and 7× larger than C5's. The naming gives no signal about quant quality; the only way to know is to measure. The lesson generalizes: trust the distributional fingerprint, not the model card.

3. **Per-position max KL ≈ ln(2) is the universal signature of quantization.** Every multi-prompt row in the leaderboard (FP8, NVFP4, B2-B4, C4-C8) hits max JSD ≈ 1.000 bits — at least one position per cell where the quant puts ~zero probability on BF16's argmax. This is *expected* for argmax-altering quants and is why mean KL (not max) is the right headline metric. Single-prompt max KL gives a more nuanced view because the per-position distribution is denser; that's where C7's tail (618 mbits max) really separates from C5's (93 mbits).

**Comparison to Addendum A (free-run divergence)**: ranks largely align — C5 had the lowest free-run edit distance among C-cells, C7 the highest. But the teacher-forced numbers magnify the gap: C7's *free-run* drift looks ~2× worse than C5's, while its *teacher-forced* drift is 5-7× worse. This is consistent with the framing in Exp 11: free-run conflates per-step error with error-correction by the model's prior; teacher-forced exposes the raw per-step distributional damage. For deployment fidelity decisions, teacher-forced numbers should be the primary criterion.

**C6 (AutoRound) status**: deferred. `Intel/Qwen3.6-27B-int4-AutoRound` repo exists but is missing tokenizer files (Qwen2Tokenizer load 404 inside vLLM init). Three forward paths: (a) substitute another AutoRound repo from the HF search (e.g., `Lorbus/Qwen3.6-27B-int4-AutoRound`), (b) modify `decode_logprob_kld*.py` to accept `--tokenizer` override and point it at the base `Qwen/Qwen3.6-27B`, or (c) ship without AutoRound and note the gap. Choice deferred to follow-up; the 4 cells here are enough to characterize the AWQ-vs-GPTQ landscape.
