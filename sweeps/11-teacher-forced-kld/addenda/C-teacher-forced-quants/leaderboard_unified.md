# Unified KL/JSD Leaderboard

All comparisons are vs `bf16-ref-{multi,single}.safetensors` (Exp 11 reference).
Bits = nats / ln(2). Numbers in scientific notation; throughput from free-run (Addendum A).

## Multi-prompt (504 positions = 8 prompts × 63 tokens)

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

## Single-prompt (16 positions, 1 prompt × ~17 tokens)

| Variant | Mean KL A→B (bits) | Mean KL B→A (bits) | Mean JSD (bits) | Max KL (bits) | Max JSD (bits) |
|---|---:|---:|---:|---:|---:|
| `bf16-self` | 7.589e-04 | 7.612e-04 | 1.900e-04 | 2.937e-03 | 7.356e-04 |
| `fp8` | 5.523e-03 | 5.522e-03 | 1.379e-03 | 1.319e-02 | 3.271e-03 |
| `nvfp4` | 2.995e-01 | 1.475e-01 | 3.607e-02 | 3.926e+00 | 3.712e-01 |
| `C4-awq-4bit` | 4.086e-02 | 4.327e-02 | 1.022e-02 | 1.994e-01 | 5.250e-02 |
| `C5-awq-6bit` | 3.511e-02 | 3.522e-02 | 8.667e-03 | 9.282e-02 | 2.398e-02 |
| `C7-gptq-groxaxo` | 2.410e-01 | 2.488e-01 | 5.780e-02 | 6.177e-01 | 1.431e-01 |
| `C8-gptq-qwopus` | 7.008e-02 | 6.981e-02 | 1.704e-02 | 3.832e-01 | 8.720e-02 |

## Notes

- **`bf16-self`**: noise floor (same model, same kernels, two runs). Any divergence below this is indistinguishable from run-to-run nondeterminism.
- **Addendum B cells (B2/B3/B4)**: compared vs B1 (BF16-weights + auto-KV), not vs Exp 11's BF16 ref. These isolate KV-cache vs weight-quant contributions to drift.
- **Addendum C cells (C4–C8)**: compared vs Exp 11's BF16 ref, same protocol as FP8/NVFP4.
- **Max KL ≈ 0.69 bits** in JSD column = `ln(2)` → indicates at least one position where the two distributions are fully disjoint (top-1 swap with vanishing mass elsewhere). Common at quant boundaries.
- **Free-run tok/s** is from Addendum A (1024-token generation, 8 prompts). Teacher-forced collect timings are not throughput-representative (include prefill + warmup).
