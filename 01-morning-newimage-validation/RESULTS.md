# EXP-1: BF16+DFlash=8 Variance Validation

**Cells:** 5 BF16+DFlash=8 cells × N=5 randomized runs, 30 s each, single-cell bench.

**Baseline:** OLD image (`f0e7574`, 2026-05-03) prior bench.

**Verdict threshold:** ±2σ. Δ within band = NOISE; > 2σ above old = WIN; > 2σ below old = REGRESSION.


## Per-cell summary

| cell | conc | ctx | N | mean tok/s | stddev | min | max | OLD | Δ vs OLD | verdict |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| c4_ctx0 | 4 | 0 | 5 | 332.6 | 6.73 | 324.2 | 343.0 | 352.8 | -5.7% | REGRESSION (Δ -5.7%, >2σ band 13.5) |
| c2_ctx244k | 2 | 249856 | 5 | 145.9 | 10.33 | 134.9 | 159.8 | 160.3 | -9.0% | NOISE (Δ -9.0% within ±2σ=20.7) |
| c2_ctx250k | 2 | 256000 | 5 | 134.1 | 3.31 | 130.9 | 137.9 | 160.3 | -16.3% | REGRESSION (Δ -16.3%, >2σ band 6.6) |
| c1_ctx128k | 1 | 131072 | 5 | 90.1 | 2.45 | 87.7 | 94.1 | 82.6 | +9.0% | WIN (Δ +9.0%, >2σ band 4.9) |
| c4_ctx128k | 4 | 131072 | 5 | 290.0 | 5.20 | 284.2 | 297.6 | 272.0 | +6.6% | WIN (Δ +6.6%, >2σ band 10.4) |

## TTFT (s) per cell — mean of N runs

| cell | TTFT mean | TTFT p50 | TTFT p90 | TTFT p99 |
|:---:|:---:|:---:|:---:|:---:|
| c4_ctx0 | 0.165 | 0.164 | 0.218 | 0.229 |
| c2_ctx244k | 1.654 | 1.515 | 2.029 | 2.186 |
| c2_ctx250k | 1.728 | 1.593 | 2.113 | 2.269 |
| c1_ctx128k | 0.752 | 0.752 | 0.761 | 0.763 |
| c4_ctx128k | 1.535 | 0.882 | 2.670 | 2.708 |

## Spec-decode acceptance rate per cell — mean of N runs

| cell | accept mean | std |
|:---:|:---:|:---:|
| c4_ctx0 | 22.9% | 1.93% |
| c2_ctx244k | 24.6% | 2.08% |
| c2_ctx250k | 21.6% | 5.47% |
| c1_ctx128k | 28.8% | 4.63% |
| c4_ctx128k | 25.0% | 1.47% |

## Per-run raw data


### c4_ctx0

| run | tps | ttft_avg | accept | cap_lim |
|:---:|:---:|:---:|:---:|:---:|
| c4_ctx0_run1 | 331.5 | 0.156 | 23.3% | False |
| c4_ctx0_run2 | 331.6 | 0.133 | 22.7% | False |
| c4_ctx0_run3 | 324.2 | 0.248 | 20.5% | False |
| c4_ctx0_run4 | 343.0 | 0.148 | 25.8% | False |
| c4_ctx0_run5 | 332.5 | 0.139 | 22.3% | False |

### c2_ctx244k

| run | tps | ttft_avg | accept | cap_lim |
|:---:|:---:|:---:|:---:|:---:|
| c2_ctx244k_run1 | 159.8 | 1.730 | 22.0% | True |
| c2_ctx244k_run2 | 139.8 | 1.684 | 25.0% | False |
| c2_ctx244k_run3 | 141.5 | 1.679 | 22.9% | False |
| c2_ctx244k_run4 | 134.9 | 1.576 | 26.9% | False |
| c2_ctx244k_run5 | 153.4 | 1.601 | 26.2% | False |

### c2_ctx250k

| run | tps | ttft_avg | accept | cap_lim |
|:---:|:---:|:---:|:---:|:---:|
| c2_ctx250k_run1 | 130.9 | 1.797 | 18.3% | False |
| c2_ctx250k_run2 | 137.9 | 1.710 | 17.8% | False |
| c2_ctx250k_run3 | 131.2 | 1.659 | 18.5% | False |
| c2_ctx250k_run4 | 133.2 | 1.663 | 22.4% | False |
| c2_ctx250k_run5 | 137.3 | 1.811 | 30.8% | False |

### c1_ctx128k

| run | tps | ttft_avg | accept | cap_lim |
|:---:|:---:|:---:|:---:|:---:|
| c1_ctx128k_run1 | 90.0 | 0.752 | 31.5% | False |
| c1_ctx128k_run2 | 87.7 | 0.747 | 30.2% | False |
| c1_ctx128k_run3 | 94.1 | 0.751 | 34.3% | False |
| c1_ctx128k_run4 | 89.9 | 0.742 | 23.8% | False |
| c1_ctx128k_run5 | 88.6 | 0.769 | 24.2% | False |

### c4_ctx128k

| run | tps | ttft_avg | accept | cap_lim |
|:---:|:---:|:---:|:---:|:---:|
| c4_ctx128k_run1 | 297.6 | 1.585 | 24.8% | False |
| c4_ctx128k_run2 | 286.5 | 1.425 | 23.8% | False |
| c4_ctx128k_run3 | 284.2 | 1.541 | 23.6% | False |
| c4_ctx128k_run4 | 292.1 | 1.572 | 26.0% | False |
| c4_ctx128k_run5 | 289.4 | 1.553 | 27.0% | False |

---

## ⛔ HALT TRIGGERED

c=4 ctx=0 confirmed REGRESSION at >2σ vs OLD image baseline.

EXP-2 will NOT proceed automatically. Manual review required.

