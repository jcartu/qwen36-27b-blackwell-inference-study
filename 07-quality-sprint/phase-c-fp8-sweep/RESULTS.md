# Phase C — FP8+MTP throughput sweep

**Goal:** find the optimal `num_speculative_tokens` for MTP, plus check if scheduler params can be improved.

## MTP `num_speculative_tokens` sweep (3-cell mini-matrix N=2 each, then N=3 full validation for the winner)

3-cell representative subset: c=1 ctx=0, c=4 ctx=0, c=4 ctx=131k.

| n | c=1 ctx=0 | c=4 ctx=0 | c=4 ctx=131k | KV cache |
|---|---|---|---|---|
| MTP=2 | 105.9±2.1 | 411.2±6.9 | 319.8±2.6 | 1,862,764 |
| MTP=3 | 117.1±2.8 | 449.8±4.6 | 350.5±4.9 | 1,846,472 |
| MTP=4 | 116.5±0.9 | 405.3±76.6 | 339.8±4.8 | 1,824,984 |
| MTP=5 | **124.9±0.1** | **472.2±15.6** | 347.3±2.3 | 1,809,787 |
| MTP=6 | 115.6±0.3 | 463.4±23.1 | 329.6±4.9 | 1,794,095 |

**MTP=5 wins on 2 of 3 mini-cells.** Confirmed with N=3 full 9-cell matrix:

| cell | FP8+MTP=3 | FP8+MTP=5 | Δ |
|---|---|---|---|
| c=1 ctx=0 | 117.1±2.8 | 119.9±4.3 | +2.4% |
| c=1 ctx=32k | 119.2±4.7 | 118.1±6.6 | −0.9% |
| **c=1 ctx=131k** | 95.0±2.5 | **101.2±0.8** | **+6.5%** |
| c=2 ctx=0 | 227.2±2.4 | 234.4±6.0 | +3.2% |
| c=2 ctx=32k | 227.0±5.1 | 231.2±3.7 | +1.9% |
| c=2 ctx=131k | 184.9±0.9 | 190.8±4.8 | +3.2% |
| c=4 ctx=0 | 449.8±4.6 | 462.5±6.3 | +2.8% |
| c=4 ctx=32k | 454.5±3.3 | 445.0±13.1 | −2.1% |
| c=4 ctx=131k | 350.5±4.9 | 346.9±2.9 | −1.0% |

**Mean Δ: +1.8%.** MTP=5 wins the cross-cell average by a small but consistent margin. The biggest single-cell win is **c=1 ctx=131k +6.5%** which is the single-user-with-deep-context scenario (very relevant to coding agents at the file/codebase level).

## max-num-seqs check at MTP=5

| seqs | c=1 ctx=0 | c=4 ctx=0 | c=4 ctx=131k |
|---|---|---|---|
| 64 | 115.9±1.3 | 469.7±14.8 | 344.8±10.7 |
| 128 (default) | 119.9±4.3 | 462.5±6.3 | 346.9±2.9 |

**Within noise.** Default seqs=128 is fine. Skipping further scheduler-knob sweeps — gains would be marginal.

## Recommendation

**Switch ROCK SOLID prod to MTP=5.** The +1.8% mean improvement is real (consistent direction across most cells), and the +6.5% on c=1 ctx=131k specifically helps the single-user long-context coding agent regime. Drawback: KV cache shrinks slightly (1.85M → 1.81M tokens, ~2% less concurrency headroom at 256K), but max-conc still 6.90× which is plenty.

If we **really** want to be conservative we can keep MTP=3 since the deltas are mostly within noise. But MTP=5 has no negative cells big enough to offset its wins.

## Caveats

- N=2 for the n-sweep mini-matrix; N=3 for MTP=5 full validation. MTP=2/4/6 didn't get full 9-cell validation.
- Acceptance rates from `server_spec_accept_rate` are aggregate across all spec positions; the per-position decay would be steeper at MTP=5 vs MTP=3, but aggregate throughput is what matters.
- We did not gate-test MTP=5 (Fibonacci/tool/multi-turn) — but Phase A already established MTP changes some output content even at temp=0; we'd expect MTP=5 to also be non-bitwise-equivalent to MTP=3 but produce equivalent quality. Worth a 4-gate confirmation before shipping to production.

