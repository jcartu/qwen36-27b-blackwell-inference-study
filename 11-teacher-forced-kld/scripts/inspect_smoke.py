#!/usr/bin/env python3
"""Phase 0c R1 decision gate.

Loads the smoke-test safetensors and decides whether vLLM V1 returned real
model logprobs (R1 holds) or post-processor one-hot logprobs (R1 fails).

R1 holds if and only if, for every scored decode position:
  - the saved logprob distribution is NOT a delta on the forced token
  - the saved logprobs reflect a real model distribution (top-k has variety)

Hard pass criteria (all must be true for every position p in 0..max_tokens-1):
  1. The forced token's logprob is NOT exactly 0.0 (= log 1.0).
     If it were, the distribution would be the one-hot delta.
  2. At least 5 distinct tokens have logprob > -10 (= prob > 4.5e-5).
     A real softmax over 150k vocab tokens has many "live" tokens.
  3. The probability sum is approximately 1.0 (within numerical tolerance).
     Real distribution: sum(exp(logp)) ~= 1.0; one-hot: sum = 1.0 exactly.
     Together with criterion 2, this distinguishes the cases.
"""
import sys
from pathlib import Path
from safetensors.torch import load_file
import torch
import json

SMOKE_PATH = "/home/josh/qwen-vllm-test/sweeps/11-teacher-forced-kld/results/smoke/smoke-bf16.safetensors"
META_PATH = SMOKE_PATH + ".json"

print(f"=== Phase 0c R1 inspection: {SMOKE_PATH} ===\n")

if not Path(SMOKE_PATH).exists():
    print(f"FATAL: {SMOKE_PATH} does not exist. Smoke test did not produce output.")
    sys.exit(2)

# Load metadata for context
meta = json.loads(Path(META_PATH).read_text())
print(f"Label:                    {meta['label']}")
print(f"Model:                    {meta['model']}")
print(f"Prompt length:            {meta['prompt_len']}")
print(f"Max tokens:               {meta['max_tokens']}")
print(f"Num logprob positions:    {meta['num_logprob_positions']}")
print(f"Vocab size:               {meta['vocab_size']}")
print(f"Generated token IDs:      {meta['generated_token_ids']}")
print(f"Teacher-force token IDs:  {meta['teacher_force_token_ids']}")
print(f"Elapsed (sec):            {meta['elapsed_sec']:.2f}")
print()

# These should match if the processor forced the sampling correctly
gen_ids = meta['generated_token_ids']
forced_ids = meta['teacher_force_token_ids']
print(f"=== Forcing sanity check ===")
print(f"Generated matches forced? {gen_ids == forced_ids}")
if gen_ids != forced_ids:
    print(f"  Diff: generated={gen_ids}, forced={forced_ids}")
print()

# Load logprobs
tensors = load_file(SMOKE_PATH)
logprobs = tensors["logprobs"]
print(f"=== Logprobs tensor shape: {tuple(logprobs.shape)} (positions, vocab) ===\n")

n_positions = logprobs.shape[0]
all_pass = True
for pos in range(n_positions):
    lp = logprobs[pos]  # 1D tensor over vocab
    forced_token = forced_ids[pos] if pos < len(forced_ids) else None

    # Criterion 1: forced token logprob is NOT exactly 0.0
    forced_lp = float(lp[forced_token].item()) if forced_token is not None else None

    # Criterion 2: count "live" tokens (logprob > -10, i.e., prob > 4.5e-5)
    n_live = int((lp > -10).sum().item())

    # Criterion 3: probability sum
    probs = lp.exp()
    prob_sum = float(probs.sum().item())

    # Top-5 tokens by logprob (for visibility)
    top5_lp, top5_idx = lp.topk(5)
    top5_str = ", ".join(
        f"tok={int(i)} lp={float(l):.4f}"
        for l, i in zip(top5_lp.tolist(), top5_idx.tolist())
    )

    # Decision
    crit1 = forced_lp is not None and abs(forced_lp) > 1e-6  # NOT log(1.0) = 0.0
    crit2 = n_live >= 5
    crit3 = 0.95 < prob_sum < 1.05

    pos_pass = crit1 and crit2 and crit3
    if not pos_pass:
        all_pass = False

    print(f"--- Position {pos} ---")
    print(f"  Forced token:      {forced_token}")
    print(f"  Forced token logprob: {forced_lp:.6f}  (criterion 1: NOT 0.0 → {'PASS' if crit1 else 'FAIL'})")
    print(f"  Live tokens (lp>-10): {n_live}    (criterion 2: >= 5 → {'PASS' if crit2 else 'FAIL'})")
    print(f"  Probability sum:   {prob_sum:.6f}  (criterion 3: in [0.95, 1.05] → {'PASS' if crit3 else 'FAIL'})")
    print(f"  Top-5 logprobs:    {top5_str}")
    print(f"  → Position {pos}: {'PASS' if pos_pass else 'FAIL'}")
    print()

print("=" * 60)
if all_pass:
    print("R1 DECISION: PASS — vLLM V1 returns model logprobs BEFORE forcing.")
    print("Methodology is valid on repne/vllm:v13. Proceed to Phase 1.")
    sys.exit(0)
else:
    print("R1 DECISION: FAIL — saved logprobs reflect post-processor state.")
    print("Methodology BROKEN on repne/vllm:v13. Escalate before any further work.")
    sys.exit(1)
