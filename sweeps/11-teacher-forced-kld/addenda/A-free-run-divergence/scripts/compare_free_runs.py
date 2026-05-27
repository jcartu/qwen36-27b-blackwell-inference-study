#!/usr/bin/env python3
"""
Addendum A: Compare free-run generation results.

Given two free_run_generate.py output JSONs (reference + candidate),
computes per-prompt and aggregate divergence metrics:

  - first_divergence_pos: token index where outputs first differ (-1 = identical)
  - token_agreement_rate_@{64,128,256,512}: fraction of tokens that match at each window
  - edit_distance_@{64,128,256,512}: Levenshtein on token-id sequences
  - exact_match_@{64,128,256,512}: 1 if sequences fully match up to that length

Usage:
  python3 compare_free_runs.py \
    --ref results/bf16-freerun.json \
    --cand results/fp8-freerun.json \
    --output results/fp8-vs-bf16-freerun.json
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="Reference JSON (bf16)")
    p.add_argument("--cand", required=True, help="Candidate JSON (fp8/nvfp4/etc)")
    p.add_argument("--output", required=True)
    return p.parse_args()


def levenshtein(a: list, b: list) -> int:
    """Token-level Levenshtein distance (bounded to len(a)+len(b) for speed)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j + 1] = min(
                prev[j + 1] + 1,   # deletion
                curr[j] + 1,       # insertion
                prev[j] + (0 if ca == cb else 1),  # substitution
            )
        prev = curr
    return prev[lb]


def first_divergence(a: list, b: list) -> int:
    """Return first index where a and b differ. -1 if identical up to min length."""
    for i, (ta, tb) in enumerate(zip(a, b)):
        if ta != tb:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1  # identical


def compare_at_window(a: list, b: list, window: int):
    a_w = a[:window]
    b_w = b[:window]
    n = min(len(a_w), len(b_w))
    if n == 0:
        return {"agreement": None, "edit_distance": None, "exact_match": None,
                "ref_len": len(a_w), "cand_len": len(b_w)}
    matches = sum(1 for x, y in zip(a_w, b_w) if x == y)
    ed = levenshtein(a_w, b_w)
    return {
        "agreement_rate": round(matches / max(len(a_w), len(b_w)), 6),
        "edit_distance": ed,
        "edit_distance_normalized": round(ed / max(len(a_w), len(b_w)), 6),
        "exact_match": int(a_w == b_w),
        "ref_len": len(a_w),
        "cand_len": len(b_w),
    }


def main():
    args = parse_args()
    ref_data = json.loads(Path(args.ref).read_text())
    cand_data = json.loads(Path(args.cand).read_text())

    ref_prompts = ref_data["prompts"]
    cand_prompts = cand_data["prompts"]
    assert len(ref_prompts) == len(cand_prompts), \
        f"Prompt count mismatch: {len(ref_prompts)} vs {len(cand_prompts)}"

    WINDOWS = [64, 128, 256, 512]
    per_prompt = []
    for rp, cp in zip(ref_prompts, cand_prompts):
        r_ids = rp["generated_token_ids"]
        c_ids = cp["generated_token_ids"]
        fd = first_divergence(r_ids, c_ids)
        windows = {f"@{w}": compare_at_window(r_ids, c_ids, w) for w in WINDOWS}
        per_prompt.append({
            "prompt_idx": rp["prompt_idx"],
            "first_divergence_pos": fd,
            "identical": fd == -1,
            **windows,
        })

    # Aggregate
    fds = [p["first_divergence_pos"] for p in per_prompt]
    fds_positive = [x for x in fds if x >= 0]

    agg = {
        "n_prompts": len(per_prompt),
        "n_identical": sum(1 for x in fds if x == -1),
        "n_divergent": len(fds_positive),
        "first_divergence_min": min(fds_positive) if fds_positive else None,
        "first_divergence_max": max(fds_positive) if fds_positive else None,
        "first_divergence_mean": round(sum(fds_positive) / len(fds_positive), 1) if fds_positive else None,
    }
    for w in WINDOWS:
        rates = [p[f"@{w}"]["agreement_rate"] for p in per_prompt
                 if p[f"@{w}"]["agreement_rate"] is not None]
        eds = [p[f"@{w}"]["edit_distance"] for p in per_prompt
               if p[f"@{w}"]["edit_distance"] is not None]
        exacts = [p[f"@{w}"]["exact_match"] for p in per_prompt
                  if p[f"@{w}"]["exact_match"] is not None]
        agg[f"@{w}"] = {
            "mean_agreement_rate": round(sum(rates) / len(rates), 6) if rates else None,
            "mean_edit_distance": round(sum(eds) / len(eds), 2) if eds else None,
            "exact_match_count": sum(exacts),
            "exact_match_rate": round(sum(exacts) / len(exacts), 4) if exacts else None,
        }

    result = {
        "ref_label": ref_data["label"],
        "cand_label": cand_data["label"],
        "ref_model": ref_data["model"],
        "cand_model": cand_data["model"],
        "max_tokens": ref_data["max_tokens"],
        "seed": ref_data["seed"],
        "aggregate": agg,
        "per_prompt": per_prompt,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    # Pretty summary to stdout
    print(f"\n=== Free-run divergence: {cand_data['label']} vs {ref_data['label']} ===")
    print(f"  prompts:          {agg['n_prompts']}")
    print(f"  identical:        {agg['n_identical']} / {agg['n_prompts']}")
    print(f"  divergent:        {agg['n_divergent']} / {agg['n_prompts']}")
    if fds_positive:
        print(f"  1st divergence:   min={agg['first_divergence_min']}  "
              f"mean={agg['first_divergence_mean']}  max={agg['first_divergence_max']}")
    for w in WINDOWS:
        a = agg[f"@{w}"]
        print(f"  @{w:>4} tokens:  agree={a['mean_agreement_rate']:.4f}  "
              f"edit_dist={a['mean_edit_distance']:.1f}  "
              f"exact={a['exact_match_count']}/{agg['n_prompts']}")
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
