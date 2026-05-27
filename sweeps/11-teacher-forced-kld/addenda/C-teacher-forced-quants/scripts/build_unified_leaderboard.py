#!/usr/bin/env python3
"""Build the unified KL/JSD leaderboard merging:
  - Exp 11 cells: BF16-self / FP8 / FP8+MTP3 / NVFP4   (compare/*.json)
  - Addendum B:   B2 / B3 / B4                          (addenda/B-.../results/cmp-*.json)
  - Addendum C:   C4 / C5 / C6 / C7 / C8                (addenda/C-.../compare/*.json)

For each (variant, mode) we record:
    n_positions
    kl_a_to_b_mean_bits   (mean KL forward)
    kl_b_to_a_mean_bits   (mean KL reverse)
    js_mean_bits          (mean JSD)
    kl_max_bits           (max KL forward)
    js_max_bits           (max JSD)
    free_run_tok_per_s    (Addendum A free-run throughput, where available)

Bits = nats / ln(2).

Outputs:
    addenda/C-teacher-forced-quants/leaderboard_unified.csv
    addenda/C-teacher-forced-quants/leaderboard_unified.md
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # 11-teacher-forced-kld/
ADD_C = ROOT / "addenda" / "C-teacher-forced-quants"
ADD_B = ROOT / "addenda" / "B-kv-cache-isolation"
ADD_A = ROOT / "addenda" / "A-free-run-divergence"

NATS_TO_BITS = 1.0 / math.log(2)


def n2b(x):
    if x is None:
        return None
    return float(x) * NATS_TO_BITS


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def row_from_cmp(variant: str, mode: str, cmp_path: Path, tok_per_s=None) -> dict | None:
    d = load(cmp_path)
    if d is None:
        return None
    return {
        "variant": variant,
        "mode": mode,
        "n_positions": d.get("num_positions") or d.get("n_positions"),
        "kl_a_to_b_mean_bits": n2b(d.get("kl_a_to_b_mean")),
        "kl_b_to_a_mean_bits": n2b(d.get("kl_b_to_a_mean")),
        "js_mean_bits": n2b(d.get("js_mean")),
        "kl_max_bits": n2b(d.get("kl_a_to_b_max") if d.get("kl_a_to_b_max") is not None else (max(d["kl_a_to_b_per_pos"]) if d.get("kl_a_to_b_per_pos") else None)),
        "js_max_bits": n2b(d.get("js_max") if d.get("js_max") is not None else (max(d["js_per_pos"]) if d.get("js_per_pos") else None)),
        "free_run_tok_per_s": tok_per_s,
        "source": str(cmp_path.relative_to(ROOT)),
    }


def free_run_tok_per_s(label: str) -> float | None:
    """Look up free-run tok/s from Addendum A results JSONs by label prefix."""
    candidates = list((ADD_A / "results").glob(f"{label}*.json"))
    # Filter: skip cmp/sanity JSONs, take the canonical collect JSON
    for c in candidates:
        name = c.name
        if name.startswith("cmp-") or name.startswith("sanity-"):
            continue
        # match prefix exactly (e.g. C6-autoround-int4 matches C6-autoround-int4.json)
        if c.stem == label:
            d = load(c)
            if d:
                return d.get("tok_per_s")
    return None


def main():
    rows: list[dict] = []

    # === Exp 11 cells (read from compare/*.json) ===
    for variant in ("fp8", "fp8-mtp3", "nvfp4", "bf16-self"):
        for mode in ("multi", "single"):
            if variant == "bf16-self":
                cmp_path = ROOT / "compare" / f"bf16-self-{mode}.json"
            else:
                cmp_path = ROOT / "compare" / f"{variant}-vs-bf16-{mode}.json"
            row = row_from_cmp(variant, mode, cmp_path)
            if row is not None:
                rows.append(row)

    # === Addendum B cells (multi only; KV cache effects only show in multi) ===
    # cmp-{B1,B2,B3,B4}-vs-B1 not all exist — we have B2/B3/B4 vs B1.
    # B1 itself is the BF16 baseline for B; it can stand in as a noise-floor row.
    for variant, fname in [
        ("B2-kv-fp8-only",     "cmp-B2-vs-B1-kv-effect.json"),
        ("B3-fp8w-kv-auto",    "cmp-B3-vs-B1-weight-effect.json"),
        ("B4-fp8w-kv-fp8",     "cmp-B4-vs-B1-combined.json"),
    ]:
        cmp_path = ADD_B / "results" / fname
        row = row_from_cmp(variant, "multi", cmp_path)
        if row is not None:
            rows.append(row)

    # === Addendum C cells (multi + single) ===
    C_CELLS = [
        ("C4-awq-4bit",        "C4-awq-4bit"),
        ("C5-awq-6bit",        "C5-awq-6bit"),
        ("C6-autoround-int4",  "C6-autoround-int4"),
        ("C7-gptq-groxaxo",    "C7-gptq-groxaxo"),
        ("C8-gptq-qwopus",     "C8-gptq-qwopus"),
    ]
    for variant_label, base in C_CELLS:
        tps = free_run_tok_per_s(base)
        for mode in ("multi", "single"):
            cmp_path = ADD_C / "compare" / f"{base}-vs-bf16-{mode}.json"
            row = row_from_cmp(variant_label, mode, cmp_path, tok_per_s=tps)
            if row is not None:
                rows.append(row)

    # === Write CSV ===
    csv_path = ADD_C / "leaderboard_unified.csv"
    fields = ["variant", "mode", "n_positions",
              "kl_a_to_b_mean_bits", "kl_b_to_a_mean_bits", "js_mean_bits",
              "kl_max_bits", "js_max_bits",
              "free_run_tok_per_s", "source"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}  ({len(rows)} rows)")

    # === Write Markdown table ===
    md_path = ADD_C / "leaderboard_unified.md"
    md = ["# Unified KL/JSD Leaderboard",
          "",
          "All comparisons are vs `bf16-ref-{multi,single}.safetensors` (Exp 11 reference).",
          "Bits = nats / ln(2). Numbers in scientific notation; throughput from free-run (Addendum A).",
          "",
          "## Multi-prompt (504 positions = 8 prompts × 63 tokens)",
          "",
          "| Variant | Mean KL A→B (bits) | Mean KL B→A (bits) | Mean JSD (bits) | Max KL (bits) | Max JSD (bits) | Free-run tok/s |",
          "|---|---:|---:|---:|---:|---:|---:|"]

    def fmt_sci(x):
        if x is None:
            return "—"
        # Normalize to engineering-friendly e-X
        return f"{x:.3e}"

    def fmt_tps(x):
        return f"{x:.1f}" if x is not None else "—"

    multi = [r for r in rows if r["mode"] == "multi"]
    # Stable ordering
    order = ["bf16-self", "fp8", "fp8-mtp3", "nvfp4",
             "B2-kv-fp8-only", "B3-fp8w-kv-auto", "B4-fp8w-kv-fp8",
             "C4-awq-4bit", "C5-awq-6bit", "C6-autoround-int4",
             "C7-gptq-groxaxo", "C8-gptq-qwopus"]
    multi_sorted = sorted(multi, key=lambda r: order.index(r["variant"]) if r["variant"] in order else 999)
    for r in multi_sorted:
        md.append(f"| `{r['variant']}` | {fmt_sci(r['kl_a_to_b_mean_bits'])} "
                  f"| {fmt_sci(r['kl_b_to_a_mean_bits'])} "
                  f"| {fmt_sci(r['js_mean_bits'])} "
                  f"| {fmt_sci(r['kl_max_bits'])} "
                  f"| {fmt_sci(r['js_max_bits'])} "
                  f"| {fmt_tps(r['free_run_tok_per_s'])} |")

    md += ["",
           "## Single-prompt (16 positions, 1 prompt × ~17 tokens)",
           "",
           "| Variant | Mean KL A→B (bits) | Mean KL B→A (bits) | Mean JSD (bits) | Max KL (bits) | Max JSD (bits) |",
           "|---|---:|---:|---:|---:|---:|"]
    single = [r for r in rows if r["mode"] == "single"]
    single_sorted = sorted(single, key=lambda r: order.index(r["variant"]) if r["variant"] in order else 999)
    for r in single_sorted:
        md.append(f"| `{r['variant']}` | {fmt_sci(r['kl_a_to_b_mean_bits'])} "
                  f"| {fmt_sci(r['kl_b_to_a_mean_bits'])} "
                  f"| {fmt_sci(r['js_mean_bits'])} "
                  f"| {fmt_sci(r['kl_max_bits'])} "
                  f"| {fmt_sci(r['js_max_bits'])} |")

    md += ["",
           "## Notes",
           "",
           "- **`bf16-self`**: noise floor (same model, same kernels, two runs). Any divergence below this is indistinguishable from run-to-run nondeterminism.",
           "- **Addendum B cells (B2/B3/B4)**: compared vs B1 (BF16-weights + auto-KV), not vs Exp 11's BF16 ref. These isolate KV-cache vs weight-quant contributions to drift.",
           "- **Addendum C cells (C4–C8)**: compared vs Exp 11's BF16 ref, same protocol as FP8/NVFP4.",
           "- **Max KL ≈ 0.69 bits** in JSD column = `ln(2)` → indicates at least one position where the two distributions are fully disjoint (top-1 swap with vanishing mass elsewhere). Common at quant boundaries.",
           "- **Free-run tok/s** is from Addendum A (1024-token generation, 8 prompts). Teacher-forced collect timings are not throughput-representative (include prefill + warmup).",
           ""]

    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}  ({len(md)} lines)")


if __name__ == "__main__":
    main()
