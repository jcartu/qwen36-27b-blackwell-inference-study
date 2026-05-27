#!/usr/bin/env python3
"""Build the Exp 11 leaderboard CSV from compare/*.json.

Output columns:
    variant            (fp8 | fp8-mtp3 | nvfp4 | bf16-self)
    mode               (single | multi)
    n_positions
    kl_a_to_b_mean_bits
    kl_b_to_a_mean_bits
    js_mean_bits
    kl_max_bits
    js_max_bits

Bits = nats / ln(2).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "compare"
OUTPUT = COMPARE / "leaderboard.csv"

NATS_TO_BITS = 1.0 / math.log(2)


def load_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def n2b(x: float | None) -> float | None:
    if x is None:
        return None
    return float(x) * NATS_TO_BITS


def main() -> int:
    if not COMPARE.exists():
        print(f"ERROR: {COMPARE} does not exist", file=sys.stderr)
        return 1

    rows = []
    for json_path in sorted(COMPARE.glob("*.json")):
        name = json_path.stem  # e.g. "fp8-vs-bf16-single" or "bf16-self-single"
        if name.endswith("-single"):
            mode = "single"
            stem = name[: -len("-single")]
        elif name.endswith("-multi"):
            mode = "multi"
            stem = name[: -len("-multi")]
        else:
            print(f"  skip (unknown suffix): {json_path}")
            continue

        if stem.startswith("bf16-self"):
            variant = "bf16-self"
        elif stem.endswith("-vs-bf16"):
            variant = stem[: -len("-vs-bf16")]
        else:
            variant = stem

        data = load_summary(json_path)

        row = {
            "variant": variant,
            "mode": mode,
            "n_positions": data.get("n_positions") or data.get("num_positions"),
            "kl_a_to_b_mean_bits": n2b(data.get("kl_a_to_b_mean")),
            "kl_b_to_a_mean_bits": n2b(data.get("kl_b_to_a_mean")),
            "js_mean_bits": n2b(data.get("js_mean")),
            "kl_max_bits": n2b(data.get("kl_a_to_b_max") or
                                (max(data["kl_a_to_b_per_pos"]) if data.get("kl_a_to_b_per_pos") else None)),
            "js_max_bits": n2b(data.get("js_max") or
                                (max(data["js_per_pos"]) if data.get("js_per_pos") else None)),
            "skip_prefill_next": data.get("skip_prefill_next"),
        }
        rows.append(row)
        print(f"  loaded: {variant} {mode}: js_mean={row['js_mean_bits']:.6e} bits "
              f"({row['n_positions']} positions)" if row['js_mean_bits'] is not None
              else f"  loaded: {variant} {mode}: (missing JS) {row}")

    if not rows:
        print("No compare JSONs found. Run scripts/run_compare.sh first.", file=sys.stderr)
        return 2

    # Sort: variants first (bf16-self last as it's the noise floor), then mode
    variant_order = {"bf16-self": 99}
    rows.sort(key=lambda r: (variant_order.get(r["variant"], 0), r["variant"], r["mode"]))

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUTPUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
