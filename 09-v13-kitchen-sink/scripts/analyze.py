#!/usr/bin/env python3
"""Aggregate per-cell statistics (mean, std, min, max) across N reps.

Reads the rich CSV from extract_csv.py, groups by (config, concurrency, context_tokens),
emits a markdown table + JSON summary.

Usage: analyze.py <rich.csv> <out-prefix>
  produces:  <out-prefix>.md      (human-readable matrix)
             <out-prefix>.json    (machine-readable summary)
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(v):
    try:
        return float(v) if v not in ("", None) else None
    except (TypeError, ValueError):
        return None


def aggregate(rows, key_fields=("config", "phase", "concurrency", "context_tokens")):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, "") for k in key_fields)
        groups[key].append(r)
    out = {}
    for key, rs in groups.items():
        tps = [f(r["aggregate_tps"]) for r in rs if f(r["aggregate_tps"])]
        ttft = [f(r["ttft_avg_ms"]) for r in rs if f(r["ttft_avg_ms"])]
        itl = [f(r["itl_avg_ms"]) for r in rs if f(r["itl_avg_ms"])]
        per_user = [f(r["output_tps_per_user_avg"]) for r in rs if f(r["output_tps_per_user_avg"])]
        spec_acc = [f(r["spec_accept_rate"]) for r in rs if f(r["spec_accept_rate"])]
        util = [f(r["server_utilization"]) for r in rs if f(r["server_utilization"])]

        def stats(xs):
            if not xs:
                return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
            return {
                "n": len(xs),
                "mean": round(statistics.fmean(xs), 3),
                "std": round(statistics.pstdev(xs), 3) if len(xs) > 1 else 0.0,
                "min": round(min(xs), 3),
                "max": round(max(xs), 3),
            }

        out[key] = {
            "config": key[0], "phase": key[1],
            "concurrency": int(key[2]) if key[2] else 0,
            "context_tokens": int(key[3]) if key[3] else 0,
            "n_reps": len(rs),
            "aggregate_tps": stats(tps),
            "ttft_avg_ms": stats(ttft),
            "itl_avg_ms": stats(itl),
            "per_user_tps": stats(per_user),
            "spec_accept_rate": stats(spec_acc),
            "server_utilization": stats(util),
        }
    return out


def ctx_label(t: int) -> str:
    if t == 0:
        return "0"
    if t == 16384:
        return "16k"
    if t == 32768:
        return "32k"
    if t == 65536:
        return "64k"
    if 131072 <= t <= 135000:  # bench reports 134144 for 131k target
        return "131k"
    return f"{t}"


def write_markdown(agg, out: Path, *, only_phase: str = "phase3-matrix") -> None:
    # Filter to matrix only
    rows = [v for v in agg.values() if v["phase"] == only_phase]
    rows.sort(key=lambda r: (r["config"], r["concurrency"], r["context_tokens"]))
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)

    concurrencies = sorted({r["concurrency"] for r in rows})
    contexts = sorted({r["context_tokens"] for r in rows})

    lines = ["# Aggregate matrix (mean ± std, N reps)", ""]
    for cfg, rs in by_cfg.items():
        n_reps = max(r["n_reps"] for r in rs)
        lines += [f"## Config: `{cfg}` (N up to {n_reps})", ""]
        # tok/s table
        lines += ["### Aggregate throughput (tok/s)", ""]
        header = "| concurrency \\ context | " + " | ".join(ctx_label(c) for c in contexts) + " |"
        sep = "|---|" + "|".join(["---:"] * len(contexts)) + "|"
        lines += [header, sep]
        cell_lookup = {(r["concurrency"], r["context_tokens"]): r for r in rs}
        for c in concurrencies:
            cells = []
            for ctx in contexts:
                r = cell_lookup.get((c, ctx))
                if not r or r["aggregate_tps"]["n"] == 0:
                    cells.append("—")
                else:
                    m = r["aggregate_tps"]["mean"]
                    s = r["aggregate_tps"]["std"]
                    cells.append(f"{m:.1f} ±{s:.1f}")
            lines.append(f"| c={c} | " + " | ".join(cells) + " |")
        lines += [""]
        # Spec accept rate
        lines += ["### Speculative decoding acceptance rate", ""]
        lines += [header, sep]
        for c in concurrencies:
            cells = []
            for ctx in contexts:
                r = cell_lookup.get((c, ctx))
                if not r or r["spec_accept_rate"]["n"] == 0:
                    cells.append("—")
                else:
                    cells.append(f"{r['spec_accept_rate']['mean']:.3f}")
            lines.append(f"| c={c} | " + " | ".join(cells) + " |")
        lines += [""]
        # ITL
        lines += ["### Inter-token latency (ms, avg)", ""]
        lines += [header, sep]
        for c in concurrencies:
            cells = []
            for ctx in contexts:
                r = cell_lookup.get((c, ctx))
                if not r or r["itl_avg_ms"]["n"] == 0:
                    cells.append("—")
                else:
                    cells.append(f"{r['itl_avg_ms']['mean']:.2f}")
            lines.append(f"| c={c} | " + " | ".join(cells) + " |")
        lines += [""]
    out.write_text("\n".join(lines))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: analyze.py <rich.csv> <out-prefix>", file=sys.stderr)
        return 2
    src, prefix = Path(sys.argv[1]), Path(sys.argv[2])
    rows = load(src)
    agg = aggregate(rows)
    # JSON
    json_path = Path(str(prefix) + ".json")
    json_path.write_text(json.dumps(
        {f"{k[0]}|{k[1]}|c{k[2]}|ctx{k[3]}": v for k, v in agg.items()},
        indent=2,
        default=str,
    ))
    print(f"Wrote {json_path}")
    # MD
    md_path = Path(str(prefix) + ".md")
    write_markdown(agg, md_path)
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
