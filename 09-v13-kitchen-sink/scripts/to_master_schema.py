#!/usr/bin/env python3
"""Transform our rich 35-col experiment CSV into the 10-col master schema.

Master schema (from jcartu/qwen36-27b-blackwell-inference-study/master-results.csv):
  experiment,build,cell,aggregate_tps,ttft_avg_ms,ttft_p99_ms,
  itl_avg_ms,per_user_tps,spec_accept_rate,server_utilization

Cell label convention in master: `c{n}_ctx{tokens}` (underscore).
Our local: `c{n}-ctx{16k|32k|...}`. Convert to master form.

Usage: to_master_schema.py <in.csv> <out.csv>
"""
from __future__ import annotations
import csv
import sys
import re
from pathlib import Path


_CELL_RE = re.compile(r"^c(\d+)-ctx(.+)$")


def normalize_cell(cell: str, ctx_tokens: int) -> str:
    """Convert c1-ctx16k -> c1_ctx16384 using authoritative ctx_tokens."""
    m = _CELL_RE.match(cell)
    if not m:
        return cell.replace("-", "_")
    n = int(m.group(1))
    return f"c{n}_ctx{ctx_tokens}"


def per_user_tps(row) -> float | None:
    """Master schema's per_user_tps is the user-perceived effective t/s.

    Our schema has output_tps_per_user_avg which is the closest match.
    """
    v = row.get("output_tps_per_user_avg")
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def fnum(v):
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: to_master_schema.py <in.csv> <out.csv>", file=sys.stderr)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)
    out_fields = [
        "experiment", "build", "cell", "aggregate_tps",
        "ttft_avg_ms", "ttft_p99_ms",
        "itl_avg_ms", "per_user_tps",
        "spec_accept_rate", "server_utilization",
    ]
    out_rows = []
    for r in rows_in:
        try:
            ctx = int(r["context_tokens"])
        except (KeyError, ValueError):
            ctx = 0
        # Compose experiment label: exp09-<config>
        config = r.get("config", "unknown")
        exp = f"exp09-{config}"
        out_rows.append({
            "experiment": exp,
            "build": r.get("build", ""),
            "cell": normalize_cell(r.get("cell", ""), ctx),
            "aggregate_tps": fnum(r.get("aggregate_tps")),
            "ttft_avg_ms": fnum(r.get("ttft_avg_ms")),
            "ttft_p99_ms": fnum(r.get("ttft_p99_ms")),
            "itl_avg_ms": fnum(r.get("itl_avg_ms")),
            "per_user_tps": per_user_tps(r),
            "spec_accept_rate": fnum(r.get("spec_accept_rate")),
            "server_utilization": fnum(r.get("server_utilization")),
        })
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote {len(out_rows)} rows to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
