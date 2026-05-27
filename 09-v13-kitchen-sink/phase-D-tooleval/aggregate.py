#!/usr/bin/env python3
"""Aggregate Phase D tool-eval-bench results into a leaderboard CSV.

Usage: python3 aggregate.py <results-dir>  >  leaderboard.csv

Each sub-dir under <results-dir> should be a run dir containing teb-results.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def collect(results_dir: Path) -> list[dict]:
    rows = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        jf = sub / "teb-results.json"
        if not jf.exists():
            continue
        try:
            d = json.loads(jf.read_text())
        except Exception as e:
            print(f"# warn: failed to parse {jf}: {e}", file=sys.stderr)
            continue
        scores = d.get("scores", {})
        meta = d.get("metadata", {})
        cfg = d.get("config", {})
        rows.append({
            "endpoint_id": sub.name,
            "model": cfg.get("model") or meta.get("model"),
            "base_url": cfg.get("base_url") or meta.get("base_url"),
            "final_score": d.get("final_score", scores.get("final_score")),
            "rating": d.get("rating", scores.get("rating")),
            "total_points": scores.get("total_points"),
            "max_points": scores.get("max_points"),
            "deployability": scores.get("deployability"),
            "responsiveness": scores.get("responsiveness"),
            "median_turn_ms": scores.get("median_turn_ms"),
            "total_tokens": scores.get("total_tokens"),
            "token_efficiency": scores.get("token_efficiency"),
            "worst_category": scores.get("worst_category"),
            "scenario_count": cfg.get("scenario_count"),
            "thinking_enabled": meta.get("thinking_enabled"),
            "run_id": d.get("run_id"),
        })
    # Sort by final_score desc, then responsiveness desc as tiebreaker
    rows.sort(key=lambda r: (-(r["final_score"] or 0), -(r["responsiveness"] or 0)))
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aggregate.py <results-dir>", file=sys.stderr)
        return 2
    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"not a dir: {results_dir}", file=sys.stderr)
        return 2
    rows = collect(results_dir)
    if not rows:
        print("# no results found", file=sys.stderr)
        return 1
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
