#!/usr/bin/env python3
"""Build leaderboard.csv from Exp 10 results.

Walks results/{stage1,stage2,stage3,frontier}/*/teb-results.json and emits a
unified CSV with one row per cell. Columns are tuned for direct comparison of
local vLLM cells against frontier API yardsticks.

Usage:
  build_leaderboard.py [--results-dir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


COLUMNS = [
    "stage",          # stage1 | stage2 | stage3 | frontier
    "id",             # cell directory name
    "kind",           # local | frontier
    "image_or_provider",
    "model",
    "quant",          # fp8 | bf16 | api
    "spec",           # mtp/N3 | dflash/N8 | -
    "tool_parser",    # qwen3_xml | qwen3_coder | -
    "final_score",
    "responsiveness",
    "median_turn_ms",
    "tool_call_quality",
    "compliance",
    "throughput_tps", # if present
    "wall_seconds",
    "scenarios_total",
    "scenarios_pass",
]


def parse_cell_meta(stage: str, cell_id: str) -> dict[str, str]:
    """Derive image/quant/spec/parser from cell id naming convention."""
    meta = {
        "image_or_provider": "",
        "quant": "",
        "spec": "",
        "tool_parser": "",
    }
    if stage == "frontier":
        meta["image_or_provider"] = "api"
        meta["quant"] = "api"
        meta["spec"] = "-"
        meta["tool_parser"] = "-"
        return meta

    # Local cell ids look like:
    #   S1-v13-fp8-xml
    #   S1-v13-fp8-coder
    #   S2-v13-bf16-qwen3_xml
    #   S2-nightly-bf16-qwen3_coder
    #   S3-nightly-fp8-qwen3_xml
    parts = cell_id.split("-", 3)
    if len(parts) >= 4:
        _, image_tag, quant, parser_tail = parts
        meta["image_or_provider"] = (
            "repne/vllm:v13" if image_tag == "v13" else "vllm/vllm-openai:nightly"
        )
        meta["quant"] = quant
        if quant == "fp8":
            meta["spec"] = "mtp/N3"
        elif quant == "bf16":
            meta["spec"] = "dflash/N8"
        # Parser tail may be "xml", "coder", "qwen3_xml", or "qwen3_coder"
        if parser_tail in {"xml", "coder"}:
            meta["tool_parser"] = f"qwen3_{parser_tail}"
        else:
            meta["tool_parser"] = parser_tail
    return meta


def get(d: dict, *keys: str, default: Any = "") -> Any:
    """Safe nested .get for results JSON."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def row_for_cell(stage: str, cell_dir: Path) -> dict[str, Any] | None:
    results_json = cell_dir / "teb-results.json"
    if not results_json.exists():
        return None
    try:
        d = json.loads(results_json.read_text())
    except Exception as e:
        print(f"  warn: failed to parse {results_json}: {e}", file=sys.stderr)
        return None

    meta = parse_cell_meta(stage, cell_dir.name)
    scores = d.get("scores", {}) or {}

    # tool-eval-bench writes per-scenario list under "scenarios" or "results"
    scenarios = d.get("scenarios") or d.get("results") or []
    n_total = len(scenarios) if isinstance(scenarios, list) else 0
    n_pass = (
        sum(1 for s in scenarios if isinstance(s, dict) and s.get("status") == "pass")
        if n_total
        else 0
    )

    return {
        "stage": stage,
        "id": cell_dir.name,
        "kind": "frontier" if stage == "frontier" else "local",
        "image_or_provider": meta["image_or_provider"],
        "model": d.get("model") or get(d, "config", "model") or "",
        "quant": meta["quant"],
        "spec": meta["spec"],
        "tool_parser": meta["tool_parser"],
        "final_score": d.get("final_score", ""),
        "responsiveness": scores.get("responsiveness", ""),
        "median_turn_ms": scores.get("median_turn_ms", ""),
        "tool_call_quality": scores.get("tool_call_quality", ""),
        "compliance": scores.get("compliance", ""),
        "throughput_tps": scores.get("throughput_tps", ""),
        "wall_seconds": d.get("wall_seconds", "") or d.get("duration_seconds", ""),
        "scenarios_total": n_total,
        "scenarios_pass": n_pass,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default="/home/josh/qwen-vllm-test/sweeps/10-parser-axis/results",
    )
    ap.add_argument(
        "--out",
        default="/home/josh/qwen-vllm-test/sweeps/10-parser-axis/leaderboard.csv",
    )
    args = ap.parse_args()

    root = Path(args.results_dir)
    rows: list[dict[str, Any]] = []
    for stage in ("stage1", "stage2", "stage3", "frontier"):
        stage_dir = root / stage
        if not stage_dir.is_dir():
            continue
        for cell_dir in sorted(stage_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            r = row_for_cell(stage, cell_dir)
            if r is None:
                print(f"  skip {stage}/{cell_dir.name} (no results)", file=sys.stderr)
                continue
            rows.append(r)

    if not rows:
        print("No completed cells found.", file=sys.stderr)
        return 1

    # Sort: local cells first (stage order), then frontier by final_score desc
    def sort_key(r: dict[str, Any]):
        order = {"stage1": 0, "stage2": 1, "stage3": 2, "frontier": 3}
        score = r.get("final_score")
        try:
            score = float(score) if score != "" else -1.0
        except (TypeError, ValueError):
            score = -1.0
        return (order.get(r["stage"], 9), -score if r["stage"] == "frontier" else 0, r["id"])

    rows.sort(key=sort_key)

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})

    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
