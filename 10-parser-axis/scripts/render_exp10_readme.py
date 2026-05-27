#!/usr/bin/env python3
"""Render the final Exp 10 README.md from template + per-cell teb-results.json.

Fills <PLACEHOLDER> tokens in EXP10_README_TEMPLATE.md with the numbers from
results/{stage1,stage2,stage3,frontier}/*/teb-results.json.

Usage:
  render_exp10_readme.py <template.md> <out.md> [--results-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_cell(cell_dir: Path) -> dict | None:
    f = cell_dir / "teb-results.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception as e:
        print(f"  warn: parse failed for {f}: {e}", file=sys.stderr)
        return None


def fmt(v: Any, kind: str = "num") -> str:
    if v is None or v == "":
        return "—"
    try:
        if kind == "ms":
            return f"{float(v):.0f}"
        if kind == "score":
            return f"{float(v):.1f}"
        if kind == "rate":
            return f"{float(v):.3f}"
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def cell_vals(cell: dict | None) -> dict[str, str]:
    """Extract canonical fields with rendering-ready formatting."""
    if not cell:
        return {
            "FINAL": "—",
            "RESP": "—",
            "MTM": "—",
            "TCQ": "—",
            "CMP": "—",
        }
    s = cell.get("scores", {}) or {}
    return {
        "FINAL": fmt(cell.get("final_score"), "score"),
        "RESP": fmt(s.get("responsiveness"), "rate"),
        "MTM": fmt(s.get("median_turn_ms"), "ms"),
        "TCQ": fmt(s.get("tool_call_quality"), "rate"),
        "CMP": fmt(s.get("compliance"), "rate"),
    }


def read_winner(results_dir: Path, stage: str) -> str:
    f = results_dir / stage / "winner.txt"
    if not f.exists():
        return ""
    return f.read_text().strip()


def derive_verdicts(
    s1_xml: dict | None,
    s1_coder: dict | None,
    s2_v13: dict | None,
    s2_nightly: dict | None,
    s3_v13: dict | None,
    s3_nightly: dict | None,
    p_star: str,
    s2_winner: str,
    s3_winner: str,
) -> dict[str, str]:
    def score(c):
        return (c or {}).get("final_score")

    def resp(c):
        return ((c or {}).get("scores") or {}).get("responsiveness")

    verdicts = {}

    # Parser axis
    a, b = score(s1_xml), score(s1_coder)
    if a is None or b is None:
        verdicts["VERDICT_PARSER"] = "— (incomplete data)"
    else:
        delta = a - b
        if abs(delta) < 0.5:
            verdicts["VERDICT_PARSER"] = (
                f"Tie within noise. `qwen3_xml`={a:.1f}, `qwen3_coder`={b:.1f}. "
                f"Picked **{p_star}** as P\\* on tiebreak (responsiveness, then median_turn_ms)."
            )
        else:
            better = "qwen3_xml" if delta > 0 else "qwen3_coder"
            worse = "qwen3_coder" if delta > 0 else "qwen3_xml"
            verdicts["VERDICT_PARSER"] = (
                f"**`{better}` wins** by {abs(delta):.1f} `final_score` points "
                f"({a:.1f} vs {b:.1f}). Adopted as P\\* for downstream stages."
            )

    # BF16 image axis (efficiency: responsiveness)
    rv, rn = resp(s2_v13), resp(s2_nightly)
    if rv is None or rn is None:
        verdicts["VERDICT_BF16_IMAGE"] = "— (incomplete data)"
    else:
        delta = rv - rn
        if abs(delta) < 0.01:
            verdicts["VERDICT_BF16_IMAGE"] = (
                f"Tie within noise on responsiveness (v13={rv:.3f}, nightly={rn:.3f}). "
                f"Picked **{s2_winner}** on tiebreak."
            )
        else:
            verdicts["VERDICT_BF16_IMAGE"] = (
                f"**{s2_winner} wins** on responsiveness "
                f"(v13={rv:.3f}, nightly={rn:.3f}, Δ={delta:+.3f})."
            )

    # FP8 image axis
    rv3, rn3 = resp(s3_v13), resp(s3_nightly)
    if rv3 is None or rn3 is None:
        verdicts["VERDICT_FP8_IMAGE"] = "— (incomplete data)"
    else:
        delta = rv3 - rn3
        if abs(delta) < 0.01:
            verdicts["VERDICT_FP8_IMAGE"] = (
                f"Tie within noise on responsiveness (v13={rv3:.3f}, nightly={rn3:.3f}). "
                f"Picked **{s3_winner}** on tiebreak. "
                f"{'Confirms' if s3_winner == s2_winner else 'Contradicts'} the BF16 result."
            )
        else:
            agree = "Confirms" if s3_winner == s2_winner else "Contradicts"
            verdicts["VERDICT_FP8_IMAGE"] = (
                f"**{s3_winner} wins** on responsiveness "
                f"(v13={rv3:.3f}, nightly={rn3:.3f}, Δ={delta:+.3f}). "
                f"{agree} the BF16 image-axis result."
            )

    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("output")
    ap.add_argument(
        "--results-dir",
        default="/home/josh/qwen-vllm-test/sweeps/10-parser-axis/results",
    )
    args = ap.parse_args()

    tpl = Path(args.template).read_text()
    R = Path(args.results_dir)

    p_star = read_winner(R, "stage1") or "qwen3_xml"
    s2_winner = read_winner(R, "stage2") or "—"
    s3_winner = read_winner(R, "stage3") or "—"
    p_short = p_star.replace("qwen3_", "")  # xml | coder

    # Stage 1
    s1_xml_dir   = R / "stage1" / "S1-v13-fp8-xml"
    s1_coder_dir = R / "stage1" / "S1-v13-fp8-coder"
    s1_xml   = load_cell(s1_xml_dir)
    s1_coder = load_cell(s1_coder_dir)
    v_xml   = cell_vals(s1_xml)
    v_coder = cell_vals(s1_coder)

    # Stage 2
    s2_v13_dir     = R / "stage2" / f"S2-v13-bf16-{p_star}"
    s2_nightly_dir = R / "stage2" / f"S2-nightly-bf16-{p_star}"
    s2_v13     = load_cell(s2_v13_dir)
    s2_nightly = load_cell(s2_nightly_dir)
    v_s2_v13     = cell_vals(s2_v13)
    v_s2_nightly = cell_vals(s2_nightly)

    # Stage 3 — v13 cell is reused from Stage 1 winner
    s3_v13_dir     = R / "stage1" / f"S1-v13-fp8-{p_short}"
    s3_nightly_dir = R / "stage3" / f"S3-nightly-fp8-{p_star}"
    s3_v13     = load_cell(s3_v13_dir)
    s3_nightly = load_cell(s3_nightly_dir)
    v_s3_v13     = cell_vals(s3_v13)
    v_s3_nightly = cell_vals(s3_nightly)

    # Frontier (Y1..Y7)
    frontier_ids = [
        ("Y1", "Y1-claude-sonnet-4.6"),
        ("Y2", "Y2-claude-haiku-4.5"),
        ("Y3", "Y3-gpt-5.5"),
        ("Y4", "Y4-gpt-5-mini"),
        ("Y5", "Y5-gpt-5-nano"),
        ("Y6", "Y6-gemini-3.5-flash"),
        ("Y7", "Y7-qwen-235b-cerebras"),
    ]
    frontier_vals: dict[str, dict[str, str]] = {}
    for tag, name in frontier_ids:
        frontier_vals[tag] = cell_vals(load_cell(R / "frontier" / name))

    verdicts = derive_verdicts(
        s1_xml, s1_coder, s2_v13, s2_nightly, s3_v13, s3_nightly,
        p_star, s2_winner, s3_winner,
    )

    repls: dict[str, str] = {
        "<P_STAR>": f"`{p_star}`",
        "<S2_WINNER>": f"`{s2_winner}`",
        "<S3_WINNER>": f"`{s3_winner}`",
        "<S1_XML_FINAL>": v_xml["FINAL"],
        "<S1_XML_RESP>":  v_xml["RESP"],
        "<S1_XML_MTM>":   v_xml["MTM"],
        "<S1_XML_TCQ>":   v_xml["TCQ"],
        "<S1_XML_CMP>":   v_xml["CMP"],
        "<S1_CODER_FINAL>": v_coder["FINAL"],
        "<S1_CODER_RESP>":  v_coder["RESP"],
        "<S1_CODER_MTM>":   v_coder["MTM"],
        "<S1_CODER_TCQ>":   v_coder["TCQ"],
        "<S1_CODER_CMP>":   v_coder["CMP"],
        "<S2_V13_FINAL>":     v_s2_v13["FINAL"],
        "<S2_V13_RESP>":      v_s2_v13["RESP"],
        "<S2_V13_MTM>":       v_s2_v13["MTM"],
        "<S2_NIGHTLY_FINAL>": v_s2_nightly["FINAL"],
        "<S2_NIGHTLY_RESP>":  v_s2_nightly["RESP"],
        "<S2_NIGHTLY_MTM>":   v_s2_nightly["MTM"],
        "<S3_V13_FINAL>":     v_s3_v13["FINAL"],
        "<S3_V13_RESP>":      v_s3_v13["RESP"],
        "<S3_V13_MTM>":       v_s3_v13["MTM"],
        "<S3_NIGHTLY_FINAL>": v_s3_nightly["FINAL"],
        "<S3_NIGHTLY_RESP>":  v_s3_nightly["RESP"],
        "<S3_NIGHTLY_MTM>":   v_s3_nightly["MTM"],
        "<VERDICT_PARSER>":      verdicts.get("VERDICT_PARSER", "—"),
        "<VERDICT_BF16_IMAGE>":  verdicts.get("VERDICT_BF16_IMAGE", "—"),
        "<VERDICT_FP8_IMAGE>":   verdicts.get("VERDICT_FP8_IMAGE", "—"),
    }
    for tag, _ in frontier_ids:
        v = frontier_vals[tag]
        repls[f"<{tag}_FINAL>"] = v["FINAL"]
        repls[f"<{tag}_RESP>"]  = v["RESP"]
        repls[f"<{tag}_MTM>"]   = v["MTM"]

    out = tpl
    missing = []
    for k, val in repls.items():
        if k in out:
            out = out.replace(k, val)
        else:
            # Some templates may not contain every placeholder; that's fine.
            pass

    # Anything left looking like <FOO> is a placeholder we forgot.
    import re
    leftovers = sorted(set(re.findall(r"<[A-Z][A-Z0-9_]+>", out)))
    if leftovers:
        print(f"warn: unfilled placeholders: {leftovers}", file=sys.stderr)

    Path(args.output).write_text(out)
    print(f"wrote {args.output} ({len(out)} bytes)")
    if missing:
        print(f"warn: missing replacements: {missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
