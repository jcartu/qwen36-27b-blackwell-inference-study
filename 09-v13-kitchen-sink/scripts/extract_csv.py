#!/usr/bin/env python3
"""Extract a CSV row per cell from a directory of benchmark JSONs."""
import json
import csv
import sys
from pathlib import Path

FIELDS = [
    "experiment", "build", "config", "phase", "rep", "cell",
    "concurrency", "context_tokens",
    "aggregate_tps", "per_request_avg_tps",
    "ttft_avg_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
    "itl_avg_ms", "itl_p50_ms", "itl_p90_ms", "itl_p99_ms",
    "output_tps_per_user_avg", "output_tps_per_user_p50",
    "e2e_output_tps_per_user_avg",
    "server_gen_throughput", "server_utilization",
    "spec_accept_rate", "spec_accept_length",
    "avg_running_reqs", "max_running_reqs", "queue_fraction",
    "completed_request_count", "num_errors",
    "capacity_limited", "underfilled",
    "warmup_timed_out", "measurement_seconds",
    "source_file",
]


def cell_label(r):
    return f"c{r['concurrency']}-ctx{r['context_tokens']}"


def row_from_result(r, *, experiment, build, config, phase, rep, source):
    def msec(x):
        return round(x * 1000, 3) if isinstance(x, (int, float)) else x
    return {
        "experiment": experiment,
        "build": build,
        "config": config,
        "phase": phase,
        "rep": rep,
        "cell": cell_label(r),
        "concurrency": r["concurrency"],
        "context_tokens": r["context_tokens"],
        "aggregate_tps": round(r.get("aggregate_tps", 0), 3),
        "per_request_avg_tps": round(r.get("per_request_avg_tps", 0), 3),
        "ttft_avg_ms": msec(r.get("ttft_avg")),
        "ttft_p50_ms": msec(r.get("ttft_p50")),
        "ttft_p90_ms": msec(r.get("ttft_p90")),
        "ttft_p99_ms": msec(r.get("ttft_p99")),
        "itl_avg_ms": msec(r.get("inter_token_latency_avg")),
        "itl_p50_ms": msec(r.get("inter_token_latency_p50")),
        "itl_p90_ms": msec(r.get("inter_token_latency_p90")),
        "itl_p99_ms": msec(r.get("inter_token_latency_p99")),
        "output_tps_per_user_avg": round(r.get("output_tps_per_user_avg", 0), 3),
        "output_tps_per_user_p50": round(r.get("output_tps_per_user_p50", 0), 3),
        "e2e_output_tps_per_user_avg": round(r.get("e2e_output_tps_per_user_avg", 0), 3),
        "server_gen_throughput": round(r.get("server_gen_throughput", 0), 3),
        "server_utilization": round(r.get("server_utilization", 0), 4),
        "spec_accept_rate": round(r.get("server_spec_accept_rate", 0), 4),
        "spec_accept_length": round(r.get("server_spec_accept_length", 0), 4),
        "avg_running_reqs": r.get("avg_running_reqs"),
        "max_running_reqs": r.get("max_running_reqs"),
        "queue_fraction": round(r.get("queue_fraction", 0), 4),
        "completed_request_count": r.get("completed_request_count"),
        "num_errors": r.get("num_errors"),
        "capacity_limited": r.get("capacity_limited"),
        "underfilled": r.get("underfilled"),
        "warmup_timed_out": r.get("warmup_timed_out"),
        "measurement_seconds": round(r.get("measurement_seconds", 0), 2),
        "source_file": str(source),
    }


def main():
    if len(sys.argv) < 3:
        print("usage: extract_csv.py <output.csv> <json_dir> [json_dir ...]", file=sys.stderr)
        sys.exit(1)
    out_csv = Path(sys.argv[1])
    json_dirs = [Path(p) for p in sys.argv[2:]]
    rows = []
    for d in json_dirs:
        for jf in sorted(d.rglob("*.json")):
            if jf.name.startswith("_"):
                continue
            try:
                data = json.loads(jf.read_text())
            except Exception as e:
                print(f"[skip] {jf}: {e}", file=sys.stderr)
                continue
            # Infer phase/rep/config from path
            parts = jf.parts
            config = "bf16-dflash" if "A-bf16-dflash" in parts else ("fp8-mtp3" if "B-fp8-mtp3" in parts else "unknown")
            phase = next((p for p in parts if p.startswith("phase")), "unknown")
            rep_part = next((p for p in parts if p.startswith("run")), "")
            rep = int(rep_part[3:]) if rep_part.startswith("run") else 0
            build = "harness-v0.4.8" if "v0.4.8" in str(jf) else ("harness-upstream" if "upstream" in str(jf) else "harness-v0.4.8")
            for r in data.get("results", []):
                rows.append(row_from_result(
                    r,
                    experiment="exp-09-v13-kitchen-sink",
                    build=build,
                    config=config,
                    phase=phase,
                    rep=rep,
                    source=jf.relative_to(jf.parents[len(parts) - parts.index("v13-kitchen-sink-bf16dflash-and-fp8mtp3")]) if "v13-kitchen-sink-bf16dflash-and-fp8mtp3" in parts else jf.name,
                ))
    rows.sort(key=lambda r: (r["config"], r["phase"], r["rep"], r["concurrency"], r["context_tokens"]))
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
