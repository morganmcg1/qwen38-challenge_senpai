#!/usr/bin/env python3
"""E71 rung 1. Gate the Tests/-only census harness against a real scored leg.

    usage: research/e71_rung1_gate.py --trace research/out/TAG/trace.txt \
                                      --census research/out/TAG2/census.json

The census harness times `callWithHiddenAndNormed -> top-2 -> eval`. Inside a
scored leg the same work is `verify_build_us + eval_wall_us`, so the leg's
in-situ V(M) is the correct comparison series, not the full round.

Round 1 is a 512-row head prime worth about +29.5 ms (E65) and is discarded.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=(-?\d+)")

# senpai/campaign-ledger.md 200(E). In-situ V per verify width, microseconds.
LEDGER_V_US = {1: 64979, 2: 69509, 3: 73985, 4: 89610, 5: 114934,
               6: 131749, 7: 150629, 8: 167074, 9: 190483}


def parse_rounds(paths: list[Path]) -> list[dict]:
    rounds = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            m = ROUND_RE.match(line.strip())
            if not m:
                continue
            row = {"round": int(m.group(1)), "d": int(m.group(2)),
                   "acc": int(m.group(3)), "source": path.name}
            row.update({k: int(v) for k, v in KV_RE.findall(m.group(4))})
            rounds.append(row)
    return rounds


def summarise(rounds: list[dict]) -> dict:
    by_width: dict[int, list[dict]] = {}
    for row in rounds:
        # E65: round 1 is a 512-row head prime costing about +29.5 ms.
        if row["round"] <= 1:
            continue
        by_width.setdefault(row["d"] + 1, []).append(row)
    out = {}
    for width, rows in sorted(by_width.items()):
        v = [r.get("verify_build_us", 0) + r.get("eval_wall_us", 0)
             for r in rows]
        out[width] = {
            "n": len(rows),
            "v_median_ms": statistics.median(v) / 1e3,
            "v_mean_ms": statistics.mean(v) / 1e3,
            "verify_build_median_ms": statistics.median(
                [r.get("verify_build_us", 0) for r in rows]) / 1e3,
            "eval_wall_median_ms": statistics.median(
                [r.get("eval_wall_us", 0) for r in rows]) / 1e3,
            "round_median_ms": statistics.median(
                [r.get("round_us", 0) for r in rows]) / 1e3,
            "draft_build_median_ms": statistics.median(
                [r.get("draft_build_us", 0) for r in rows]) / 1e3,
            "ledger_200E_v_ms": LEDGER_V_US.get(width, 0) / 1e3 or None,
        }
    return out


def census_curve(path: Path) -> dict:
    payload = json.loads(path.read_text())
    per_width: dict[int, list[float]] = {}
    for block in payload["blocks"]:
        if block["arm"] != "baseline":
            continue
        per_width.setdefault(block["width"], []).append(
            1e3 * block["seconds_median"])
    return {
        w: {"mean_ms": sum(v) / len(v), "values_ms": v, "n": len(v),
            "half_range_ms": (max(v) - min(v)) / 2 if len(v) > 1 else 0.0}
        for w, v in sorted(per_width.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", nargs="*", default=[])
    parser.add_argument("--census")
    parser.add_argument("--json")
    args = parser.parse_args()

    report: dict = {
        "experiment": "e71-in-situ-width-tax-census",
        "rung": 1,
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }

    paths = []
    for pattern in args.trace:
        p = Path(pattern)
        paths.extend(sorted(p.parent.glob(p.name + "*")) if not p.exists()
                     else [p])
    if paths:
        rounds = parse_rounds(paths)
        report["leg"] = {
            "trace_files": [str(p) for p in paths],
            "rounds_parsed": len(rounds),
            "rounds_used": len([r for r in rounds if r["round"] > 1]),
            "per_width": {str(k): v for k, v in summarise(rounds).items()},
        }

    if args.census:
        curve = census_curve(Path(args.census))
        report["census"] = {str(k): v for k, v in curve.items()}
        session_null = max((v["half_range_ms"] for v in curve.values()),
                           default=0.0)
        report["session_null_ms"] = session_null

        # The gate. The harness must reproduce the shape of the in-situ curve.
        rows = []
        leg = report.get("leg", {}).get("per_width", {})
        for width, v in curve.items():
            ledger = LEDGER_V_US.get(width)
            row = {
                "width": width,
                "census_ms": v["mean_ms"],
                "ledger_200E_v_ms": ledger / 1e3 if ledger else None,
            }
            if ledger:
                row["census_minus_ledger_ms"] = v["mean_ms"] - ledger / 1e3
                row["census_over_ledger"] = v["mean_ms"] / (ledger / 1e3)
            if str(width) in leg:
                row["leg_v_ms"] = leg[str(width)]["v_median_ms"]
                row["leg_n"] = leg[str(width)]["n"]
                row["census_minus_leg_ms"] = (
                    v["mean_ms"] - leg[str(width)]["v_median_ms"])
                row["head_overlap_estimate_ms"] = -row["census_minus_leg_ms"]
            rows.append(row)
        report["gate"] = {
            "rows": rows,
            "note": ("A negative census_minus_leg_ms is expected: E65 shows "
                     "verify_build_us overlaps head-chain GPU work inside a "
                     "real leg, so the leg's V is an upper bound on the "
                     "isolated verify cost. That residual IS the head-chain "
                     "overlap estimate."),
        }
        if 1 in curve and max(curve) > 1:
            top = max(curve)
            report["gate"]["census_width_tax_ms"] = (
                curve[top]["mean_ms"] - curve[1]["mean_ms"])
            report["gate"]["census_width_tax_widths"] = [1, top]

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
