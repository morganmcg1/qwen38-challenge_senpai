#!/usr/bin/env python3
"""E51 decisive-margin distribution between two arms.

The brief asks for this only for arm M, but the quantity is exactly what decides
whether a dose flips a token, so it is worth computing for whichever arm actually
moved the sensor.

For every position both arms evaluated:

  margin       = top1 - top2 declared logit, taken from the reference arm
  delta_logit  = max absolute change in either declared logit under the arm
  ratio        = margin / delta_logit

A small minimum ratio means the arm survived by luck on this fixture and would
be expected to flip a token on a prompt with a tighter margin. The reported
values are the declared hexfloat logits the session hands to the trusted parent,
not an intermediate, so they are logit displacements and not relative errors on
an inner quantity.

usage: research/e51_margin.py REFERENCE_ARM CANDIDATE_ARM
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

ROW = re.compile(
    r"^mtp-row: pos=(\d+) ids=(\d+),(\d+) v=([^,]+),(\S+)")


def rows(arm: pathlib.Path) -> dict[int, tuple[int, int, float, float]]:
    """Last emitted row per position, which is the one on the accepted path."""
    out: dict[int, tuple[int, int, float, float]] = {}
    for line in (arm / "trace.txt").read_text(errors="replace").splitlines():
        match = ROW.match(line)
        if match:
            out[int(match.group(1))] = (
                int(match.group(2)), int(match.group(3)),
                float.fromhex(match.group(4)), float.fromhex(match.group(5)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=pathlib.Path)
    parser.add_argument("candidate", type=pathlib.Path)
    parser.add_argument("--json")
    args = parser.parse_args()

    ref = rows(args.reference)
    cand = rows(args.candidate)
    shared = sorted(set(ref) & set(cand))

    per_pos = []
    for pos in shared:
        r_id0, r_id1, r_v0, r_v1 = ref[pos]
        c_id0, c_id1, c_v0, c_v1 = cand[pos]
        margin = r_v0 - r_v1
        delta = max(abs(c_v0 - r_v0), abs(c_v1 - r_v1))
        per_pos.append({
            "pos": pos,
            "top1_id_ref": r_id0,
            "top1_id_cand": c_id0,
            "top1_id_flipped": r_id0 != c_id0,
            "top2_id_flipped": r_id1 != c_id1,
            "margin": margin,
            "delta_logit": delta,
            "ratio": (margin / delta) if delta > 0 else float("inf"),
            "values_moved": (r_v0, r_v1) != (c_v0, c_v1),
        })

    moved = [p for p in per_pos if p["values_moved"]]
    flipped = [p for p in per_pos if p["top1_id_flipped"]
               or p["top2_id_flipped"]]
    finite = [p["ratio"] for p in per_pos if p["delta_logit"] > 0]

    report = {
        "reference": args.reference.name,
        "candidate": args.candidate.name,
        "shared_positions": len(shared),
        "positions_with_moved_logits": len(moved),
        "positions_with_flipped_ids": len(flipped),
        "flipped_positions": [p["pos"] for p in flipped],
        "min_margin_ratio": min(finite) if finite else None,
        "median_margin_ratio": statistics.median(finite) if finite else None
    }
    if moved:
        deltas = [p["delta_logit"] for p in moved]
        margins = [p["margin"] for p in moved]
        report["max_delta_logit"] = max(deltas)
        report["median_delta_logit"] = statistics.median(deltas)
        report["min_margin"] = min(margins)
        report["median_margin"] = statistics.median(margins)

    print(json.dumps(report, indent=2))
    if finite:
        tight = sorted((p for p in per_pos if p["delta_logit"] > 0),
                       key=lambda p: p["ratio"])[:5]
        print("\ntightest positions (pos, margin, delta_logit, ratio):")
        print("\n".join(f"  {p['pos']}  {p['margin']:.6f}  "
                        f"{p['delta_logit']:.6f}  {p['ratio']:.2f}"
                        for p in tight))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"summary": report, "per_position": per_pos}, indent=2)
            + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
