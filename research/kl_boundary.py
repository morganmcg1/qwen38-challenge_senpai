#!/usr/bin/env python3
"""Per-verify-width key-length coverage for one traced MTP run.

Answers: at which absolute key lengths did each verify width actually run?
A width whose rounds never reached a suspected numeric boundary has not been
tested at that boundary, however many rows it matched.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--seed", type=int, default=512)
    ap.add_argument("--boundary", type=int, default=1022)
    ap.add_argument("--out")
    args = ap.parse_args()

    phase = None
    rounds = []
    for line in open(args.trace):
        if "measuring native-MTP decode" in line:
            phase = "mtp"
        elif "measuring the TRUE serial control" in line:
            phase = "serial"
        elif "generating the MTP reference" in line:
            phase = "reference"
        m = ROUND_RE.search(line)
        if m and phase == "mtp":
            rounds.append(tuple(int(g) for g in m.groups()))

    pos = args.seed
    per_width = defaultdict(lambda: {"rounds": 0, "max_key_len": 0, "at_boundary": []})
    for rn, d, acc in rounds:
        width = d + 1
        key_len = pos + width
        rec = per_width[width]
        rec["rounds"] += 1
        rec["max_key_len"] = max(rec["max_key_len"], key_len)
        if key_len >= args.boundary:
            rec["at_boundary"].append({"round": rn, "key_len": key_len})
        pos += acc + 1

    report = {
        "trace": args.trace,
        "seed_tokens": args.seed,
        "boundary_key_len": args.boundary,
        "final_position": pos,
        "per_width": {str(k): per_width[k] for k in sorted(per_width)},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
