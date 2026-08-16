#!/usr/bin/env python3
"""Split one traced MTP run into decode-index regimes.

The public fixture is a long-copy prompt whose easy copying stretch ends near
decode index 301 (the EOS region the fixed-window base was built for). The
rounds after it are the only low-acceptance material the local loop offers, so
a schedule that only wins on the copy stretch has to be visible as a regime
split rather than as a whole-window average.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)")
KV_RE = re.compile(r"([a-z_0-9]+)=([^\s]+)")


def parse(trace: str):
    phase = None
    rounds = []
    for line in open(trace):
        if "measuring native-MTP decode" in line:
            phase = "mtp"
        elif "measuring the TRUE serial control" in line:
            phase = "serial"
        elif "generating the MTP reference" in line:
            phase = "reference"
        m = ROUND_RE.search(line)
        if not m or phase != "mtp":
            continue
        kv = dict(KV_RE.findall(m.group(4)))
        rounds.append(
            {
                "round": int(m.group(1)),
                "depth": int(m.group(2)),
                "accepted": int(m.group(3)),
                "streak_in": int(kv.get("streak_in", -1)),
                "round_us": int(kv.get("round_us", 0)),
            }
        )
    return rounds


def summarise(group):
    if not group:
        return None
    emitted = sum(r["accepted"] + 1 for r in group)
    drafted = sum(r["depth"] for r in group)
    micros = sum(r["round_us"] for r in group)
    return {
        "rounds": len(group),
        "emitted_tokens": emitted,
        "drafted_tokens": drafted,
        "accepted_tokens": sum(r["accepted"] for r in group),
        "rejected_tokens": drafted - sum(r["accepted"] for r in group),
        "accept_rate": round(sum(r["accepted"] for r in group) / max(1, drafted), 4),
        "reject_round_share": round(
            sum(1 for r in group if r["accepted"] < r["depth"]) / len(group), 4),
        "mean_depth": round(sum(r["depth"] for r in group) / len(group), 3),
        "depth_histogram": dict(sorted(Counter(r["depth"] for r in group).items())),
        "accepted_tokens_per_round": round(emitted / len(group), 3),
        "rounds_per_token": round(len(group) / emitted, 5),
        "mean_round_us": round(micros / len(group), 1),
        "us_per_token": round(micros / emitted, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--split-index", type=int, default=301)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    rounds = parse(args.trace)
    decode_index = 0
    early, late = [], []
    for r in rounds:
        (early if decode_index < args.split_index else late).append(r)
        decode_index += r["accepted"] + 1

    report = {
        "label": args.label,
        "trace": args.trace,
        "split_decode_index": args.split_index,
        "whole_window": summarise(rounds),
        "before_split": summarise(early),
        "after_split": summarise(late),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
