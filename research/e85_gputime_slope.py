#!/usr/bin/env python3
"""Per-draft GPU, driver and encode time for one E85 census arm.

    usage: research/e85_gputime_slope.py --leg DRAFTS CENSUS_JSONL
                                         [--leg DRAFTS CENSUS_JSONL ...]
                                         [--phase draft_head] [--skip-rounds N]
                                         [--json OUT]

`gputime` records are deltas since the previous snapshot, so the totals for a
leg are the sum of every snapshot whose rounds are past warmup. Two legs at
different forced draft widths give the slope per draft token.

A census build serialises every dispatch behind the swizzle lock, so `encode_ns`
and `driver_ns` are inflated and are only comparable between two legs of the
same census binary. `gpu_ns` comes from Metal's own command-buffer timestamps
and is not inflated by host-side instrumentation.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

FIELDS = ("gpu_ns", "driver_ns", "encode_ns", "dispatches", "buffers")


def load(path: pathlib.Path, phase: str, skip_rounds: int) -> dict:
    """Sum the post-warmup snapshots of the worker that runs the head phase."""
    per_pid: dict[int, dict] = collections.defaultdict(
        lambda: {"rounds": 0, "totals": collections.Counter()})
    has_phase: set[int] = set()

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") != "gputime":
            continue
        pid = rec["pid"]
        if any(key.endswith(f"|{phase}") for key in rec.get("by_width_phase", {})):
            has_phase.add(pid)
        if rec.get("round_first") is None or rec["round_first"] < skip_rounds:
            continue
        entry = per_pid[pid]
        entry["rounds"] += rec.get("rounds", 0)
        for key, bucket in rec.get("by_width_phase", {}).items():
            if not key.endswith(f"|{phase}"):
                continue
            for field in FIELDS:
                entry["totals"][field] += bucket.get(field, 0)

    # the reference worker never runs the head phase; keep only the timed one
    candidates = {pid: v for pid, v in per_pid.items()
                  if pid in has_phase and v["rounds"] > 0}
    if not candidates:
        raise SystemExit(f"{path}: no post-warmup '{phase}' snapshots")
    pid, entry = max(candidates.items(), key=lambda kv: kv[1]["rounds"])
    rounds = entry["rounds"]
    out = {"path": str(path), "pid": pid, "rounds": rounds}
    for field in FIELDS:
        out[f"{field}_per_round"] = entry["totals"][field] / rounds
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", nargs=2, action="append", metavar=("DRAFTS", "PATH"),
                    required=True)
    ap.add_argument("--phase", default="draft_head")
    ap.add_argument("--skip-rounds", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    legs = []
    for drafts, path in args.leg:
        leg = load(pathlib.Path(path), args.phase, args.skip_rounds)
        leg["drafts"] = int(drafts)
        legs.append(leg)
        print(f"drafts={leg['drafts']:>2} pid={leg['pid']} rounds={leg['rounds']:>3} "
              f"{args.phase}: " + "  ".join(
                  f"{f}={leg[f + '_per_round']:.1f}" for f in FIELDS))

    report = {"phase": args.phase, "skip_rounds": args.skip_rounds, "legs": legs}
    if len(legs) >= 2:
        lo, hi = min(legs, key=lambda x: x["drafts"]), max(legs, key=lambda x: x["drafts"])
        span = hi["drafts"] - lo["drafts"]
        print(f"\nper-draft slope over forced width {lo['drafts']} -> {hi['drafts']}")
        for field in FIELDS:
            slope = (hi[f"{field}_per_round"] - lo[f"{field}_per_round"]) / span
            report[f"{field}_per_draft"] = slope
            unit = "us" if field.endswith("_ns") else ""
            shown = slope / 1000.0 if field.endswith("_ns") else slope
            print(f"  d{field}/ddraft = {shown:+12.3f} {unit}")

    print(json.dumps(report, indent=2) if False else "")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
