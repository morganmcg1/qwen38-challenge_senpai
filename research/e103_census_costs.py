#!/usr/bin/env python3
"""Reduce one or more E58 census legs to per-kernel exclusive GPU cost.

    usage: research/e103_census_costs.py TAG [TAG ...] [--json OUT] [--grep PAT]

Each leg is `research/out/TAG/census.jsonl`, produced by
`research/e96_census_leg.sh TAG DRAFTS TOKENS 0`, which pins
`MLX_E58_BUFFER_LIMIT_OPS=0` and `MLX_E58_BUFFER_LIMIT_MB=1` so that one
command buffer holds exactly one dispatch and a buffer interval is one
kernel's exclusive GPU time.

A census leg is never a timing leg: the census swizzle serialises every
dispatch, so host wall clock is invalid and only Metal's GPU clock counts.
The output here is exclusive per-dispatch GPU nanoseconds, dispatch counts
per round keyed by verify width and phase, and the round GPU-busy total that
the campaign uses as its local round anchor.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def load(tag: str) -> tuple[dict, dict, dict]:
    path = pathlib.Path("research/out") / tag / "census.jsonl"
    if not path.exists():
        sys.exit(f"e103_census_costs: no census at {path}")
    rounds_by_width: collections.Counter[str] = collections.Counter()
    kernels: dict[str, dict[str, float]] = {}
    phases: dict[str, dict[str, float]] = {}
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime":
            continue
        # `MLX_E80_SNAPSHOT_ROUNDS=1` makes every snapshot one round, and the
        # verify width of that round is carried in the `w<N>|phase` keys, not
        # in `window_width`, which stays 0 in a forced-depth leg.
        for width in {k.split("|", 1)[0] for k in rec.get("by_width_phase", {})}:
            rounds_by_width[width] += rec.get("rounds", 1)
        for key, v in rec.get("exclusive_kernels", {}).items():
            e = kernels.setdefault(
                key, {"buffers": 0, "gpu_ns": 0.0, "min_ns": float("inf"),
                      "max_ns": 0.0})
            e["buffers"] += v["buffers"]
            e["gpu_ns"] += v["gpu_ns"]
            e["min_ns"] = min(e["min_ns"], v["min_ns"])
            e["max_ns"] = max(e["max_ns"], v["max_ns"])
        for key, v in rec.get("by_width_phase", {}).items():
            e = phases.setdefault(key, {"dispatches": 0, "gpu_ns": 0.0,
                                        "buffers": 0})
            e["dispatches"] += v.get("dispatches", 0)
            e["buffers"] += v.get("buffers", 0)
            e["gpu_ns"] += v.get("gpu_ns", 0)
    return dict(rounds_by_width), kernels, phases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--grep", default="sdpa")
    ap.add_argument("--min-us-per-round", type=float, default=0.0)
    args = ap.parse_args()

    out: dict[str, object] = {}
    for tag in args.tags:
        rounds, kernels, phases = load(tag)
        print(f"=== {tag} ===")
        print("rounds by width:", dict(sorted(rounds.items())))

        # Round GPU busy per width, the campaign's local round anchor.
        busy: dict[str, float] = collections.defaultdict(float)
        for key, v in phases.items():
            width = key.split("|", 1)[0]
            busy[width] += v["gpu_ns"]
        print("round GPU busy, us/round, by width and phase:")
        for key in sorted(phases):
            width = key.split("|", 1)[0]
            r = rounds.get(width, 0)
            if not r:
                continue
            us = phases[key]["gpu_ns"] / 1e3 / r
            if us < 1.0:
                continue
            print(f"  {key:<28} {us:12.1f} us/round  "
                  f"{phases[key]['dispatches'] / r:8.2f} dispatches/round")
        for width in sorted(busy):
            r = rounds.get(width, 0)
            if r:
                print(f"  TOTAL {width:<22} {busy[width] / 1e3 / r:12.1f} "
                      f"us/round over {r} rounds")

        print(f"kernels matching {args.grep!r}:")
        rows = []
        for key, v in kernels.items():
            if args.grep not in key:
                continue
            width = key.split("|", 1)[0]
            r = rounds.get(width, 0)
            if not r:
                continue
            us_disp = v["gpu_ns"] / 1e3 / v["buffers"]
            us_round = v["gpu_ns"] / 1e3 / r
            if us_round < args.min_us_per_round:
                continue
            rows.append((us_round, key, v["buffers"], v["buffers"] / r,
                         us_disp, v["min_ns"] / 1e3, v["max_ns"] / 1e3))
        rows.sort(reverse=True)
        for us_round, key, n, per_round, us_disp, lo, hi in rows:
            print(f"  {key}")
            print(f"      dispatches={n:6d}  per_round={per_round:6.2f}  "
                  f"us/dispatch={us_disp:8.2f}  us/round={us_round:9.1f}  "
                  f"min={lo:7.2f}  max={hi:7.2f}")
        out[tag] = {
            "rounds_by_width": rounds,
            "round_busy_us": {w: busy[w] / 1e3 / rounds[w] for w in busy
                              if rounds.get(w)},
            "kernels": {
                key: {
                    "dispatches": v["buffers"],
                    "dispatches_per_round": v["buffers"]
                    / rounds[key.split("|", 1)[0]],
                    "us_per_dispatch": v["gpu_ns"] / 1e3 / v["buffers"],
                    "us_per_round": v["gpu_ns"] / 1e3
                    / rounds[key.split("|", 1)[0]],
                    "min_us": v["min_ns"] / 1e3,
                    "max_us": v["max_ns"] / 1e3,
                }
                for key, v in kernels.items()
                if rounds.get(key.split("|", 1)[0])
            },
        }
        print()

    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
