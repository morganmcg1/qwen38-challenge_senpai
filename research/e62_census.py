#!/usr/bin/env python3
"""Reduce E62 census legs to dispatches per command-buffer commit, and test the
advisor's granularity model against them.

The model under test is

    dispatches_per_commit = min(OPS / ops_per_dispatch, MB / mb_per_dispatch)

with `ops_per_dispatch = 1.8369` and `mb_per_dispatch = 10.58` from the E60
census fit. The two constants are refitted here from the legs that are clearly
op-capped and byte-capped respectively, so the reported model is this session's,
not an imported one.

  research/e62_census.py --out research/e62-artifacts/e62-census.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

OUT_ROOT = pathlib.Path("research/out")
TAG = re.compile(r"^e62-census-mb(\d+)-ops(\d+)$")

# Only rounds count: `gap` records carry warmup, the seed prefill and whatever
# MLX encoded between rounds, none of which is the decode geometry under test.
ROUND_EVENT = "round"


def read_meta(leg: pathlib.Path) -> dict:
    meta: dict[str, str] = {}
    path = leg / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                meta[key.strip()] = value.strip()
    return meta


def reduce_leg(leg: pathlib.Path) -> dict | None:
    census = leg / "census.jsonl"
    if not census.exists() or census.stat().st_size == 0:
        return None
    meta = read_meta(leg)
    rounds = 0
    dispatches = 0
    commits = 0
    barriers = 0
    by_phase: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"dispatches": 0, "commits": 0}
    )
    gap_dispatches = 0
    gap_commits = 0
    for line in census.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        phases = record.get("phases", {})
        if record.get("event") == ROUND_EVENT:
            rounds += 1
            for name, counters in phases.items():
                dispatches += counters["dispatches"]
                commits += counters["commits"]
                barriers += counters["barriers"]
                by_phase[name]["dispatches"] += counters["dispatches"]
                by_phase[name]["commits"] += counters["commits"]
        else:
            for counters in phases.values():
                gap_dispatches += counters["dispatches"]
                gap_commits += counters["commits"]
    if rounds == 0 or commits == 0:
        return None
    return {
        "tag": leg.name,
        "mb": int(meta.get("mlx_max_mb_per_buffer", 0) or 0),
        "ops": int(meta.get("mlx_max_ops_per_buffer", 0) or 0),
        "worker_env_proof": (leg / "worker-env.txt").read_text().split()
        if (leg / "worker-env.txt").exists() else [],
        "rounds": rounds,
        "dispatches": dispatches,
        "commits": commits,
        "barriers": barriers,
        "dispatches_per_round": dispatches / rounds,
        "commits_per_round": commits / rounds,
        "dispatches_per_commit": dispatches / commits,
        "gap_dispatches": gap_dispatches,
        "gap_commits": gap_commits,
        "phases": {
            name: {
                **counters,
                "dispatches_per_commit": (
                    counters["dispatches"] / counters["commits"]
                    if counters["commits"] else None
                ),
            }
            for name, counters in sorted(by_phase.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--ops-per-dispatch", type=float, default=1.8369)
    parser.add_argument("--mb-per-dispatch", type=float, default=10.58)
    args = parser.parse_args()

    legs = []
    for directory in sorted(OUT_ROOT.glob("e62-census-*")):
        if not TAG.match(directory.name):
            continue
        record = reduce_leg(directory)
        if record is None:
            print(f"skip {directory.name}: no usable census records")
            continue
        legs.append(record)
    if not legs:
        raise SystemExit("e62: no census legs found")

    # Refit the two granularity constants from this session. An op-capped leg
    # fixes ops_per_dispatch; a byte-capped leg fixes mb_per_dispatch. A leg is
    # taken as op-capped when its measured granularity is well below what the
    # byte budget alone would allow under the imported byte constant.
    op_capped = [
        leg for leg in legs
        if leg["dispatches_per_commit"] < 0.8 * leg["mb"] / args.mb_per_dispatch
    ]
    fitted_ops_per_dispatch = (
        sum(leg["ops"] / leg["dispatches_per_commit"] for leg in op_capped)
        / len(op_capped)
    ) if op_capped else None

    for leg in legs:
        predicted = min(
            leg["ops"] / args.ops_per_dispatch,
            leg["mb"] / args.mb_per_dispatch,
        )
        leg["model_e60_predicted_dispatches_per_commit"] = predicted
        leg["model_e60_error_percent"] = (
            100.0 * (leg["dispatches_per_commit"] - predicted) / predicted
        )
        leg["binding_cap"] = (
            "ops"
            if leg["ops"] / args.ops_per_dispatch
            <= leg["mb"] / args.mb_per_dispatch
            else "mb"
        )
        if fitted_ops_per_dispatch:
            refit = min(
                leg["ops"] / fitted_ops_per_dispatch,
                leg["mb"] / args.mb_per_dispatch,
            )
            leg["model_refit_predicted_dispatches_per_commit"] = refit
            leg["model_refit_error_percent"] = (
                100.0 * (leg["dispatches_per_commit"] - refit) / refit
            )

    payload = {
        "harness": "local",
        "instrument": "E58 in-process dispatch census, restored from d0b337d^",
        "timed": False,
        "model_e60": {
            "ops_per_dispatch": args.ops_per_dispatch,
            "mb_per_dispatch": args.mb_per_dispatch,
        },
        "session_refit_ops_per_dispatch": fitted_ops_per_dispatch,
        "legs": legs,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    header = f"{'mb':>6} {'ops':>5} {'d/commit':>9} {'model':>8} {'err%':>8} {'binds':>6} {'d/round':>9}"
    print(header)
    for leg in sorted(legs, key=lambda item: (item["mb"], item["ops"])):
        print(
            f"{leg['mb']:>6} {leg['ops']:>5} {leg['dispatches_per_commit']:>9.3f} "
            f"{leg['model_e60_predicted_dispatches_per_commit']:>8.3f} "
            f"{leg['model_e60_error_percent']:>8.2f} {leg['binding_cap']:>6} "
            f"{leg['dispatches_per_round']:>9.1f}"
        )
    if fitted_ops_per_dispatch:
        print(f"session refit ops_per_dispatch = {fitted_ops_per_dispatch:.4f} "
              f"(E60 fit {args.ops_per_dispatch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
