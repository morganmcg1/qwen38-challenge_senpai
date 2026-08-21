#!/usr/bin/env python3
"""E109 rung 1: turn one threadgroup-shape sweep into a named mechanism.

    usage: research/e109_shape_verdict.py --timing T --census C --spec S
                                          [--cores N] [--out OUT]

Each arm folds the same total work into `units` times fewer threadgroups, so
the sweep is indexed by the fold factor u = threadgroups(u=1) / threadgroups.
The three candidate mechanisms predict different curves, and the discriminating
quantity is the LOCAL log-log slope

    s_j = log( t_{j+1} / t_j ) / log( u_{j+1} / u_j )

    H3 per-threadgroup granularity   every s_j is about -1: the cost is paid
                                     once per threadgroup, so halving the
                                     threadgroups halves the time
    H2 dependent chain               every s_j is about 0: the critical path
                                     lives inside one thread and widening the
                                     threadgroup cannot shorten it
    H1 occupancy                     s_j starts clearly negative and returns
                                     to about 0: folding helps until the
                                     machine is full, then stops helping

A single global slope cannot separate H1 from a weak H3, which is why the
verdict reads the sequence rather than one fitted exponent.

THE STOP RULE IS APPLIED HERE, NOT ARGUED LATER. The assignment fixed it in
advance: the sweep is negative unless some arm beats the shipped arm by more
than 15 %. `verdict.actionable` reports that decision, and it is independent of
which mechanism the curve names -- naming H1 does not make a 3 % win worth
shipping.

`actionable` also requires a positive control that mismatched in the same
session. Arms that claim bit exactness are cleared by a byte comparison, and a
comparison that has never failed is not evidence. A control arm is excluded
from the curve, from the lever list, and from the best-arm search.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

FLAT_SLOPE = 0.15      # |s| below this counts as flat
STRONG_SLOPE = 0.70    # |s| above this counts as inverse-proportional
SATURATED_SLOPE = 0.20 # |s| below this counts as saturated at the wide end
MIN_USEFUL_GAIN = 0.15 # the assignment's 15 % promotion floor for rung 1


def read_census(path: pathlib.Path) -> dict[str, dict[str, dict]]:
    """`agx_crossarch.py census` prints one `<arch> <json>` line per arch."""
    out: dict[str, dict[str, dict]] = {}
    for line in path.read_text().splitlines():
        arch, _, payload = line.partition(" ")
        if not payload.strip():
            continue
        out[arch] = json.loads(payload)
    return out


def slopes(points: list[tuple[float, float]]) -> list[float]:
    return [
        math.log(points[j + 1][1] / points[j][1])
        / math.log(points[j + 1][0] / points[j][0])
        for j in range(len(points) - 1)
    ]


def classify(seq: list[float]) -> tuple[str, str]:
    if not seq:
        return "inconclusive", "the sweep has fewer than two arms"
    worst = max(abs(s) for s in seq)
    if worst < FLAT_SLOPE:
        return "H2_dependent_chain", (
            f"every local slope is within {FLAT_SLOPE} of zero (worst"
            f" {worst:.2f}), so folding threadgroups does not shorten the"
            " kernel: the critical path is inside one thread")
    if all(s < -STRONG_SLOPE for s in seq):
        return "H3_per_threadgroup_granularity", (
            f"every local slope is below -{STRONG_SLOPE} (max {max(seq):.2f}),"
            " so time tracks 1/threadgroups: the cost is paid once per"
            " threadgroup")
    if seq[0] < -FLAT_SLOPE and abs(seq[-1]) < SATURATED_SLOPE:
        return "H1_occupancy", (
            f"the slope starts at {seq[0]:.2f} and saturates at"
            f" {seq[-1]:.2f}: folding helps until the machine is full, then"
            " stops helping")
    if all(s > 0 for s in seq):
        return "anti_folding", (
            "time RISES with wider threadgroups, so the shipped narrow shape"
            " is already the better one for this kernel")
    return "mixed", (
        "the slope sequence matches no single mechanism: "
        + ", ".join(f"{s:.2f}" for s in seq))


def build(timing: dict, census: dict, spec: dict, cores: int) -> dict:
    by_name = {a["name"]: a for a in timing["arms"]}
    spec_by_name = {a["name"]: a for a in spec["arms"]}
    base_tgs = max(a["threadgroups"] for a in timing["arms"])

    arms = []
    for arm in timing["arms"]:
        fn = arm["function"]
        isa = {
            arch: {
                "registers": rec.get(fn, {}).get("registers"),
                "spill_bytes": rec.get(fn, {}).get("spill_bytes"),
                "text_bytes": rec.get(fn, {}).get("text_bytes"),
                "text_sha8": rec.get(fn, {}).get("text_sha8"),
            }
            for arch, rec in census.items()
        }
        entry = dict(arm)
        entry["fold_factor"] = base_tgs / arm["threadgroups"]
        # How many rounds of threadgroups the machine must take if each core
        # holds one at a time. H1 predicts the time curve follows this.
        entry["waves_over_cores"] = math.ceil(arm["threadgroups"] / cores)
        entry["isa"] = isa
        entry["source_grid"] = spec_by_name[arm["name"]]["grid"]
        entry["source_threadgroup"] = spec_by_name[arm["name"]]["threadgroup"]
        entry["variant"] = spec_by_name[arm["name"]].get("reduction", "shipped")
        arms.append(entry)
    arms.sort(key=lambda a: (a["variant"] != "shipped", a["fold_factor"]))

    # A control arm is deliberately wrong, so it may never win the sweep, sit
    # on the curve, or be offered as a lever. It answers one question only:
    # can the byte comparison that clears every other arm actually fail?
    controls = [a for a in arms if a["variant"] == "positive_control"]
    candidates = [a for a in arms if a["variant"] != "positive_control"]

    shipped = next((a for a in candidates if a["shipped"]), candidates[0])
    best = min(candidates, key=lambda a: a["us_per_dispatch_median"])
    gain = 1.0 - best["us_per_dispatch_median"] / shipped["us_per_dispatch_median"]

    # Only the shape ladder can be read as a curve. A lever arm changes the
    # kernel body at a fold factor the ladder already occupies, so including it
    # would put two times at one abscissa.
    sweep = [a for a in candidates if a["variant"] == "shipped"]
    levers = [a for a in candidates if a["variant"] != "shipped"]
    seq = slopes([(a["fold_factor"], a["us_per_dispatch_median"]) for a in sweep])
    mechanism, why = classify(seq)

    exact = all(a["exact_vs_arm0"] for a in candidates)
    # An exactness claim counts only when a control mismatched in the same
    # session. With no control the byte comparison is untested, so `exact` is
    # an unproven assertion and cannot license promotion.
    control_fired = bool(controls) and all(
        not a["exact_vs_arm0"] and a.get("mismatch_bytes", 0) > 0
        for a in controls)
    return {
        "family": timing["family"],
        "harness": "local",
        "device": timing["device"],
        "live_kernel": spec["live_kernel"],
        "dispatch": timing["dispatch"],
        "gpu_cores_assumed": cores,
        "reps": timing["reps"],
        "inner": timing["inner"],
        "arms": arms,
        "log_log_slopes": seq,
        "levers": [
            {
                "name": a["name"],
                "variant": a["variant"],
                "us_per_dispatch_median": a["us_per_dispatch_median"],
                "gain_vs_shipped": 1.0 - a["us_per_dispatch_median"]
                / shipped["us_per_dispatch_median"],
                "exact_vs_arm0": a["exact_vs_arm0"],
            }
            for a in levers
        ],
        "positive_controls": [
            {
                "name": a["name"],
                "perturbation": spec_by_name[a["name"]]["perturb"],
                "mismatched": not a["exact_vs_arm0"],
                "mismatch_bytes": a.get("mismatch_bytes"),
                "output_bytes": a.get("output_bytes"),
                "mismatch_by_buffer": {
                    k: v for k, v in a.get("mismatch_by_buffer", {}).items()
                    if v
                },
            }
            for a in controls
        ],
        "verdict": {
            "mechanism": mechanism,
            "reason": why,
            "shipped_arm": shipped["name"],
            "shipped_us_per_dispatch": shipped["us_per_dispatch_median"],
            "best_arm": best["name"],
            "best_us_per_dispatch": best["us_per_dispatch_median"],
            "fractional_gain_vs_shipped": gain,
            "min_useful_gain": MIN_USEFUL_GAIN,
            "actionable": bool(
                gain > MIN_USEFUL_GAIN and exact and control_fired),
            "all_folds_bit_exact": exact,
            "exactness_check_proven_by_control": control_fired,
        },
    }


def render(out: dict) -> str:
    lines = [
        f"E109 rung 1 -- {out['family']} ({out['live_kernel']})",
        f"  {out['device']}   {out['dispatch']}   reps {out['reps']}"
        f" x inner {out['inner']}   cores assumed {out['gpu_cores_assumed']}",
        "",
        f"{'arm':<8} {'fold':>5} {'tgs':>5} {'simd':>5} {'thr':>5} {'waves':>6}"
        f" {'us/disp':>9} {'sd':>7} {'g16s reg':>9} {'g16s spill':>11}"
        f" {'g17s reg':>9} {'g17s spill':>11} {'exact':>6}",
    ]
    for a in out["arms"]:
        g16 = a["isa"].get("applegpu_g16s", {})
        g17 = a["isa"].get("applegpu_g17s", {})
        lines.append(
            f"{a['name']:<8} {a['fold_factor']:>5.0f} {a['threadgroups']:>5}"
            f" {a['simdgroups_per_threadgroup']:>5}"
            f" {a['threads_per_threadgroup']:>5} {a['waves_over_cores']:>6}"
            f" {a['us_per_dispatch_median']:>9.3f} {a['us_per_dispatch_sd']:>7.3f}"
            f" {str(g16.get('registers')):>9} {str(g16.get('spill_bytes')):>11}"
            f" {str(g17.get('registers')):>9} {str(g17.get('spill_bytes')):>11}"
            f" {'yes' if a['exact_vs_arm0'] else 'NO':>6}"
        )
    v = out["verdict"]
    lines += [
        "",
        "  local log-log slopes " + ", ".join(f"{s:.2f}" for s in out["log_log_slopes"]),
        f"  MECHANISM {v['mechanism']}",
        f"    {v['reason']}",
        f"  best {v['best_arm']} {v['best_us_per_dispatch']:.3f} us vs shipped"
        f" {v['shipped_arm']} {v['shipped_us_per_dispatch']:.3f} us"
        f"   gain {100 * v['fractional_gain_vs_shipped']:+.1f} %"
        f" (floor {100 * v['min_useful_gain']:.0f} %)",
        f"  bit-exact folds {v['all_folds_bit_exact']}"
        f"   proven by control {v['exactness_check_proven_by_control']}"
        f"   ACTIONABLE {v['actionable']}",
    ]
    for lever in out["levers"]:
        lines.append(
            f"  lever {lever['name']} ({lever['variant']})"
            f" {lever['us_per_dispatch_median']:.3f} us"
            f"   gain {100 * lever['gain_vs_shipped']:+.1f} %"
            f"   bit-exact {lever['exact_vs_arm0']}")
    if not out["positive_controls"]:
        lines.append("  positive control ABSENT -- no exactness claim is proven")
    for c in out["positive_controls"]:
        p = c["perturbation"]
        moved = ", ".join(f"{k} {v}" for k, v in c["mismatch_by_buffer"].items())
        lines.append(
            f"  control {c['name']} flips {p['buffer']}[{p['element']}] by"
            f" 1 ULP -> mismatched {c['mismatched']}"
            f" ({c['mismatch_bytes']} of {c['output_bytes']} output bytes)"
            f"   moved [{moved}]")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", required=True, type=pathlib.Path)
    parser.add_argument("--census", required=True, type=pathlib.Path)
    parser.add_argument("--spec", required=True, type=pathlib.Path)
    parser.add_argument("--cores", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    out = build(
        json.loads(args.timing.read_text()),
        read_census(args.census),
        json.loads(args.spec.read_text()),
        args.cores,
    )
    print(render(out))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
