#!/usr/bin/env python3
"""E64 rung 0b: turn the palindrome legs into one preregistered decision.

Estimator, fixed before the run:

  Each rep runs the six legs `plain forced ballast ballast forced plain`, so
  every arm has mean leg position 2.5 and a monotone drift inside a rep cancels
  to first order. The per-rep arm estimate is the mean of that arm's two legs.
  The effect is the median over reps of `candidate / plain - 1`.

Null bar, the bar the effect must clear:

  The SAME estimator applied to a contrast that must be zero. `plain` is split
  by leg position into a pseudo-A arm (position 0) and a pseudo-B arm
  (position 5), and the identical median-of-ratios is computed. This is a null
  contrast with the same estimator, the same session and the maximum leg
  separation inside a rep, so it carries the drift the real contrast could
  carry. The widest same-arm leg spread is reported next to it, because the
  local null floor has not been shown to scale with leg separation.

Preregistered decision on the forced arm (assignment, rung 0b):

  >= 10 %          mechanism confirmed with power; go to rung 1
  4 % to 10 %      partial support; carry the reduced ceiling into rung 1
  < 4 %, or inside the widest same-arm spread: the hypothesis is dead. Stop.

  python3 research/e64_analyze.py research/e64-artifacts/rung0b-timing-na5.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as stats
import sys

CONFIRM = 0.10
PARTIAL = 0.04


def per_rep(legs: list[dict], key: str = "seconds_per_dispatch") -> dict:
    """rep -> arm -> [leg values in position order]."""
    reps: dict[int, dict[str, list[float]]] = {}
    for leg in legs:
        reps.setdefault(leg["rep"], {}).setdefault(leg["arm"], []).append(
            (leg["position"], leg[key]))
    return {
        rep: {arm: [value for _, value in sorted(values)]
              for arm, values in arms.items()}
        for rep, arms in reps.items()
    }


def ratios(reps: dict, candidate: str, reference: str) -> list[float]:
    out = []
    for arms in reps.values():
        if candidate not in arms or reference not in arms:
            continue
        c = stats.fmean(arms[candidate])
        r = stats.fmean(arms[reference])
        out.append(c / r - 1.0)
    return out


def position_split_null(reps: dict, arm: str) -> list[float]:
    """The same estimator on a contrast that must be zero: plain vs plain."""
    out = []
    for arms in reps.values():
        legs = arms.get(arm, [])
        if len(legs) < 2:
            continue
        out.append(legs[-1] / legs[0] - 1.0)
    return out


def widest_same_arm_spread(reps: dict, arm: str) -> float:
    values = [value for arms in reps.values() for value in arms.get(arm, [])]
    if not values:
        return float("nan")
    return (max(values) - min(values)) / stats.median(values)


def widest_mirrored_spread(reps: dict, arm: str) -> float:
    worst = 0.0
    for arms in reps.values():
        legs = arms.get(arm, [])
        if len(legs) < 2:
            continue
        worst = max(worst, abs(legs[-1] - legs[0]) / stats.fmean(legs))
    return worst


def verdict(effect: float, null_bar: float) -> str:
    magnitude = abs(effect)
    if magnitude < PARTIAL or magnitude <= null_bar:
        return "dead"
    if magnitude >= CONFIRM:
        return "confirmed"
    return "partial"


def analyze_shape(shape: dict) -> dict:
    reps = per_rep(shape["legs"])
    arms = sorted({leg["arm"] for leg in shape["legs"]})
    medians = {
        arm: stats.median([value for r in reps.values()
                           for value in r.get(arm, [])])
        for arm in arms
    }
    null = position_split_null(reps, "plain")
    null_bar_estimator = abs(stats.median(null)) if null else float("nan")
    spread_bar = max(widest_same_arm_spread(reps, arm) for arm in arms)
    mirrored_bar = max(widest_mirrored_spread(reps, arm) for arm in arms)

    out = {
        "shape": shape["shape"],
        "k": shape["k"],
        "n": shape["n"],
        "inner": shape["inner"],
        "reps": len(reps),
        "entry_gpu_temp_c": shape["entry_gpu_temp_c"],
        "exit_gpu_temp_c": shape["exit_gpu_temp_c"],
        "parity_differing_vs_plain": shape["parity_differing_vs_plain"],
        "median_seconds_per_dispatch": medians,
        "gb_per_s": {arm: shape["bytes_per_dispatch"] / value / 1e9
                     for arm, value in medians.items()},
        "null_position_split_median": null_bar_estimator,
        "null_position_split_max": max((abs(v) for v in null), default=None),
        "widest_same_arm_spread": spread_bar,
        "widest_mirrored_leg_spread": mirrored_bar,
        "effects": {},
    }
    for arm in arms:
        if arm == "plain":
            continue
        values = ratios(reps, arm, "plain")
        positive = sum(1 for v in values if v > 0)
        out["effects"][arm] = {
            "median": stats.median(values),
            "mean": stats.fmean(values),
            "min": min(values),
            "max": max(values),
            "sign_stable_fraction": max(positive, len(values) - positive)
                                    / len(values),
            "reps": len(values),
            "verdict_vs_spread_bar": verdict(stats.median(values), spread_bar),
            "verdict_vs_null_estimator":
                verdict(stats.median(values), null_bar_estimator),
        }
    return out


# askeladd's E61 rung 1 single-stream ladder: the only prior measurement of the
# step this experiment exists to explain.
ASKELADD_LADDER_GB_S = {2: 223.784, 3: 199.693, 4: 175.238, 5: 150.946,
                        6: 117.8, 7: 97.9}


def ladder(sessions: list[dict]) -> dict:
    rungs = []
    for session in sessions:
        for shape in session["shapes"]:
            rungs.append({
                "na": session["na"],
                "shape": shape["shape"],
                "ms": {arm: value * 1e3 for arm, value
                       in shape["median_seconds_per_dispatch"].items()},
                "gb_per_s": dict(shape["gb_per_s"]),
                "widest_same_arm_spread": shape["widest_same_arm_spread"],
                "plain_gb_per_s": shape["gb_per_s"]["plain"],
                "plain_ms": shape["median_seconds_per_dispatch"]["plain"] * 1e3,
                "entry_gpu_temp_c": session["entry_gpu_temp_c"],
                "exit_gpu_temp_c": session["exit_gpu_temp_c"],
                "reference_gb_per_s": ASKELADD_LADDER_GB_S.get(session["na"]),
            })
    rungs.sort(key=lambda r: (r["shape"], r["na"]))

    steps = []
    by_shape = {}
    for rung in rungs:
        by_shape.setdefault(rung["shape"], {})[rung["na"]] = rung
    for shape, at_na in by_shape.items():
        for na in sorted(at_na)[:-1]:
            if na + 1 not in at_na:
                continue
            lo, hi = at_na[na], at_na[na + 1]
            step = {
                "shape": shape,
                "from_na": na,
                "to_na": na + 1,
                "seconds_step": hi["plain_ms"] / lo["plain_ms"] - 1.0,
                "seconds_step_by_arm": {
                    arm: hi["ms"][arm] / lo["ms"][arm] - 1.0
                    for arm in lo["ms"] if arm in hi["ms"]},
                # The bar a step must clear is the two rungs' same-arm spreads
                # compounded: each rung's median carries its own session noise.
                "null_bar": ((1.0 + lo["widest_same_arm_spread"])
                             * (1.0 + hi["widest_same_arm_spread"]) - 1.0),
            }
            if lo["reference_gb_per_s"] and hi["reference_gb_per_s"]:
                step["reference_seconds_step"] = (
                    lo["reference_gb_per_s"] / hi["reference_gb_per_s"] - 1.0)
            steps.append(step)
    return {"rungs": rungs, "steps": steps}


def print_ladder(data: dict) -> None:
    print("\nladder over NA (instrument check)")
    for rung in data["rungs"]:
        reference = rung["reference_gb_per_s"]
        arms = "  ".join(
            f"{arm} {rung['ms'][arm]:7.4f} ms {rung['gb_per_s'][arm]:6.1f} GB/s"
            for arm in sorted(rung["ms"]))
        print(f"  {rung['shape']:24s} NA={rung['na']}  {arms}"
              f"  reference {reference if reference else float('nan'):6.1f} GB/s"
              f"  {rung['entry_gpu_temp_c']:.1f}C -> {rung['exit_gpu_temp_c']:.1f}C")
    print("  step in seconds per dispatch")
    for step in data["steps"]:
        reference = step.get("reference_seconds_step")
        text = f"{reference * 100:+.1f} %" if reference is not None else "n/a"
        arms = "  ".join(f"{arm} {value * 100:+7.2f} %" for arm, value
                         in sorted(step["seconds_step_by_arm"].items()))
        print(f"    {step['shape']:24s} NA {step['from_na']}->{step['to_na']}  "
              f"{arms}   bar {step['null_bar'] * 100:.2f} %"
              f"   reference {text}")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("timing", type=pathlib.Path, nargs="+")
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--ladder", action="store_true",
                        help="also report the plain-arm bandwidth ladder over "
                             "NA against askeladd's E61 reference")
    args = parser.parse_args()

    report = {"sessions": []}
    for path in args.timing:
        data = json.loads(path.read_text())
        session = {
            "path": str(path),
            "na": data["na"],
            "device": data["device"],
            "reps": data["reps"],
            "warmup_reps_discarded": data["warmup_reps_discarded"],
            "order": data["order"],
            "entry_gpu_temp_c": data["entry_gpu_temp_c"],
            "exit_gpu_temp_c": data["exit_gpu_temp_c"],
            "shapes": [analyze_shape(shape) for shape in data["shapes"]],
        }
        for arm in ("forced", "ballast"):
            effects = [s["effects"][arm]["median"] for s in session["shapes"]
                       if arm in s["effects"]]
            if effects:
                session[f"{arm}_effect_median_over_shapes"] = stats.median(effects)
        bars = [s["widest_same_arm_spread"] for s in session["shapes"]]
        session["widest_same_arm_spread_over_shapes"] = max(bars)
        session["decision_forced"] = verdict(
            session.get("forced_effect_median_over_shapes", 0.0),
            max(bars))
        report["sessions"].append(session)

    for session in report["sessions"]:
        print(f"NA={session['na']}  {session['device']}  "
              f"reps={session['reps']} (+{session['warmup_reps_discarded']} "
              f"discarded)  {session['entry_gpu_temp_c']:.1f}C -> "
              f"{session['exit_gpu_temp_c']:.1f}C")
        for shape in session["shapes"]:
            print(f"  {shape['shape']:34s} k={shape['k']:5d} n={shape['n']:6d} "
                  f"inner={shape['inner']:4d}")
            for arm, value in shape["median_seconds_per_dispatch"].items():
                print(f"    {arm:8s} {value * 1e3:8.4f} ms  "
                      f"{shape['gb_per_s'][arm]:6.1f} GB/s")
            for arm, effect in shape["effects"].items():
                print(f"    {arm:8s} effect {effect['median'] * 100:+7.3f} %  "
                      f"sign stable {effect['sign_stable_fraction'] * 100:5.1f} %  "
                      f"{effect['verdict_vs_spread_bar']}")
            print(f"    null: position-split {shape['null_position_split_median'] * 100:+.3f} %  "
                  f"widest same-arm spread {shape['widest_same_arm_spread'] * 100:.3f} %  "
                  f"widest mirrored {shape['widest_mirrored_leg_spread'] * 100:.3f} %")
            print(f"    parity vs plain: {shape['parity_differing_vs_plain']}")
        print(f"  forced over shapes: "
              f"{session.get('forced_effect_median_over_shapes', float('nan')) * 100:+.3f} %  "
              f"bar {session['widest_same_arm_spread_over_shapes'] * 100:.3f} %  "
              f"-> {session['decision_forced']}")

    if args.ladder:
        report["ladder"] = ladder(report["sessions"])
        print_ladder(report["ladder"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
