#!/usr/bin/env python3
"""Rung 1: extend the E54 lone-group bandwidth ladder to NA = 6 and NA = 7.

E54 measured what one working group achieves as a function of its row count NA:
223.784 / 199.693 / 175.238 / 150.946 GB/s at NA = 2, 3, 4, 5, three steps that
agree to 0.36 GB/s. The shipped table has no lone group above NA = 4, and E54
had to add an isolated IPG=5 arm to reach NA = 5. E61 does the same thing two
steps further: `t6` makes `case 6` a lone NA=6 group and `t7` makes `case 7` a
lone NA=7 group, so both rates become directly measurable.

The ladder decides M = 6 through M = 9 at once. Each width M has a break-even
rate: the rate a single NA=M group must sustain for the halved weight traffic
to repay the lost group parallelism. If bw(6) lands above 114.00 GB/s the
single-stream form pays at M=6; if the ladder stays linear it lands at 126.65
and M = 6, 7, 8 and 9 all clear their thresholds.

This script also reads the direct M=6 and M=7 cell times out of the same legs,
because a measured cell time is a better estimate of break-even than a model of
break-even. All thresholds and predictions come from research/e61-prereg.md and
were written down before any leg ran.

  python3 research/e61_bandwidth.py --out research/e61-artifacts/e61-bandwidth.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e54_bandwidth import bandwidth_rows  # noqa: E402
from e46_analyze import load  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
LEGS = REPO / ".mlxfast-private/e61-legs"

# research/e54-artifacts/e54-bandwidth.json, lone_group_curve.
E54_ANCHORS = {2: 223.784, 3: 199.693, 4: 175.238, 5: 150.946}
# Prereg: linear extrapolation of the mean E54 step past NA=5.
PREDICTED = {6: 126.65, 7: 102.35}
# Prereg: |measured - predicted| above this refutes linearity at that NA.
LINEARITY_TOLERANCE_GBPS = 5.0
# Prereg: the untreated widths re-measure the anchors inside every leg.
CONTROL_TOLERANCE_PCT = 2.5
# Assignment, carried verbatim: the rate a lone NA=M group must sustain.
BREAKEVEN = {6: 114.00, 7: 106.55, 8: 100.01, 9: 92.56}
# Prereg cell-delta predictions and acceptance bands.
CELL_PREDICTION_PCT = {("t6", 6): -9.95, ("t7", 7): +4.16}
UNTREATED_TOLERANCE_PCT = 1.0
GATE_CELL_DELTA_PCT = -2.0


def per_arm_widths(legs_dir: pathlib.Path) -> tuple[dict, float, dict]:
    """arm -> width -> weighted summary, plus the measured stream peak."""
    per_arm: dict[str, dict[str, list[dict]]] = {}
    peak = None
    provenance: dict[str, list[str]] = {}
    for path in sorted(glob.glob(str(legs_dir / "*-leg.json"))):
        leg = json.loads(pathlib.Path(path).read_text())
        tag, arm = leg["tag"], leg["arm"]
        try:
            data, _ = load(tag)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {tag}: {exc}", file=sys.stderr)
            continue
        peak = data["roofline"]["peak_bandwidth_bytes_per_second"]
        per_arm.setdefault(arm, {})[tag] = bandwidth_rows(data)
        provenance.setdefault(arm, []).append(tag)
    if not per_arm:
        raise SystemExit(f"no readable legs under {legs_dir}; run the timed legs first")

    summary: dict[str, dict[int, dict]] = {}
    for arm, legs in per_arm.items():
        by_m: dict[int, list[dict]] = {}
        legs_at_m: dict[int, set[str]] = {}
        for tag, rows in legs.items():
            for r in rows:
                if r["bandwidth_reliable"]:
                    by_m.setdefault(r["m"], []).append(r)
                    legs_at_m.setdefault(r["m"], set()).add(tag)
        summary[arm] = {}
        for m, rows in sorted(by_m.items()):
            n_legs = len(legs_at_m[m])
            wbytes = sum(r["calls_per_verify"] * r["aggregate_bytes"] for r in rows)
            wsecs = sum(r["calls_per_verify"] * r["seconds_per_call"] for r in rows)
            lone = [g for r in rows for g in r["per_group_gbps"] if r["lone_group"]]
            summary[arm][m] = {
                "kernels": sorted({r["kernel"] for r in rows}),
                "group_na": sorted({tuple(r["group_na"]) for r in rows}),
                "working_groups": sorted({r["working_groups"] for r in rows}),
                "legs": sorted(legs_at_m[m]),
                "weighted_bytes_per_verify": wbytes / n_legs,
                "weighted_seconds_per_verify": wsecs / n_legs,
                "weighted_gbps": wbytes / wsecs / 1e9,
                "weighted_frac_of_measured_peak": wbytes / wsecs / peak,
                "lone_group": all(r["lone_group"] for r in rows),
                "lone_group_gbps_mean": (sum(lone) / len(lone)) if lone else None,
                "samples": len(rows),
            }
    return summary, peak, provenance


def lone_group_ladder(summary: dict) -> dict:
    """Achieved rate of ONE working group against its row count, all arms."""
    obs: dict[int, dict[str, float]] = {}
    for arm, widths in summary.items():
        for m, s in widths.items():
            if s["working_groups"] == [1] and s["group_na"] == [(m,)]:
                obs.setdefault(m, {})[arm] = s["weighted_gbps"]

    rates = {na: sum(v.values()) / len(v) for na, v in obs.items()}
    rungs = {}
    for na in sorted(rates):
        entry = {
            "measured_gbps": rates[na],
            "per_arm_gbps": obs[na],
            "arm_spread_pct": (
                (max(obs[na].values()) - min(obs[na].values())) / rates[na] * 100.0
                if len(obs[na]) > 1 else 0.0),
            "e54_anchor_gbps": E54_ANCHORS.get(na),
            "predicted_gbps": PREDICTED.get(na),
        }
        if entry["e54_anchor_gbps"]:
            delta = rates[na] - entry["e54_anchor_gbps"]
            entry["anchor_delta_gbps"] = delta
            entry["anchor_delta_pct"] = delta / entry["e54_anchor_gbps"] * 100.0
            entry["positive_control_passed"] = (
                abs(entry["anchor_delta_pct"]) <= CONTROL_TOLERANCE_PCT)
        if entry["predicted_gbps"]:
            miss = rates[na] - entry["predicted_gbps"]
            entry["prediction_miss_gbps"] = miss
            entry["linearity_refuted"] = abs(miss) > LINEARITY_TOLERANCE_GBPS
        if na in BREAKEVEN:
            entry["breakeven_gbps"] = BREAKEVEN[na]
            entry["clears_breakeven"] = rates[na] > BREAKEVEN[na]
            entry["headroom_gbps"] = rates[na] - BREAKEVEN[na]
        rungs[na] = entry

    steps = {}
    for na in sorted(rates):
        if na - 1 in rates:
            steps[f"{na - 1}->{na}"] = rates[na] - rates[na - 1]
    return {
        "rungs": rungs,
        "measured_steps_gbps": steps,
        "e54_measured_steps_gbps": {"2->3": -24.091, "3->4": -24.455, "4->5": -24.292},
        "method": "one working group per dispatch; weighted by calls per verify",
    }


def cell_deltas(summary: dict, control: str = "shipped") -> dict:
    """Direct treated-minus-control cell time at every measured width."""
    if control not in summary:
        raise SystemExit(f"no {control} legs; the control arm is required")
    out = {}
    for arm, widths in summary.items():
        if arm == control:
            continue
        cells = {}
        for m, t in sorted(widths.items()):
            c = summary[control].get(m)
            if c is None:
                continue
            delta = ((t["weighted_seconds_per_verify"] - c["weighted_seconds_per_verify"])
                     / c["weighted_seconds_per_verify"] * 100.0)
            treated = t["group_na"] != c["group_na"]
            cells[m] = {
                "treated_width": treated,
                "control_group_na": c["group_na"],
                "treated_group_na": t["group_na"],
                "control_seconds_per_verify": c["weighted_seconds_per_verify"],
                "treated_seconds_per_verify": t["weighted_seconds_per_verify"],
                "seconds_delta_pct": delta,
                "traffic_ratio_treated_over_control":
                    t["weighted_bytes_per_verify"] / c["weighted_bytes_per_verify"],
                "predicted_delta_pct": CELL_PREDICTION_PCT.get((arm, m)),
            }
        untreated = [v["seconds_delta_pct"] for v in cells.values()
                     if not v["treated_width"]]
        out[arm] = {
            "cells": cells,
            # A systematic shift across every untreated width is the shared
            # register-ceiling term, measured without a ballast arm.
            "untreated_mean_delta_pct": (sum(untreated) / len(untreated)) if untreated else None,
            "untreated_max_abs_delta_pct": max((abs(x) for x in untreated), default=None),
            "untreated_widths": sorted(m for m, v in cells.items() if not v["treated_width"]),
            "untreated_within_band": (
                all(abs(x) <= UNTREATED_TOLERANCE_PCT for x in untreated)
                if untreated else None),
        }
    return out


def gate(ladder: dict, cells: dict) -> dict:
    """The preregistered rung 1 decision, evaluated mechanically."""
    rung6 = ladder["rungs"].get(6)
    t6 = cells.get("t6", {})
    m6 = t6.get("cells", {}).get(6)

    controls = {na: r.get("positive_control_passed")
                for na, r in ladder["rungs"].items() if "positive_control_passed" in r}
    controls_ok = bool(controls) and all(controls.values())

    clause_a = bool(rung6 and rung6["measured_gbps"] > BREAKEVEN[6])
    clause_b = bool(
        m6 and m6["seconds_delta_pct"] <= GATE_CELL_DELTA_PCT
        and t6.get("untreated_within_band"))

    return {
        "positive_controls": controls,
        "positive_controls_passed": controls_ok,
        "clause_a_bandwidth": {
            "measured_bw6_gbps": rung6["measured_gbps"] if rung6 else None,
            "threshold_gbps": BREAKEVEN[6],
            "fired": clause_a,
        },
        "clause_b_measured_cell": {
            "m6_delta_pct": m6["seconds_delta_pct"] if m6 else None,
            "threshold_pct": GATE_CELL_DELTA_PCT,
            "untreated_within_band": t6.get("untreated_within_band"),
            "fired": clause_b,
        },
        "proceed_to_rung2_and_rung3": controls_ok and (clause_a or clause_b),
        "note": ("The ladder is void if any positive control misses; the gate then "
                 "reports the control failure instead of a verdict."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default=str(LEGS))
    ap.add_argument("--out", default="research/e61-artifacts/e61-bandwidth.json")
    args = ap.parse_args()

    summary, peak, provenance = per_arm_widths(pathlib.Path(args.legs))
    ladder = lone_group_ladder(summary)
    cells = cell_deltas(summary)
    verdict = gate(ladder, cells)

    out = {
        "measured_peak_bandwidth_bytes_per_second": peak,
        "legs_per_arm": provenance,
        "lone_group_ladder": ladder,
        "cell_deltas": cells,
        "gate": verdict,
        "per_arm_per_width": summary,
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")

    print("measured stream peak: %.1f GB/s" % (peak / 1e9))
    print("\nlone working group, achieved device rate against its NA")
    for na, r in sorted(ladder["rungs"].items()):
        line = "   NA=%d  %6.1f GB/s" % (na, r["measured_gbps"])
        if r.get("e54_anchor_gbps"):
            line += "   anchor %6.1f (%+.2f %%) control %s" % (
                r["e54_anchor_gbps"], r["anchor_delta_pct"],
                "PASS" if r["positive_control_passed"] else "MISS")
        if r.get("predicted_gbps"):
            line += "   predicted %6.1f (miss %+.1f) linear %s" % (
                r["predicted_gbps"], r["prediction_miss_gbps"],
                "REFUTED" if r["linearity_refuted"] else "held")
        if r.get("breakeven_gbps"):
            line += "   break-even %6.2f -> %s" % (
                r["breakeven_gbps"], "CLEARS" if r["clears_breakeven"] else "fails")
        print(line)
    print("   measured steps: %s" % {k: round(v, 2) for k, v in
                                     ladder["measured_steps_gbps"].items()})

    for arm, blk in sorted(cells.items()):
        print("\n%s minus shipped, per width" % arm)
        for m, c in sorted(blk["cells"].items()):
            pred = ("  predicted %+.2f %%" % c["predicted_delta_pct"]
                    if c["predicted_delta_pct"] is not None else "")
            print("   m=%-2d %-9s na %s -> %s  %+.2f %%%s"
                  % (m, "TREATED" if c["treated_width"] else "untreated",
                     c["control_group_na"], c["treated_group_na"],
                     c["seconds_delta_pct"], pred))
        if blk["untreated_mean_delta_pct"] is not None:
            print("   untreated widths %s: mean %+.2f %%, max |d| %.2f %% -> %s"
                  % (blk["untreated_widths"], blk["untreated_mean_delta_pct"],
                     blk["untreated_max_abs_delta_pct"],
                     "within band" if blk["untreated_within_band"] else "OUT OF BAND"))

    print("\ngate: controls %s, clause A %s, clause B %s -> %s"
          % ("PASS" if verdict["positive_controls_passed"] else "MISS",
             "fired" if verdict["clause_a_bandwidth"]["fired"] else "no",
             "fired" if verdict["clause_b_measured_cell"]["fired"] else "no",
             "PROCEED" if verdict["proceed_to_rung2_and_rung3"] else "STOP"))
    print("wrote %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
