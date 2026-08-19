#!/usr/bin/env python3
"""E59 rung 3 analysis: score the isolated-cell legs against the pre-registration.

Reads every leg manifest under `.mlxfast-private/e59-legs/*-leg.json`, pairs it
with that leg's `vendored.json`, and applies `research/e59-artifacts/e59-prereg.json`.
No constant is re-derived here.

  python3 research/e59_analyze.py --out research/e59-artifacts/e59-cell-metrics.json

Exit code 0 means the rung 3 gate passed and a route was chosen. Exit code 2
means the pre-registered stop rule fired.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e49_analyze import contrast  # noqa: E402
from e54_analyze import by_arm  # noqa: E402
import e54_analyze  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
PREREG_PATH = REPO / "research/e59-artifacts/e59-prereg.json"
CENSUS_PATH = REPO / "research/e59-artifacts/e59-reg-census.json"

# e54_analyze.leg_records() is the exact reader this experiment needs; only the
# manifest directory differs, and that module reads it from a module constant.
e54_analyze.LEGS = REPO / ".mlxfast-private/e59-legs"


def entry_dose(census: dict, control_arm: str, treated_arm: str) -> dict | None:
    """Entry-point register delta carried by an isolated pair.

    Every width case is inlined into one entry point, so an isolated arm runs at
    the occupancy its own cell demands rather than at the shipped entry's 163.
    A pair that also moves the entry count therefore mixes an occupancy change
    into the algorithmic change, and the size of that dose has to be visible
    next to the timing before the number is read as a pure cell cost.
    """
    a, b = census["arms"].get(control_arm), census["arms"].get(treated_arm)
    if not a or not b:
        return None
    return {
        "control_entry_regs": a["entry_point_reg_max"],
        "treated_entry_regs": b["entry_point_reg_max"],
        "entry_reg_delta": b["entry_point_reg_max"] - a["entry_point_reg_max"],
        "control_cell_regs": a["kernel_wide_reg_max"],
        "treated_cell_regs": b["kernel_wide_reg_max"],
    }


def pair_report(arms: dict, spec: dict, floor: float) -> dict | None:
    control_arm, treated_arm = spec["control"], spec["treated"]
    if control_arm not in arms or treated_arm not in arms:
        return None
    widths = sorted(set(arms[control_arm][0]["t_ms"])
                    & set(arms[treated_arm][0]["t_ms"]))
    rows = contrast(arms[treated_arm], arms[control_arm], widths)
    if not rows:
        return None

    treated_m = spec["width"]
    controls = {m: r["delta_pct"] for m, r in rows.items() if m != treated_m}
    control_bar = max((abs(v) for v in controls.values()), default=0.0)
    pooled_ctrl = sum(controls.values()) / len(controls) if controls else None

    spread = None
    if treated_m in rows and rows[treated_m]["mde_ms"] is not None \
            and rows[treated_m]["control_ms"]:
        spread = 100.0 * rows[treated_m]["mde_ms"] / rows[treated_m]["control_ms"]
    bar = max([floor, control_bar] + ([spread] if spread is not None else []))

    d = rows[treated_m]["delta_pct"] if treated_m in rows else None
    return {
        "control_arm": control_arm,
        "treated_arm": treated_arm,
        "treated_width": treated_m,
        "question": spec["question"],
        "delta_pct": d,
        "delta_ms": rows[treated_m]["delta_ms"] if treated_m in rows else None,
        "exceeds_bar": bool(d is not None and abs(d) > bar),
        "direction": ("win" if d is not None and d < -bar
                      else "regress" if d is not None and d > bar else "null"),
        "bar_pct": round(bar, 3),
        "bar_inputs": {
            "prereg_floor_pct": floor,
            "worst_unchanged_code_width_pct": round(control_bar, 3),
            "treated_replicate_spread_pct": (
                round(spread, 3) if spread is not None else None),
        },
        "per_width": rows,
        "unchanged_code_widths_pct": controls,
        "pooled_unchanged_code_pct": (
            round(pooled_ctrl, 3) if pooled_ctrl is not None else None),
        "reference_pct": spec.get("reference_pct"),
        "reference_source": spec.get("reference_source"),
        "is_gate": bool(spec.get("gate")),
    }


def compose(a: float, b: float) -> float:
    """Compose two percent deltas multiplicatively and return percent."""
    return ((1.0 + a / 100.0) * (1.0 + b / 100.0) - 1.0) * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e59-artifacts/e59-cell-metrics.json")
    args = ap.parse_args()

    prereg = json.loads(PREREG_PATH.read_text())
    census = json.loads(CENSUS_PATH.read_text())
    floor = prereg["decision_bar_pct"]
    legs = e54_analyze.leg_records()
    arms = by_arm(legs)

    print("E59 rung 3 legs, in run order:")
    print("  %-26s %-22s %-9s %s" % ("tag", "arm", "gate", "T(M) ms"))
    for leg in legs:
        if leg["status"] != "ok":
            print("  %-26s %-22s %s" % (leg["tag"], leg["arm"], leg["status"]))
            continue
        ts = " ".join("%d:%.1f" % (m, v) for m, v in sorted(leg["t_ms"].items()))
        print("  %-26s %-22s %-9s %s"
              % (leg["tag"], leg["arm"], leg["gpu_gate"]["state"], ts))

    payload = {"legs": legs, "arms_measured": sorted(arms), "pairs": {},
               "prereg_floor_pct": floor}
    observed: dict[str, float] = {}

    for name, spec in prereg["pairs"].items():
        rep = pair_report(arms, spec, floor)
        if rep is None:
            continue
        rep["entry_dose"] = entry_dose(census, spec["control"], spec["treated"])
        payload["pairs"][name] = rep
        if rep["delta_pct"] is not None:
            observed[name] = rep["delta_pct"]

        print("\n%s -- %s vs %s  (M=%d treated)"
              % (name, rep["control_arm"], rep["treated_arm"], rep["treated_width"]))
        print("  %s" % rep["question"])
        print("  %-4s %-11s %-11s %-10s %-9s %-8s %s"
              % ("M", "control", "treated", "delta ms", "delta %", "MDE ms", ""))
        for m, r in sorted(rep["per_width"].items()):
            mark = "<- TREATED" if m == rep["treated_width"] else "unchanged code"
            print("  %-4d %-11.3f %-11.3f %-10.3f %-9.3f %-8s %s"
                  % (m, r["control_ms"], r["treated_ms"], r["delta_ms"],
                     r["delta_pct"], r["mde_ms"], mark))
        print("  bar %.3f %%  (floor %.3f, worst unchanged %.3f, spread %s)"
              % (rep["bar_pct"], rep["bar_inputs"]["prereg_floor_pct"],
                 rep["bar_inputs"]["worst_unchanged_code_width_pct"],
                 rep["bar_inputs"]["treated_replicate_spread_pct"]))
        print("  M=%d measured %+.3f %%  ->  %s"
              % (rep["treated_width"], rep["delta_pct"], rep["direction"]))
        if rep["reference_pct"] is not None:
            print("  reference for contrast only: %+.3f %%  (%s)"
                  % (rep["reference_pct"], rep["reference_source"]))
        if rep["entry_dose"]:
            e = rep["entry_dose"]
            print("  entry registers %d -> %d (%+d); cell %d -> %d"
                  % (e["control_entry_regs"], e["treated_entry_regs"],
                     e["entry_reg_delta"], e["control_cell_regs"],
                     e["treated_cell_regs"]))

    # --- additivity: does net = replication x tax? -----------------------------
    add = {}
    for net, tax in (("N1", "T1"), ("N2", "T2")):
        if net in observed and tax in observed and "R1" in observed:
            predicted = compose(observed["R1"], observed[tax])
            add[net] = {
                "predicted_pct": round(predicted, 3),
                "measured_pct": round(observed[net], 3),
                "residual_pct": round(observed[net] - predicted, 3),
                "model": "compose(R1, %s)" % tax,
            }
    if add:
        payload["additivity"] = add
        print("\nADDITIVITY  (reported, not a gate)")
        for net, a in add.items():
            print("  %-4s predicted %+.3f %%   measured %+.3f %%   residual %+.3f %%"
                  % (net, a["predicted_pct"], a["measured_pct"], a["residual_pct"]))

    # --- rung 3 gate -----------------------------------------------------------
    rule = prereg["stop_rules"]["rung3_net_cell_win"]
    threshold = rule["threshold_pct"]
    nets = {k: v for k, v in observed.items() if k in ("N1", "N2")}
    verdict = {"threshold_pct": threshold, "nets": nets}
    rc = 0
    if not nets:
        verdict["state"] = "undecided"
        verdict["reason"] = "no net-cell pair measured"
        rc = 2
    else:
        best = min(nets, key=lambda k: nets[k])
        verdict["best_pair"] = best
        verdict["best_pct"] = nets[best]
        route = {"N1": "m5_rb2", "N2": "m5_rbx"}
        if nets[best] <= threshold:
            others = [k for k in nets if k != best]
            bar = payload["pairs"][best]["bar_pct"]
            tie = bool(others and abs(nets[best] - nets[others[0]]) <= bar)
            chosen = "m5_rbx" if tie else route[best]
            verdict["state"] = "pass"
            verdict["tie_inside_bar"] = tie
            verdict["route_for_rung4"] = chosen
            verdict["route_rule"] = (
                "tie inside the bar -> carry rbx for its 90-register headroom"
                if tie else "carried the more negative net cell win")
        else:
            verdict["state"] = "stop"
            verdict["reason"] = (
                "best net cell win %+.3f %% is larger than the pre-registered "
                "%.1f %% threshold" % (nets[best], threshold))
            rc = 2
    payload["rung3_gate"] = verdict

    print("\nRUNG 3 GATE  (threshold %.1f %%)" % threshold)
    for k, v in sorted(nets.items()):
        print("  %-4s %+.3f %%" % (k, v))
    print("  state: %s" % verdict["state"].upper())
    if verdict["state"] == "pass":
        print("  route for rung 4: %s  (%s)"
              % (verdict["route_for_rung4"], verdict["route_rule"]))
    else:
        print("  %s" % verdict.get("reason", ""))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("\nwrote %s" % out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
