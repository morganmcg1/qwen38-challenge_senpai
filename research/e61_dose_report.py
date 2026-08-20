#!/usr/bin/env python3
"""E61 rung 1b: read the register dose response at fixed routing.

The session is an 8-leg palindrome d1 e1 f1 g1 g2 f2 e2 d2 over four arms whose
scored-table register maxima are known from research/e61_reg_census.py:

  shipped_rbx  125   routing identical to shipped
  shipped      129   routing identical to shipped
  ballast      144   routing identical to shipped, ceiling raised by an
                     unreachable switch case
  t6_rbx       145   M=6 routed to <T,6,6> (single weight stream)

The first three share routing, so any cost difference between them is a
register/occupancy effect, not a scheduling effect. t6_rbx adds the routing
change on top of the highest ceiling.

  python3 research/e61_dose_report.py [--out PATH]
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

CURVES = pathlib.Path(".mlxfast-private/qmv-curve")
LEGS = pathlib.Path(".mlxfast-private/e61-legs")

# leg tag -> (arm, scored-table register max from the rung 1b census)
ORDER = [
    ("d1", "shipped", 129),
    ("e1", "ballast", 144),
    ("f1", "shipped_rbx", 125),
    ("g1", "t6_rbx", 145),
    ("g2", "t6_rbx", 145),
    ("f2", "shipped_rbx", 125),
    ("e2", "ballast", 144),
    ("d2", "shipped", 129),
]


def leg(tag: str) -> dict:
    curve = json.loads((CURVES / tag / "vendored.json").read_text())
    meta = json.loads((LEGS / f"{tag}-leg.json").read_text())
    temps = {}
    for line in meta["identity"]:
        if line.startswith("gpu_temp_c_") and "=" in line:
            name, value = line.split("=", 1)
            temps[name] = float(value)
    total = 0.0
    shapes = []
    for shape in curve["shapes"]:
        for row in shape["rows"]:
            if row["m"] != 9:
                continue
            calls = shape["calls_per_verify"]
            total += calls * row["seconds_per_call"]
            shapes.append(
                {
                    "name": shape["name"],
                    "calls_per_verify": calls,
                    "seconds_per_call": row["seconds_per_call"],
                    "in_kernel_path": row["in_kernel_path"],
                    "weight_streams": row["weight_streams"],
                    "inputs_per_group": row["inputs_per_group"],
                    "row0_bitwise_matches_m1": row["row0_bitwise_matches_m1"],
                }
            )
    return {
        "tag": tag,
        "verify_seconds_m9": total,
        "entry_temp_c": temps.get("gpu_temp_c_before_vendored"),
        "exit_temp_c": temps.get("gpu_temp_c_after_vendored"),
        "measured_commit": meta["measured_commit_unwound"],
        "source_sha256": meta["sources_as_measured"],
        "shapes": shapes,
    }


def main() -> None:
    legs = {tag: leg(tag) for tag, _, _ in ORDER}
    arms: dict[str, dict] = {}
    for tag, arm, regs in ORDER:
        arms.setdefault(arm, {"registers": regs, "tags": [], "values": []})
        arms[arm]["tags"].append(tag)
        arms[arm]["values"].append(legs[tag]["verify_seconds_m9"])

    for arm, rec in arms.items():
        rec["mean"] = statistics.fmean(rec["values"])
        rec["spread_pct"] = (
            100.0 * (max(rec["values"]) - min(rec["values"])) / rec["mean"]
        )

    ref = arms["shipped"]["mean"]
    for rec in arms.values():
        rec["delta_pct_vs_shipped"] = 100.0 * (rec["mean"] - ref) / ref

    routing_equal = {
        arm: [s["in_kernel_path"] for s in legs[rec["tags"][0]]["shapes"]]
        == [s["in_kernel_path"] for s in legs["d1"]["shapes"]]
        for arm, rec in arms.items()
    }

    out = {
        "session": "e61-rung1b-register-dose-response",
        "widths": 9,
        "reps": 21,
        "inner_calls": 10,
        "order": [t for t, _, _ in ORDER],
        "arms": arms,
        "routing_matches_shipped": routing_equal,
        "legs": legs,
        "session_null_pct_same_arm_two_apart": arms["shipped"]["spread_pct"],
        "cool_gate_passed_real_gate": True,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "wired_residency_active": False,
    }

    print(f"{'arm':13s} {'regs':>4s} {'T9 ms':>9s} {'spread%':>8s} {'vs shipped%':>12s}  routing")
    for arm in ("shipped_rbx", "shipped", "ballast", "t6_rbx"):
        r = arms[arm]
        print(
            f"{arm:13s} {r['registers']:4d} {r['mean']*1e3:9.5f} "
            f"{r['spread_pct']:8.4f} {r['delta_pct_vs_shipped']:12.4f}  "
            f"{'same as shipped' if routing_equal[arm] else 'CHANGED'}"
        )
    print()
    print("per-leg order, entry/exit temperature:")
    for tag, arm, _ in ORDER:
        L = legs[tag]
        print(
            f"  {tag} {arm:13s} {L['verify_seconds_m9']*1e3:9.5f} ms  "
            f"in={L['entry_temp_c']:.2f}C out={L['exit_temp_c']:.2f}C"
        )
    entries = [legs[t]["entry_temp_c"] for t, _, _ in ORDER]
    print(f"  entry-temperature spread = {max(entries)-min(entries):.2f} C")

    if "--out" in sys.argv:
        path = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
