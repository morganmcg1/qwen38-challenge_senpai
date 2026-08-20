#!/usr/bin/env python3
"""E59 rung 4 verdict: turn the parity digests into a pass or a hard stop.

Stop rule 2 is "any delta at any M <= 9 is a hard stop", and a check that cannot
fail is not a check. This asserts BOTH halves:

  candidates   `t55` and `m5_rbx` must be bit-identical to the unchanged base at
               every swept width and bit width. `m5_rbx` holds NA fixed, so its
               answer is exact by construction and this only confirms the build.
               `t55` merges the accumulator groups {3, 2} into one group of 5,
               so its exactness is an empirical claim and this gate is the only
               thing that establishes it before the arm is timed.
  controls     each defect arm must diverge, and only where its defect lives.
               `perturb_lanes` is gated on NA == 5, and E55 shipped `<T,9,5>`,
               so the lane control fires at M=5 and M=9. The row-drop control
               fires at M=5 alone.

Exits non-zero if any expectation fails, so the session can stop before it
spends a GPU hour on a route that is not exact.

  python3 research/e59_parity_verdict.py --parity-dir DIR --out JSON
"""

from __future__ import annotations

import argparse
import json
import pathlib

# The scored session verifies one primary token plus at most eight drafts.
MAX_REACHABLE_WIDTH = 9


def load(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text())
    return {(e["shape"], e["bits"], e["m"]): e for e in payload["entries"]}


def diff_cells(ref: dict, arm: dict) -> list[tuple]:
    return sorted(k for k in set(ref) & set(arm)
                  if ref[k]["digest"] != arm[k]["digest"])


def route_changes(ref: dict, arm: dict) -> list[dict]:
    out = []
    for k in sorted(set(ref) & set(arm)):
        before, after = ref[k]["in_kernel_path"], arm[k]["in_kernel_path"]
        if before != after:
            out.append({"shape": k[0], "bits": k[1], "m": k[2],
                        "from": before, "to": after})
    return out


def check(name: str, ref: dict, arm: dict, expect: str) -> dict:
    differing = diff_cells(ref, arm)
    routes = route_changes(ref, arm)
    widths = sorted({m for _, _, m in differing})
    bits = sorted({b for _, b, _ in differing})
    reachable = [m for m in widths if m <= MAX_REACHABLE_WIDTH]
    unreachable = [m for m in widths if m > MAX_REACHABLE_WIDTH]
    if expect == "identical":
        passed = not differing
    elif expect == "identical_at_reachable_widths":
        passed = not reachable
    elif expect == "diverges_at_m5":
        passed = widths == [5] and bits == [4]
    elif expect == "diverges_at_m5_and_m9":
        passed = widths == [5, 9] and bits == [4]
    else:
        raise SystemExit("e59_parity_verdict: unknown expectation %s" % expect)
    return {
        "comparison": name,
        "expectation": expect,
        "cells_compared": len(set(ref) & set(arm)),
        "cells_differing": len(differing),
        "widths_differing": widths,
        "bits_differing": bits,
        "reachable_widths_differing": reachable,
        "unreachable_widths_differing": unreachable,
        "first_difference": (
            {"shape": differing[0][0], "bits": differing[0][1],
             "m": differing[0][2]} if differing else None),
        "route_changes": len(routes),
        "route_change_widths": sorted({r["m"] for r in routes}),
        "route_change_detail": routes[:1],
        "passed": passed,
    }


COMPARISONS = [
    ("t55 vs base", "shipped", "t55", "identical"),
    ("m5_rbx vs base", "shipped", "m5_rbx", "identical"),
    ("lane perturbation vs t55", "t55", "t55_lane_perturb",
     "diverges_at_m5_and_m9"),
    ("four written rows vs t55", "t55", "t55_row_drop", "diverges_at_m5"),
    ("one x-group vs m5_rbx", "m5_rbx", "m5_rbx_coverage_drop",
     "diverges_at_m5"),
]

CONTROL_EXPECTATIONS = ("diverges_at_m5", "diverges_at_m5_and_m9")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-dir", required=True)
    ap.add_argument("--out",
                    default="research/e59-artifacts/e59-parity-rung4.json")
    args = ap.parse_args()

    d = pathlib.Path(args.parity_dir)
    digests = {p.stem: load(p) for p in sorted(d.glob("*.json"))}

    rows = []
    for name, ref, arm, expect in COMPARISONS:
        if ref not in digests or arm not in digests:
            rows.append({"comparison": name, "expectation": expect,
                         "passed": False, "error": "missing digest file"})
            continue
        rows.append(check(name, digests[ref], digests[arm], expect))

    print("E59 PARITY VERDICT")
    print("  %-30s %-30s %8s %9s %7s  %s"
          % ("comparison", "expectation", "compared", "differing", "routed",
             "verdict"))
    for r in rows:
        print("  %-30s %-30s %8s %9s %7s  %s"
              % (r["comparison"], r["expectation"],
                 r.get("cells_compared", "-"), r.get("cells_differing", "-"),
                 r.get("route_changes", "-"),
                 "PASS" if r["passed"] else "FAIL"))
        if r.get("widths_differing"):
            print("      widths differing: %s   bits: %s"
                  % (r["widths_differing"], r["bits_differing"]))
        if r.get("route_change_detail"):
            d = r["route_change_detail"][0]
            print("      route change at M=%s: %s -> %s"
                  % (d["m"], d["from"], d["to"]))

    controls = [r for r in rows if r["expectation"] in CONTROL_EXPECTATIONS]
    ok = all(r["passed"] for r in rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"max_reachable_width": MAX_REACHABLE_WIDTH, "comparisons": rows,
         "controls_total": len(controls),
         "controls_fired": sum(1 for r in controls if r["passed"]),
         "all_passed": ok}, indent=2, sort_keys=True) + "\n")
    print("\nall_passed=%s   controls_fired=%s/%s   wrote %s"
          % (ok, sum(1 for r in controls if r["passed"]), len(controls),
             args.out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
