#!/usr/bin/env python3
"""E59 rung 2 verdict: turn the parity digests into a pass or a hard stop.

Stop rule 2 is "any delta at any M <= 9 is a hard stop", and a check that cannot
fail is not a check. This asserts BOTH halves:

  candidates   `m5_rb2` and `m5_rbx` must be bit-identical to the unchanged base
               at every swept width and bit width.
  ceiling arm  `ceil_only` must be identical at every REACHABLE width (M <= 9)
               and must differ at M = 10, where the base falls through to
               `qmv_fast_impl` and the arm reaches its unreachable case.
  controls     each defect arm must diverge, and only at M = 5, bits = 4.

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
    return {(e["shape"], e["bits"], e["m"]): e["digest"]
            for e in payload["entries"]}


def diff_cells(ref: dict, arm: dict) -> list[tuple]:
    return sorted(k for k in set(ref) & set(arm) if ref[k] != arm[k])


def check(name: str, ref: dict, arm: dict, expect: str) -> dict:
    differing = diff_cells(ref, arm)
    widths = sorted({m for _, _, m in differing})
    bits = sorted({b for _, b, _ in differing})
    reachable = [m for m in widths if m <= MAX_REACHABLE_WIDTH]
    if expect == "identical":
        passed = not differing
    elif expect == "identical_below_10":
        passed = not reachable and widths == [10]
    elif expect == "diverges_at_m5":
        passed = widths == [5] and bits == [4]
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
        "passed": passed,
    }


COMPARISONS = [
    ("m5_rb2 vs base", "shipped", "m5_rb2", "identical"),
    ("m5_rbx vs base", "shipped", "m5_rbx", "identical"),
    ("ceil_only vs base", "shipped", "ceil_only", "identical_below_10"),
    ("lane perturbation vs m5_rb2", "m5_rb2", "m5_rb2_lane_perturb",
     "diverges_at_m5"),
    ("one row block vs m5_rb2", "m5_rb2", "m5_rb2_coverage_drop",
     "diverges_at_m5"),
    ("one x-group vs m5_rbx", "m5_rbx", "m5_rbx_coverage_drop",
     "diverges_at_m5"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-dir", required=True)
    ap.add_argument("--out", default="research/e59-artifacts/e59-parity.json")
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
    print("  %-30s %-20s %8s %9s  %s"
          % ("comparison", "expectation", "compared", "differing", "verdict"))
    for r in rows:
        print("  %-30s %-20s %8s %9s  %s"
              % (r["comparison"], r["expectation"],
                 r.get("cells_compared", "-"), r.get("cells_differing", "-"),
                 "PASS" if r["passed"] else "FAIL"))
        if r.get("widths_differing"):
            print("      widths differing: %s   bits: %s"
                  % (r["widths_differing"], r["bits_differing"]))

    ok = all(r["passed"] for r in rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"max_reachable_width": MAX_REACHABLE_WIDTH, "comparisons": rows,
         "all_passed": ok}, indent=2, sort_keys=True) + "\n")
    print("\nall_passed=%s   wrote %s" % (ok, args.out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
