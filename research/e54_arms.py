#!/usr/bin/env python3
"""E54 arm definitions: is NA=5 profitable only when a sibling group shares it?

E49 measured ONE NA=5 cell, `<T,9,5>`, isolated, and it won by 12.255 %. That
cell runs its NA=5 group beside an NA=4 tail sibling. `<T,5,5>` is the only
legal IPG=5 configuration in the whole table whose NA=5 group runs ALONE
(`5 % 5 == 0`, one working group). E54 times the lone cell and two more mixed
cells so the four-cell table can separate three laws:

  A  weight traffic only          every cell wins, M=5 wins most
  B  NA=5 always loses            every cell loses (already dead at M=9)
  C  sibling overlap hides NA=5   M=7, 8, 9 win, M=5 regresses

ISOLATED pairs, E49 Arm 1's method exactly. Each arm keeps exactly ONE case in
the `out_vec_size >= 4096` crossrow switch, so the cell under test is the only
crossrow body the shared `[[kernel]]` allocation has to cover, and the other
nine swept widths fall through to `qmv_fast_impl` in BOTH arms of the pair and
are byte-identical in-session controls.

  iso_m5_ipg3 / iso_m5_ipg5   P1, the lone NA=5 cell. The primary.
  iso_m7_ipg4 / iso_m7_ipg5   P2, mixed sibling, group NA 5 + 2
  iso_m8_ipg4 / iso_m8_ipg5   P3, mixed sibling, group NA 5 + 3

COMPOSITE, for P4 only, and only after the isolated cells:

  shipped     the tip unmodified
  e27_full    E27's real composite on the REAL table: case 5 -> <T,5,5> and
              case 9 -> <T,9,5> together

CORRECTNESS. `iso_m5_ipg5_lane_perturb` swaps the write lanes of input rows 3
and 4 inside NA=5 groups only. It is a positive control: a bitwise lane check
that does not fail on it has no power over a real lane bug either. It is never
timed and never submitted.

Every edit is applied to the readable header AND its runtime-effective generated
twin, and both files are unwound on every leg exit path, so the branch's scored
surface stays byte-identical to the campaign base.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
SOURCES = (HEADER, TWIN)

NA_ASSERT = ('static_assert(NA >= 2 && NA <= 4, '
             '"wide multi-row QMV supports NA in [2, 4]");')
NA_ASSERT_RELAXED = ('static_assert(NA >= 2 && NA <= 5, '
                     '"wide multi-row QMV supports NA in [2, 5]");')

WIDE_TIER_ANCHOR = "if (out_vec_size >= 4096) {"
SWITCH_OPEN = "switch (ntg.x) {"
WIDE_TIER_DEFAULT = "        default:\n          break;\n      }\n    } else {"

SHIPPED_IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# The four lines that map input row m onto accumulator lane m. The positive
# control is a permutation of that map and nothing else.
LANE_WRITE = """        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];"""
LANE_WRITE_PERTURBED = """        const int lane = (NA == 5) ? ((m == 3) ? 4 : ((m == 4) ? 3 : m)) : m;
        a0[lane] = xc[0];
        a1[lane] = xc[1];
        a2[lane] = xc[2];
        a3[lane] = xc[3];"""

CASE_BLOCK = re.compile(
    r"        case (?P<m>\d+):\n(?P<body>.*?\n          return;\n)", re.S)


def relax_asserts(text: str) -> str:
    """Admit the probe-only NA range the shipped assert refuses.

    NA only. `M in [3, 9]` and `M % IPG != 1` already hold for every cell E54
    compiles, so widening them would widen the patch without widening the
    experiment.
    """
    if text.count(NA_ASSERT) != 1:
        raise SystemExit("e54_arms: NA assert anchor not unique")
    return text.replace(NA_ASSERT, NA_ASSERT_RELAXED)


def _tier_bounds(text: str) -> tuple[int, int]:
    tier = text.index(WIDE_TIER_ANCHOR)
    start = text.index(SWITCH_OPEN, tier) + len(SWITCH_OPEN)
    end = text.index(WIDE_TIER_DEFAULT, start)
    return start, end


def only_case(text: str, m: int) -> str:
    """Keep exactly `case m:` in the >=4096 crossrow switch, drop the rest."""
    start, end = _tier_bounds(text)
    kept = [blk.group(0) for blk in CASE_BLOCK.finditer(text[start:end])
            if int(blk.group("m")) == m]
    if len(kept) != 1:
        raise SystemExit("e54_arms: case %d is not unique in the wide tier" % m)
    return text[:start] + "\n" + kept[0] + text[end:]


def swap_ipg(text: str, m: int, ipg: int) -> str:
    """Repoint `case m:` at `<T, m, ipg>` instead of its shipped IPG."""
    old = "qmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>" % (m, SHIPPED_IPG[m])
    new = "qmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>" % (m, ipg)
    if text.count(old) != 1:
        raise SystemExit("e54_arms: M=%d dispatch anchor not unique" % m)
    return text.replace(old, new)


def perturb_lanes(text: str) -> str:
    """Positive control: swap the write lanes of rows 3 and 4 when NA == 5."""
    if text.count(LANE_WRITE) != 1:
        raise SystemExit("e54_arms: lane-write anchor not unique")
    return text.replace(LANE_WRITE, LANE_WRITE_PERTURBED)


def _iso(m: int, ipg: int, structure: str) -> dict:
    tail = m % ipg
    groups = (m + ipg - 1) // ipg
    na = sorted({ipg} | ({max(tail, 2)} if tail else set()))
    steps: list[tuple[str, dict]] = []
    if ipg > 4:
        steps.append(("relax", {}))
    steps.append(("only_case", {"m": m}))
    if ipg != SHIPPED_IPG[m]:
        steps.append(("swap_ipg", {"m": m, "ipg": ipg}))
    return {
        "family": "isolated",
        "doc": "only case %d in the wide tier, <T,%d,%d>: %d working group%s, "
               "group NA %s (%s)" % (m, m, ipg, groups,
                                     "" if groups == 1 else "s", na, structure),
        "cell": "<T,%d,%d>" % (m, ipg),
        "width": m,
        "ipg": ipg,
        "working_groups": groups,
        "group_na_values": na,
        "structure": structure,
        "steps": steps,
    }


ARMS: dict[str, dict] = {
    "iso_m5_ipg3": _iso(5, 3, "shipped"),
    "iso_m5_ipg5": _iso(5, 5, "LONE NA=5"),
    "iso_m7_ipg4": _iso(7, 4, "shipped"),
    "iso_m7_ipg5": _iso(7, 5, "mixed sibling"),
    "iso_m8_ipg4": _iso(8, 4, "shipped"),
    "iso_m8_ipg5": _iso(8, 5, "mixed sibling"),
    "shipped": {
        "family": "control",
        "doc": "the tip, unmodified",
        "cell": None,
        "steps": [],
    },
    "e27_full": {
        "family": "composite",
        "doc": "E27's real composite on the shipped table: case 5 -> <T,5,5> "
               "and case 9 -> <T,9,5>, every other width untouched",
        "cell": "<T,5,5>+<T,9,5>",
        "steps": [("relax", {}), ("swap_ipg", {"m": 5, "ipg": 5}),
                  ("swap_ipg", {"m": 9, "ipg": 5})],
    },
    # Census-only: which half of E27 raises the shared register max. Never timed.
    "e27_m5_only": {
        "family": "composite",
        "doc": "real table, case 5 -> <T,5,5> only: E27's M=5 half alone",
        "cell": "<T,5,5>",
        "steps": [("relax", {}), ("swap_ipg", {"m": 5, "ipg": 5})],
        "never_time": True,
    },
    "e27_m9_only": {
        "family": "composite",
        "doc": "real table, case 9 -> <T,9,5> only: E27's M=9 half alone, and "
               "exactly what askeladd composes on PR #57",
        "cell": "<T,9,5>",
        "steps": [("relax", {}), ("swap_ipg", {"m": 9, "ipg": 5})],
        "never_time": True,
    },
    "iso_m5_ipg5_lane_perturb": {
        "family": "positive_control",
        "doc": "iso_m5_ipg5 with rows 3 and 4 written to each other's lane in "
               "NA=5 groups: the bitwise lane check MUST fail on this",
        "cell": "<T,5,5> lane 3<->4",
        "steps": [("relax", {}), ("only_case", {"m": 5}),
                  ("swap_ipg", {"m": 5, "ipg": 5}), ("perturb", {})],
        "never_time": True,
    },
}

_STEPS = {
    "relax": lambda t, **kw: relax_asserts(t),
    "only_case": lambda t, m=9, **kw: only_case(t, m),
    "swap_ipg": lambda t, m=9, ipg=5, **kw: swap_ipg(t, m, ipg),
    "perturb": lambda t, **kw: perturb_lanes(t),
}


def apply_arm(text: str, name: str) -> str:
    if name not in ARMS:
        raise SystemExit("e54_arms: unknown arm %s" % name)
    for step, kwargs in ARMS[name]["steps"]:
        text = _STEPS[step](text, **kwargs)
    return text


def main() -> int:
    import argparse
    import hashlib
    import json

    ap = argparse.ArgumentParser(description="apply one E54 arm in the worktree")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--out", help="write the patched-file digests here")
    ap.add_argument("--dry-run", action="store_true",
                    help="apply in memory only and report the digests")
    args = ap.parse_args()

    digests = {}
    for path in SOURCES:
        text = apply_arm(path.read_text(), args.arm)
        if not args.dry_run:
            path.write_text(text)
        digests[str(path.relative_to(REPO))] = hashlib.sha256(
            text.encode()).hexdigest()
    spec = ARMS[args.arm]
    payload = {"arm": args.arm, "doc": spec["doc"], "family": spec["family"],
               "cell": spec["cell"], "dry_run": bool(args.dry_run),
               "sha256": digests}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
