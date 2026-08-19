#!/usr/bin/env python3
"""E49 arm definitions: the exact source edits behind every timed leg.

Two questions, two families of arm.

ISOLATED (arm 1)  `iso3` / `iso5` keep ONE case in the >=4096 crossrow switch,
                  so the `<T,9,IPG>` cell under test is the only crossrow body
                  the shared `[[kernel]]` allocation has to cover. Everything
                  else about the two builds is identical, so the M=9 delta is
                  the local cost of 3 streams -> 2 streams including whatever
                  that cell's own register footprint costs it.

SHARED (arm 2)    `dose_*` inject an extra `case 10:` that the workload can
                  never dispatch (`ntg.x == M <= 9`). No dispatched width's code
                  changes; only the kernel-wide register allocation does. Any
                  movement on widths 3..9 is the shared-allocation tax and
                  nothing else. `dose_null` injects a cell BELOW the shipped
                  108 max, so it adds a branch without raising the ceiling: the
                  harness check that reproduces E38 arm (b)'s null.

`e27_replica` is the composite E27 actually shipped for M=9 (local + shared),
kept so `composite == local + tax` is falsifiable rather than assumed.

Every edit is applied to the readable header AND its runtime-effective
generated twin, and reverted before the branch is submitted: both files are
held byte-identical to the frontier by `research/scored-surface-gate.sh`.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
SOURCES = (HEADER, TWIN)

NA_ASSERT = ('static_assert(NA >= 2 && NA <= 4, '
             '"wide multi-row QMV supports NA in [2, 4]");')
NA_ASSERT_RELAXED = ('static_assert(NA >= 2 && NA <= 8, '
                     '"wide multi-row QMV supports NA in [2, 8]");')
M_ASSERT = ('static_assert(M >= 3 && M <= 9, '
            '"wide multi-row QMV dispatch covers M in [3, 9]");')
M_ASSERT_RELAXED = ('static_assert(M >= 3 && M <= 16, '
                    '"wide multi-row QMV dispatch covers M in [3, 16]");')
M9_SHIPPED = "qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>"
M9_TWO_STREAM = "qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>"

WIDE_TIER_ANCHOR = "if (out_vec_size >= 4096) {"
SWITCH_OPEN = "switch (ntg.x) {"
CASE_9 = "        case 9:\n"
WIDE_TIER_DEFAULT = "        default:\n          break;\n      }\n    } else {"

CASE_TEMPLATE = """        case %(label)d:
          qmv_fast_crossrow_affine4_g64_m<T, %(m)d, %(ipg)d, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
"""


def _relax(text: str) -> str:
    """Admit the probe-only NA and M ranges the shipped asserts refuse."""
    out = text
    for old, new in ((NA_ASSERT, NA_ASSERT_RELAXED), (M_ASSERT, M_ASSERT_RELAXED)):
        if out.count(old) != 1:
            raise SystemExit("e49_arms: assert anchor not unique: %s" % old)
        out = out.replace(old, new)
    return out


def _only_case_9(text: str) -> str:
    """Delete cases 2..8 from the >=4096 crossrow switch, keeping case 9."""
    tier = text.index(WIDE_TIER_ANCHOR)
    sw = text.index(SWITCH_OPEN, tier)
    c9 = text.index(CASE_9, sw)
    return text[:sw] + SWITCH_OPEN + "\n" + text[c9:]


def _swap_m9(text: str, ipg: int) -> str:
    new = "qmv_fast_crossrow_affine4_g64_m<T, 9, %d, true>" % ipg
    if text.count(M9_SHIPPED) != 1:
        raise SystemExit("e49_arms: M=9 dispatch anchor not unique")
    return text.replace(M9_SHIPPED, new)


def _inject_dose(text: str, m: int, ipg: int, label: int = 10) -> str:
    """Add an unreachable `case label:` to the >=4096 switch.

    `ntg.x == M` and the workload never verifies more than 9 rows, so the cell
    is compiled and allocated for but never executed. That is the whole point:
    it moves the kernel-wide allocation without moving any dispatched width's
    instructions.
    """
    if text.count(WIDE_TIER_DEFAULT) != 1:
        raise SystemExit("e49_arms: wide-tier default anchor not unique")
    case = CASE_TEMPLATE % {"label": label, "m": m, "ipg": ipg}
    return text.replace(WIDE_TIER_DEFAULT, case + WIDE_TIER_DEFAULT)


# name -> (description, transform). Transforms take and return file text.
ARMS: dict[str, dict] = {
    "shipped": {
        "family": "control",
        "doc": "the tip, unmodified: kernel-wide max 108 at <T,7,4>",
        "steps": [],
    },
    "iso3": {
        "family": "isolated",
        "doc": "only case 9 in the wide tier, <T,9,3> (3 streams)",
        "steps": [("only_case_9", {})],
    },
    "iso5": {
        "family": "isolated",
        "doc": "only case 9 in the wide tier, <T,9,5> (2 streams)",
        "steps": [("relax", {}), ("only_case_9", {}), ("swap_m9", {"ipg": 5})],
    },
    "dose_null": {
        "family": "shared",
        "doc": "unreachable case 10 = <T,4,4> (104 regs, below the 108 max)",
        "steps": [("inject", {"m": 4, "ipg": 4})],
    },
    "dose_129": {
        "family": "shared",
        "doc": "unreachable case 10 = <T,9,5> (129): E27's exact ceiling, no "
               "dispatched width changed",
        "steps": [("relax", {}), ("inject", {"m": 9, "ipg": 5})],
    },
    "dose_big": {
        "family": "shared",
        "doc": "unreachable case 10 = <T,12,6>: a deliberately large dose well "
               "past 129, so a null is a null at a dose we cannot have missed",
        "steps": [("relax", {}), ("inject", {"m": 12, "ipg": 6})],
    },
    "dose_huge": {
        "family": "shared",
        "doc": "unreachable case 10 = <T,16,8>: the largest dose that compiles",
        "steps": [("relax", {}), ("inject", {"m": 16, "ipg": 8})],
    },
    "e27_replica": {
        "family": "composite",
        "doc": "E27's M=9 edit on the full table: local + shared together",
        "steps": [("relax", {}), ("swap_m9", {"ipg": 5})],
    },
}

_STEPS = {
    "relax": lambda t, **kw: _relax(t),
    "only_case_9": lambda t, **kw: _only_case_9(t),
    "swap_m9": lambda t, ipg=5, **kw: _swap_m9(t, ipg),
    "inject": lambda t, m=9, ipg=5, label=10, **kw: _inject_dose(t, m, ipg, label),
}


def apply_arm(text: str, name: str) -> str:
    if name not in ARMS:
        raise SystemExit("e49_arms: unknown arm %s" % name)
    for step, kwargs in ARMS[name]["steps"]:
        text = _STEPS[step](text, **kwargs)
    return text


def main() -> int:
    import argparse
    import hashlib
    import json

    ap = argparse.ArgumentParser(description="apply one E49 arm in the worktree")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--out", help="write the patched-file digests here")
    args = ap.parse_args()

    digests = {}
    for path in SOURCES:
        text = apply_arm(path.read_text(), args.arm)
        path.write_text(text)
        digests[str(path.relative_to(REPO))] = hashlib.sha256(
            text.encode()).hexdigest()
    payload = {"arm": args.arm, "doc": ARMS[args.arm]["doc"],
               "family": ARMS[args.arm]["family"], "sha256": digests}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
