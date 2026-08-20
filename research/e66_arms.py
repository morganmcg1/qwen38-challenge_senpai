#!/usr/bin/env python3
"""E66 arm definitions: do `t55` and `t6` compose additively at the whole leg?

The base already carries `t6` (`case 6` -> `<T,6,6>`, wide-helper bound raised to
NA <= 6). `t55` moves `case 5` from two weight streams `{3,2}` to one `[5]` and
needs no bound change, because `t6` already raised it.

Arms, all applied to BOTH quantized twins and unwound on every leg exit path:

  a_neither   `case 5` -> `<T,5,3>`, `case 6` -> `<T,6,3>`, bound back to NA <= 5.
              The pre-`t6` shipped table. Restoring the bound is required: with
              the bound left at 6 the arm is not the table that shipped.
  b_t6        the merged base, unmodified: `<T,5,3>` and `<T,6,6>`.
  c_t55_t6    the candidate: `<T,5,5>` and `<T,6,6>`.
  c_lane_perturb
              positive control for the rung 2 instrument: arm C with input rows
              3 and 4 written to each other's accumulator lane in every NA=5
              group. The row ledger MUST differ from arm C on this. Never timed
              and never submitted.

Source-byte growth for `c_t55_t6` against the base is 0: one character per twin.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
SOURCES = (HEADER, TWIN)

LANE_WRITE = """        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];"""
# NA == 5 is the group width `t55` introduces at M=5. It is also the FIRST group
# of the M=9 cell, so this control perturbs both and proves only that the ledger
# instrument reads the NA=5 helper body, not that it isolates M=5.
LANE_WRITE_PERTURBED = """        const int lane = (NA == 5) ? ((m == 3) ? 4 : ((m == 4) ? 3 : m)) : m;
        a0[lane] = xc[0];
        a1[lane] = xc[1];
        a2[lane] = xc[2];
        a3[lane] = xc[3];"""


def na_assert(bound: int) -> str:
    return ('static_assert(NA >= 2 && NA <= %d, '
            '"wide multi-row QMV supports NA in [2, %d]");' % (bound, bound))


def set_na_bound(text: str, bound: int) -> str:
    pat = re.compile(r'static_assert\(NA >= 2 && NA <= (\d+), '
                     r'"wide multi-row QMV supports NA in \[2, \d+\]"\);')
    if len(pat.findall(text)) != 1:
        raise SystemExit("e66_arms: NA assert anchor not unique")
    return pat.sub(na_assert(bound).replace("\\", "\\\\"), text)


def swap_ipg(text: str, m: int, ipg: int) -> str:
    """Write `case m:` as `<T, m, ipg>` whatever IPG it currently carries."""
    pat = re.compile(r"qmv_fast_crossrow_affine4_g64_m<T, %d, \d+, true>" % m)
    if len(pat.findall(text)) != 1:
        raise SystemExit("e66_arms: M=%d dispatch anchor not unique" % m)
    return pat.sub("qmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>" % (m, ipg),
                   text)


def perturb_lanes(text: str) -> str:
    if text.count(LANE_WRITE) != 1:
        raise SystemExit("e66_arms: lane-write anchor not unique")
    return text.replace(LANE_WRITE, LANE_WRITE_PERTURBED)


ARMS: dict[str, dict] = {
    "a_neither": {
        "doc": "pre-t6 shipped table: case 5 <T,5,3>, case 6 <T,6,3>, "
               "wide-helper bound back to NA <= 5",
        "cells": {5: 3, 6: 3},
        "na_bound": 5,
    },
    "b_t6": {
        "doc": "t6 only: case 5 <T,5,3>, case 6 <T,6,6>",
        "cells": {5: 3, 6: 6},
        "na_bound": 6,
    },
    "c_t55_t6": {
        "doc": "the candidate: case 5 -> <T,5,5> composed with "
               "case 6 <T,6,6>. One weight stream at both M=5 and M=6",
        "cells": {5: 5, 6: 6},
        "na_bound": 6,
    },
    "c_lane_perturb": {
        "doc": "positive control: arm C with input rows 3 and 4 written to each "
               "other's lane in every NA=5 group; the row ledger MUST differ",
        "cells": {5: 5, 6: 6},
        "na_bound": 6,
        "perturb": True,
        "never_time": True,
        "never_submit": True,
    },
}


def apply_arm(text: str, name: str) -> str:
    """Write the arm's declared final table, whatever table is checked out.

    Arms are states, not patches: `a_neither` and `b_t6` are reversions once the
    base carries `t55`, and `c_t55_t6` is then a no-op. The compiled object is
    the same either way, so the measured contrast does not depend on the base.
    """
    if name not in ARMS:
        raise SystemExit("e66_arms: unknown arm %s" % name)
    arm = ARMS[name]
    for m, ipg in sorted(arm["cells"].items()):
        text = swap_ipg(text, m, ipg)
    text = set_na_bound(text, arm["na_bound"])
    if arm.get("perturb"):
        text = perturb_lanes(text)
    return text


def main() -> int:
    import argparse
    import hashlib
    import json

    ap = argparse.ArgumentParser(description="apply one E66 arm in the worktree")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--out", help="write the patched-file digests here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    digests = {}
    sizes = {}
    for path in SOURCES:
        text = apply_arm(path.read_text(), args.arm)
        if not args.dry_run:
            path.write_text(text)
        rel = str(path.relative_to(REPO))
        digests[rel] = hashlib.sha256(text.encode()).hexdigest()
        sizes[rel] = len(text.encode())
    spec = ARMS[args.arm]
    payload = {"arm": args.arm, "doc": spec["doc"], "cells": spec["cells"],
               "na_bound": spec["na_bound"], "dry_run": bool(args.dry_run),
               "never_submit": bool(spec.get("never_submit")),
               "never_time": bool(spec.get("never_time")),
               "source_bytes": sizes, "sha256": digests}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
