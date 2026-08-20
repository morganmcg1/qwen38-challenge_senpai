#!/usr/bin/env python3
"""E61 arm definitions: does a single weight stream at M=6 pay?

M=6 is the largest single verify width at rank, 30.9-34.7 % of ranked QMV time,
and the first width where the single-stream form has to buy its algorithmic win
with a register-ceiling rise: `<T,6,6>` needs 146 registers against the 129 the
`d2139c92` table sits at.

Arms, all applied to BOTH quantized twins and unwound on every leg exit path:

  shipped     the `d2139c92` tip, unmodified. `case 9` is already `<T,9,5>`.
  t6          whole table, `case 6` -> `<T,6,6>`. One weight stream at M=6.
              This is the rung-3 candidate.
  t7          whole table, `case 7` -> `<T,7,7>`. Bandwidth probe ONLY: it
              exists to measure the lone-group rate at NA=7, which closes M=7,
              M=8 and M=9 permanently. Never submitted.
  iso_m6_ipg3 / iso_m6_ipg6
              isolated M=6 pair, E49 Arm 1's method: only `case 6` survives in
              the wide tier, so the cell under test is the only crossrow body
              the shared `[[kernel]]` allocation has to cover.
  ballast     rung 4, contingent. `case 6` stays `<T,6,3>` and a dead `case 12`
              at `<T,12,6>` holds the table maximum at 146 without changing any
              scored route, so `ballast - shipped` prices the ceiling alone and
              `t6 - ballast` prices the algorithm alone. Never submitted.
  t6_lane_perturb
              positive control: rows 4 and 5 write each other's accumulator
              lane in NA=6 groups. The bitwise parity check MUST fail on it.
              Never timed.

Every arm that raises NA above the shipped bound also relaxes the wide helper's
NA static_assert to exactly the bound that arm needs, so an arm can never
compile a width its own assert would reject.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = REPO / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
SOURCES = (HEADER, TWIN)

NA_ASSERT = ('static_assert(NA >= 2 && NA <= 5, '
             '"wide multi-row QMV supports NA in [2, 5]");')
M_ASSERT = ('static_assert(M >= 3 && M <= 9, '
            '"wide multi-row QMV dispatch covers M in [3, 9]");')

WIDE_TIER_ANCHOR = "if (out_vec_size >= 4096) {"
SWITCH_OPEN = "switch (ntg.x) {"
WIDE_TIER_DEFAULT = "        default:\n          break;\n      }\n    } else {"

# The shipped table at d2139c92. `case 9` is <T,9,5> since E55 merged.
SHIPPED_IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5}

LANE_WRITE = """        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];"""
LANE_WRITE_PERTURBED = """        const int lane = (NA == 6) ? ((m == 4) ? 5 : ((m == 5) ? 4 : m)) : m;
        a0[lane] = xc[0];
        a1[lane] = xc[1];
        a2[lane] = xc[2];
        a3[lane] = xc[3];"""

CASE_BLOCK = re.compile(
    r"        case (?P<m>\d+):\n(?P<body>.*?\n          return;\n)", re.S)

CASE9_TAIL = """        case 9:
          qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        default:"""
# M=12 at IPG=6 splits {6,6}: one distinct group size, so the census law puts
# it at 20 + 21*6 = 146, exactly the t6 table maximum, with no mixed-group
# over-count. The host offers at most 8 drafts, so ntg.x never reaches 12 and
# this case is unreachable dead code that only the register allocator sees.
CASE9_TAIL_PLUS_BALLAST = """        case 9:
          qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        case 12:
          qmv_fast_crossrow_affine4_g64_m<T, 12, 6, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
        default:"""


WRAPPER_BODY = """  constexpr int TAIL = M % IPG;
  const int first_m = int(tid.x) * IPG;
  if (first_m >= M) {
    return;
  }
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  if (TAIL == 0 || M - first_m >= IPG) {
    qmv_fast_crossrow_affine4_g64_wide<T, IPG, DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);
  } else {
    qmv_fast_crossrow_affine4_g64_wide<
        T, (TAIL >= 2 ? TAIL : 2), DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, out_row, simd_lid);
  }"""

# M <= 9 and IPG >= 3, so a group index never exceeds 2. Every guard below is a
# compile-time constant, so the branches a cell does not use are dead code.
WRAPPER_BODY_RBX = """  constexpr int TAIL = M % IPG;
  constexpr int GROUPS = (M + IPG - 1) / IPG;
  constexpr int LAST_NA = (TAIL == 0 ? IPG : (TAIL >= 2 ? TAIL : 2));
  constexpr int NA0 = (GROUPS > 1 ? IPG : LAST_NA);
  constexpr int NA1 = (GROUPS > 2 ? IPG : LAST_NA);
  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;
  if (int(tid.x) == 0) {
    qmv_fast_crossrow_affine4_g64_wide<T, NA0, DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        0, out_row, simd_lid);
  } else if (GROUPS > 1 && int(tid.x) == 1) {
    qmv_fast_crossrow_affine4_g64_wide<T, NA1, DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        IPG, out_row, simd_lid);
  } else if (GROUPS > 2 && int(tid.x) == 2) {
    qmv_fast_crossrow_affine4_g64_wide<T, LAST_NA, DIRECT_NIBBLES>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        2 * IPG, out_row, simd_lid);
  }"""


def relax_na(text: str, bound: int) -> str:
    """Admit the probe-only NA range this arm needs, and no more."""
    if text.count(NA_ASSERT) != 1:
        raise SystemExit("e61_arms: NA assert anchor not unique")
    return text.replace(NA_ASSERT, 'static_assert(NA >= 2 && NA <= %d, '
                        '"wide multi-row QMV supports NA in [2, %d]");'
                        % (bound, bound))


def relax_m(text: str, bound: int) -> str:
    if text.count(M_ASSERT) != 1:
        raise SystemExit("e61_arms: M assert anchor not unique")
    return text.replace(M_ASSERT, 'static_assert(M >= 3 && M <= %d, '
                        '"wide multi-row QMV dispatch covers M in [3, %d]");'
                        % (bound, bound))


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
        raise SystemExit("e61_arms: case %d is not unique in the wide tier" % m)
    return text[:start] + "\n" + kept[0] + text[end:]


def swap_ipg(text: str, m: int, ipg: int) -> str:
    """Repoint `case m:` at `<T, m, ipg>` instead of its shipped IPG."""
    old = "qmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>" % (m, SHIPPED_IPG[m])
    new = "qmv_fast_crossrow_affine4_g64_m<T, %d, %d, true>" % (m, ipg)
    if text.count(old) != 1:
        raise SystemExit("e61_arms: M=%d dispatch anchor not unique" % m)
    return text.replace(old, new)


def add_ballast(text: str) -> str:
    """Dead `case 12` at <T,12,6>: register pressure with no reachable route."""
    if text.count(CASE9_TAIL) != 1:
        raise SystemExit("e61_arms: case 9 tail anchor not unique")
    return text.replace(CASE9_TAIL, CASE9_TAIL_PLUS_BALLAST)


def perturb_lanes(text: str) -> str:
    """Positive control: swap the write lanes of rows 4 and 5 when NA == 6."""
    if text.count(LANE_WRITE) != 1:
        raise SystemExit("e61_arms: lane-write anchor not unique")
    return text.replace(LANE_WRITE, LANE_WRITE_PERTURBED)


def rbx_wrapper(text: str) -> str:
    """Select the weight stream from tid.x before the branch, not after it.

    The shipped wrapper computes a runtime `first_m` and then decides the tail
    NA from it, so the wide helper carries a runtime input-row base and both
    group widths sit inside one inlined body. This form branches on tid.x
    first and hands each group a literal first input row.

    Group count, per-group NA and per-group first input row are identical to
    the shipped wrapper, so no output element changes its K accumulation order.
    """
    if text.count(WRAPPER_BODY) != 1:
        raise SystemExit("e61_arms: wrapper-body anchor not unique")
    return text.replace(WRAPPER_BODY, WRAPPER_BODY_RBX)


def _iso(m: int, ipg: int, structure: str) -> dict:
    tail = m % ipg
    groups = (m + ipg - 1) // ipg
    na = sorted({ipg} | ({max(tail, 2)} if tail else set()))
    steps: list[tuple[str, dict]] = []
    if ipg > 5:
        steps.append(("relax_na", {"bound": ipg}))
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


def _whole(m: int, ipg: int, doc: str, rbx: bool = False, **extra) -> dict:
    steps = [("relax_na", {"bound": ipg}), ("swap_ipg", {"m": m, "ipg": ipg})]
    if rbx:
        steps.append(("rbx", {}))
    return dict({
        "family": "whole_table",
        "doc": doc,
        "cell": "<T,%d,%d>" % (m, ipg),
        "width": m,
        "ipg": ipg,
        "working_groups": 1,
        "group_na_values": [ipg],
        "rbx": rbx,
        "steps": steps,
    }, **extra)


ARMS: dict[str, dict] = {
    "shipped": {
        "family": "control",
        "doc": "the d2139c92 tip, unmodified",
        "cell": None,
        "steps": [],
    },
    "t6": _whole(6, 6, "whole table, case 6 -> <T,6,6>: ONE weight stream at "
                 "M=6, table register maximum rises 129 -> 146"),
    "t7": _whole(7, 7, "whole table, case 7 -> <T,7,7>: bandwidth probe for the "
                 "lone NA=7 rate; closes M=7, M=8 and M=9",
                 never_submit=True),
    "t6_rbx": _whole(6, 6, "t6 plus the rbx wrapper: the same <T,6,6> schedule "
                     "reached by selecting the stream from tid.x before the "
                     "branch, so the wide helper takes a literal first input "
                     "row. Tests whether the NA=6 bandwidth cliff is an "
                     "occupancy cliff caused by the register jump.",
                     rbx=True),
    "shipped_rbx": {
        "family": "whole_table",
        "doc": "the rbx wrapper alone, every cell at its shipped IPG: isolates "
               "what the wrapper rewrite costs or buys with no schedule change",
        "cell": None,
        "steps": [("rbx", {})],
        "never_submit": True,
    },
    "iso_m6_ipg3": _iso(6, 3, "shipped, two NA=3 groups"),
    "iso_m6_ipg6": _iso(6, 6, "LONE NA=6"),
    "ballast": {
        "family": "ceiling_probe",
        "doc": "case 6 stays <T,6,3>; a dead case 12 at <T,12,6> holds the "
               "table register maximum at 146 with no reachable route change",
        "cell": "<T,6,3> + dead <T,12,6>",
        "steps": [("relax_na", {"bound": 6}), ("relax_m", {"bound": 12}),
                  ("ballast", {})],
        "never_submit": True,
    },
    "t6_lane_perturb": {
        "family": "positive_control",
        "doc": "t6 with rows 4 and 5 written to each other's lane in NA=6 "
               "groups: the bitwise lane check MUST fail on this",
        "cell": "<T,6,6> lane 4<->5",
        "steps": [("relax_na", {"bound": 6}), ("swap_ipg", {"m": 6, "ipg": 6}),
                  ("perturb", {})],
        "never_time": True,
    },
}

_STEPS = {
    "relax_na": lambda t, bound=5, **kw: relax_na(t, bound),
    "relax_m": lambda t, bound=9, **kw: relax_m(t, bound),
    "only_case": lambda t, m=6, **kw: only_case(t, m),
    "swap_ipg": lambda t, m=6, ipg=6, **kw: swap_ipg(t, m, ipg),
    "ballast": lambda t, **kw: add_ballast(t),
    "perturb": lambda t, **kw: perturb_lanes(t),
    "rbx": lambda t, **kw: rbx_wrapper(t),
}


def apply_arm(text: str, name: str) -> str:
    if name not in ARMS:
        raise SystemExit("e61_arms: unknown arm %s" % name)
    for step, kwargs in ARMS[name]["steps"]:
        text = _STEPS[step](text, **kwargs)
    return text


def main() -> int:
    import argparse
    import hashlib
    import json

    ap = argparse.ArgumentParser(description="apply one E61 arm in the worktree")
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
               "never_submit": bool(spec.get("never_submit")),
               "never_time": bool(spec.get("never_time")),
               "sha256": digests}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
