#!/usr/bin/env python3
"""E59 arm definitions: can `<T,5,5>` be reached under the 108-register floor?

E54 measured the lone NA=5 cell `<T,5,5>` at -20.253 % in isolation, and E54's
census measured it at 125 registers against a shipped kernel maximum of 108. So
the fastest cell in the table cannot ship at `rows_per_simd = 4`.

187(P) proves 108 is a LEGALITY floor rather than a tuning choice: M=7 has no
legal accumulator count below 4, and its only legal split {4,3} is mixed, so the
shipped table already sits on its floor. Dropping `rows_per_simd` is the only
lever left.

The wall is coverage. The host grid is IPG-blind and frozen
(`backend/metal/quantized.cpp`: `group_dims(32,2,1)`,
`grid_dims(M, (N+7)/8, B)`), so two simdgroups must cover 8 output rows per
`tid.y`. At `rows_per_simd = 2` a naive `out_row` leaves four rows unwritten.

E59 builds TWO bit-exact coverings, because they place the same duplicated `x`
traffic differently and only measurement can rank them:

  rb2   two SEQUENTIAL row blocks in one x-group (the advisor's indexing)
        out_row = tid.y*8 + simd_gid*4 + rb*2,  rb in {0,1}
  rbx   two PARALLEL row blocks in two x-groups
        out_row = tid.y*8 + tid.x*4  + simd_gid*2,  tid.x in {0,1}

`rbx` exists because the frozen grid launches M threadgroups in x and a
single-working-group width leaves M-1 of them returning immediately: at M=5 the
machine already pays to launch four idle threadgroups. Both mappings write
{0..7} exactly once per tile (`research/e59_coverage_proof.py`), and both keep
every output row inside one simdgroup over the same K in the same order, so both
are bit-exact by the same argument.

Arms:

  shipped                 the tip, unmodified
  m5_rb2 / m5_rbx         real table, case 5 -> <T,5,5> at rows_per_simd = 2
  ceil_only               real table plus an UNREACHABLE case 10 -> <T,10,5> at
                          rows_per_simd = 4, which pays E27's M=5 register dose
                          and is never dispatched
  iso_*                   one case kept in the wide tier, for isolated cell
                          timing: shipped <T,5,3>, <T,5,5> at r=4, and both
                          r=2 mappings
  *_lane_perturb          positive control: rows 3 and 4 swap accumulator lanes
  *_coverage_drop         positive control: the second row block is dropped, so
                          four of every eight output rows are never written

Every edit is applied to the readable header AND its runtime-effective generated
twin. The callers unwind both files on every exit path, so the branch's scored
surface stays byte-identical to the campaign base between legs.
"""

from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e54_arms import (  # noqa: E402
    HEADER,
    SHIPPED_IPG,
    SOURCES,
    only_case,
    perturb_lanes,
    relax_asserts,
    swap_ipg,
)

REPO = HERE.parent

WIDE_SIGNATURE = ("template <typename T, int NA, bool DIRECT_NIBBLES = false>\n"
                  "METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide(")
WIDE_SIGNATURE_RPS = (
    "template <typename T, int NA, bool DIRECT_NIBBLES = false,\n"
    "          int ROWS_PER_SIMD = 4>\n"
    "METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide(")

WIDE_PROLOGUE = ('  static_assert(NA >= 2 && NA <= 5, '
                 '"wide multi-row QMV supports NA in [2, 5]");\n'
                 "  typedef vec<float, NA> VF;\n"
                 "  constexpr int rows_per_simd = 4;")
WIDE_PROLOGUE_RPS = (
    '  static_assert(NA >= 2 && NA <= 5, '
    '"wide multi-row QMV supports NA in [2, 5]");\n'
    "  static_assert(ROWS_PER_SIMD == 2 || ROWS_PER_SIMD == 4,\n"
    '                "one simdgroup covers its four rows in one or two blocks");\n'
    "  typedef vec<float, NA> VF;\n"
    "  constexpr int rows_per_simd = ROWS_PER_SIMD;")

M_ASSERT = ('static_assert(M >= 3 && M <= 9, '
            '"wide multi-row QMV dispatch covers M in [3, 9]");')
M_ASSERT_10 = ('static_assert(M >= 3 && M <= 10, '
               '"wide multi-row QMV dispatch covers M in [3, 10]");')

WRAPPER_ANCHOR = ("template <typename T, int group_size, int bits>\n"
                  "METAL_FUNC void qmv_impl(")

WRAPPER_RB2 = """\
// Two-row-block form of the wide multi-row dispatch, for the one width whose
// cheapest legal accumulator width is a LONE NA=5 group. At rows_per_simd = 4
// that group holds four vec<float,5> accumulators and censuses at 125
// registers, above the shipped table's 108-register legality floor. Halving
// rows_per_simd halves the accumulator array. The host grid is IPG-blind and
// frozen, so each tid.y still owns eight output rows and two simdgroups, and
// each simdgroup covers its four rows as two sequential blocks: simdgroup 0
// writes {0,1} then {2,3}, simdgroup 1 writes {4,5} then {6,7}.
//
// Exact: every output row is still reduced inside ONE simdgroup, over the same
// K, in the same order, by the same expression, so splitting the row loop
// cannot reassociate any scalar chain. The cost is one extra pass over the NA
// activation rows.
template <typename T, int M, int IPG, bool DIRECT_NIBBLES = false>
METAL_FUNC void qmv_fast_crossrow_affine4_g64_m_rb2(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(M >= 3 && M <= 9, "wide multi-row QMV dispatch covers M in [3, 9]");
  static_assert(M % IPG == 0, "the row-block route runs no tail group");
  const int first_m = int(tid.x) * IPG;
  if (first_m >= M) {
    return;
  }
  for (int rb = 0; rb < 2; ++rb) {
    qmv_fast_crossrow_affine4_g64_wide<T, IPG, DIRECT_NIBBLES, 2>(
        w, scales, biases, x, y, in_vec_size, out_vec_size,
        first_m, int(tid.y) * 8 + int(simd_gid) * 4 + rb * 2, simd_lid);
  }
}

"""

WRAPPER_RBX = """\
// The same halved accumulator, with the two row blocks placed in two x-groups
// instead of two sequential passes. The frozen host grid launches M
// threadgroups in x. A width with one working group (M % IPG == 0 and
// M / IPG == 1) returns immediately from M - 1 of them, so the machine already
// pays to launch threadgroups that do no work. Two of them carry one row block
// each: tid.x 0 writes {0,1} and {2,3} through its two simdgroups, tid.x 1
// writes {4,5} and {6,7}. Same eight rows, each written exactly once, same
// per-row reduction, therefore bit-identical to the four-row form.
template <typename T, int M, int IPG, bool DIRECT_NIBBLES = false>
METAL_FUNC void qmv_fast_crossrow_affine4_g64_m_rbx(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  static_assert(M >= 3 && M <= 9, "wide multi-row QMV dispatch covers M in [3, 9]");
  static_assert(M % IPG == 0 && M / IPG == 1,
                "the x-group row-block route needs exactly one working group");
  if (int(tid.x) >= 2) {
    return;
  }
  qmv_fast_crossrow_affine4_g64_wide<T, IPG, DIRECT_NIBBLES, 2>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      0, int(tid.y) * 8 + int(tid.x) * 4 + int(simd_gid) * 2, simd_lid);
}

"""

CASE5_SHIPPED = "qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>"
CASE5_RB2 = "qmv_fast_crossrow_affine4_g64_m_rb2<T, 5, 5, true>"
CASE5_RBX = "qmv_fast_crossrow_affine4_g64_m_rbx<T, 5, 5, true>"

CASE9_ANCHOR = """\
        case 9:
          qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
"""
CASE10 = """\
        case 10:
          // UNREACHABLE ceiling probe. The scored session verifies at most one
          // primary token plus eight drafts, so ntg.x never reaches 10. This
          // case exists to pay <T,5,5>'s rows_per_simd = 4 register dose
          // WITHOUT dispatching it, which separates a shared-allocation tax
          // from an in-round cell win. Research arm only; never submitted.
          qmv_fast_crossrow_affine4_g64_m<T, 10, 5, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
"""

RB2_LOOP = "for (int rb = 0; rb < 2; ++rb) {"
RB2_LOOP_DROPPED = "for (int rb = 0; rb < 1; ++rb) {"
RBX_GUARD = "if (int(tid.x) >= 2) {"
RBX_GUARD_DROPPED = "if (int(tid.x) >= 1) {"


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit("e59_arms: %s anchor is not unique (%d)"
                         % (what, text.count(old)))
    return text.replace(old, new)


def add_rps(text: str) -> str:
    """Give the wide helper a defaulted rows_per_simd template parameter.

    Defaulted, so every existing instantiation keeps rows_per_simd = 4 and the
    untouched widths compile to the same code. The census checks that claim
    rather than assuming it.
    """
    text = relax_asserts(text)
    text = _replace_once(text, WIDE_SIGNATURE, WIDE_SIGNATURE_RPS,
                         "wide signature")
    return _replace_once(text, WIDE_PROLOGUE, WIDE_PROLOGUE_RPS,
                         "wide prologue")


def add_wrapper(text: str, kind: str) -> str:
    body = {"rb2": WRAPPER_RB2, "rbx": WRAPPER_RBX}[kind]
    return _replace_once(text, WRAPPER_ANCHOR, body + WRAPPER_ANCHOR,
                         "wrapper insertion point")


def route_case5(text: str, kind: str) -> str:
    new = {"rb2": CASE5_RB2, "rbx": CASE5_RBX}[kind]
    return _replace_once(text, CASE5_SHIPPED, new, "case 5 dispatch")


def add_case10(text: str) -> str:
    text = _replace_once(text, M_ASSERT, M_ASSERT_10, "M range assert")
    return _replace_once(text, CASE9_ANCHOR, CASE9_ANCHOR + CASE10,
                         "case 9 block")


def drop_coverage(text: str, kind: str) -> str:
    """Positive control: leave four of every eight output rows unwritten."""
    old, new = {"rb2": (RB2_LOOP, RB2_LOOP_DROPPED),
                "rbx": (RBX_GUARD, RBX_GUARD_DROPPED)}[kind]
    return _replace_once(text, old, new, "%s coverage" % kind)


_STEPS = {
    "rps": lambda t, **kw: add_rps(t),
    "relax": lambda t, **kw: relax_asserts(t),
    "wrapper": lambda t, kind="rb2", **kw: add_wrapper(t, kind),
    "route5": lambda t, kind="rb2", **kw: route_case5(t, kind),
    "case10": lambda t, **kw: add_case10(t),
    "only_case": lambda t, m=5, **kw: only_case(t, m),
    "swap_ipg": lambda t, m=5, ipg=5, **kw: swap_ipg(t, m, ipg),
    "perturb": lambda t, **kw: perturb_lanes(t),
    "drop": lambda t, kind="rb2", **kw: drop_coverage(t, kind),
}


def _route(kind: str) -> list[tuple[str, dict]]:
    return [("rps", {}), ("wrapper", {"kind": kind}), ("route5", {"kind": kind})]


ARMS: dict[str, dict] = {
    "shipped": {
        "family": "control",
        "doc": "the tip, unmodified",
        "cell": None,
        "steps": [],
    },
    "m5_rb2": {
        "family": "candidate",
        "doc": "real table, case 5 -> <T,5,5> at rows_per_simd = 2, two "
               "sequential row blocks in one x-group",
        "cell": "<T,5,5> r=2 rb2",
        "steps": _route("rb2"),
    },
    "m5_rbx": {
        "family": "candidate",
        "doc": "real table, case 5 -> <T,5,5> at rows_per_simd = 2, two "
               "parallel row blocks in two x-groups",
        "cell": "<T,5,5> r=2 rbx",
        "steps": _route("rbx"),
    },
    "ceil_only": {
        "family": "control",
        "doc": "real table plus an unreachable case 10 -> <T,10,5> at "
               "rows_per_simd = 4: E27's M=5 register dose, never dispatched",
        "cell": "<T,10,5> unreachable",
        "steps": [("relax", {}), ("case10", {})],
    },
    "iso_m5_ipg3": {
        "family": "isolated",
        "doc": "only case 5 in the wide tier, shipped <T,5,3>",
        "cell": "<T,5,3>",
        "steps": [("only_case", {"m": 5})],
    },
    "iso_m5_ipg5_r4": {
        "family": "isolated",
        "doc": "only case 5, <T,5,5> at rows_per_simd = 4: E54's lone NA=5 cell",
        "cell": "<T,5,5> r=4",
        "steps": [("relax", {}), ("only_case", {"m": 5}),
                  ("swap_ipg", {"m": 5, "ipg": 5})],
    },
    "iso_m5_ipg5_rb2": {
        "family": "isolated",
        "doc": "only case 5, <T,5,5> at rows_per_simd = 2, sequential blocks",
        "cell": "<T,5,5> r=2 rb2",
        "steps": _route("rb2") + [("only_case", {"m": 5})],
    },
    "iso_m5_ipg5_rbx": {
        "family": "isolated",
        "doc": "only case 5, <T,5,5> at rows_per_simd = 2, x-group blocks",
        "cell": "<T,5,5> r=2 rbx",
        "steps": _route("rbx") + [("only_case", {"m": 5})],
    },
    "m5_rb2_lane_perturb": {
        "family": "positive_control",
        "doc": "m5_rb2 with rows 3 and 4 written to each other's accumulator "
               "lane in NA=5 groups: the bitwise check MUST fail at M=5",
        "cell": "<T,5,5> r=2 rb2, lanes 3<->4",
        "steps": _route("rb2") + [("perturb", {})],
        "never_time": True,
    },
    "m5_rb2_coverage_drop": {
        "family": "positive_control",
        "doc": "m5_rb2 with the second row block dropped: rows 2,3,6,7 of every "
               "tile are never written, so the bitwise check MUST fail at M=5",
        "cell": "<T,5,5> r=2 rb2, one block",
        "steps": _route("rb2") + [("drop", {"kind": "rb2"})],
        "never_time": True,
    },
    "m5_rbx_coverage_drop": {
        "family": "positive_control",
        "doc": "m5_rbx with the second x-group dropped: rows 4..7 of every tile "
               "are never written, so the bitwise check MUST fail at M=5",
        "cell": "<T,5,5> r=2 rbx, one block",
        "steps": _route("rbx") + [("drop", {"kind": "rbx"})],
        "never_time": True,
    },
}

# Which wrapper and IPG each width case reaches, read from the patched text.
CASE_CALL = re.compile(
    r"        case (?P<m>\d+):\n(?:(?!        case ).)*?"
    r"(?P<fn>qmv_fast_crossrow_affine4_g64(?:_m|_m_rb2|_m_rbx)?)"
    r"<T, (?P<a>\d+)(?:, (?P<b>\d+))?", re.S)


def routing_table(text: str) -> dict[int, dict]:
    """Per-width dispatch of the `out_vec_size >= 4096` tier."""
    tier = text.index("if (out_vec_size >= 4096) {")
    end = text.index("        default:\n          break;\n      }\n    } else {",
                     tier)
    out: dict[int, dict] = {}
    for hit in CASE_CALL.finditer(text[tier:end]):
        m = int(hit.group("m"))
        ipg = int(hit.group("b")) if hit.group("b") else int(hit.group("a"))
        out[m] = {"wrapper": hit.group("fn"), "ipg": ipg,
                  "rows_per_simd": 2 if hit.group("fn").endswith(("rb2", "rbx"))
                  else 4}
    return out


def apply_arm(text: str, name: str) -> str:
    if name not in ARMS:
        raise SystemExit("e59_arms: unknown arm %s" % name)
    for step, kwargs in ARMS[name]["steps"]:
        text = _STEPS[step](text, **kwargs)
    return text


def main() -> int:
    import argparse
    import hashlib
    import json

    ap = argparse.ArgumentParser(description="apply one E59 arm in the worktree")
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--out", help="write the patched-file digests here")
    ap.add_argument("--dry-run", action="store_true",
                    help="apply in memory only and report the digests")
    args = ap.parse_args()

    digests = {}
    routing = None
    for path in SOURCES:
        text = apply_arm(path.read_text(), args.arm)
        if path == HEADER:
            routing = routing_table(text)
        if not args.dry_run:
            path.write_text(text)
        digests[str(path.relative_to(REPO))] = hashlib.sha256(
            text.encode()).hexdigest()
    spec = ARMS[args.arm]
    payload = {"arm": args.arm, "doc": spec["doc"], "family": spec["family"],
               "cell": spec["cell"], "dry_run": bool(args.dry_run),
               "routing": {str(k): v for k, v in sorted(routing.items())},
               "sha256": digests}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
