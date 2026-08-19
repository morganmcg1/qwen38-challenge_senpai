#!/usr/bin/env python3
"""E48 - dose-matched uniform QMV regression (Arm U), on the fb0a09d dispatch table.

Reuses E42's bit-exact redundant-pass injection verbatim (`research/e42_perturb.py`)
and changes exactly two things:

1. **The dispatch table.** E42 ran on `04ad6bf1`, whose IPG row was `3 4 5 3 4 4 5`.
   This base is `3 4 3 3 4 4 3` with `static_assert(NA >= 2 && NA <= 4)`. The case
   anchors are literal instantiation text, so E42's tables do not match here.

2. **Independent doses per kernel family.** E42's arms treated one family at one
   level. Arm U has to treat BOTH families, and the pass loop is not uniform in the
   fractional cost it injects: it covers ~90 % of a crossrow kernel but only ~43 %
   of `qmv_fast_impl` (E42 measured x = 0.9030 vs 0.4271 per extra pass). Applying
   the same level everywhere is a 2.11x over-dose of the candidate leg and would
   return a confidently wrong sign. So the two families take separate levels, and
   a matched dose needs `m1_level / crossrow_level ~= 0.9030/0.4271 = 2.114`.

   Levels are integers, so no arm matches exactly, and two arms at the same ratio
   would make the two-unknown MTP-leg system singular. The arms therefore sit at
   ratios 2 and 1.5: both under-dose width 1, which is the direction that makes a
   rise in raw_p a one-sided proof rather than an estimate.

Nothing here ships. Every arm is a deliberate regression; the branch ends with both
twins byte-identical to the base.

Usage
-----
    research/e48_perturb.py --crossrow-level 1 --m1-level 2   # Arm U-lo, x1/xX ~ 0.95
    research/e48_perturb.py --crossrow-level 2 --m1-level 3   # Arm U-hi, x1/xX ~ 0.71
    research/e48_perturb.py --revert
    research/e48_perturb.py --self-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e42_perturb import (  # noqa: E402
    ECHO,
    LVL,
    M1_EDITS,
    STRUCT_EDITS,
    apply_edits,
    base_header,
    revert,
    write_twins,
    _strip_markers,
)

BASE_SHA = "fb0a09d3912477d94ed631bdb90fd04172d7b4cf"

# Read off the dispatch switch at kernels/quantized.h:1924-1977 (out_vec_size >= 4096).
IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

HI_CASE2 = """      // promoted pair kernel is kept there byte-for-byte.
      switch (ntg.x) {
        case 2:
          qmv_fast_crossrow_affine4_g64<T, 2@ECHO@@LVL@>("""
CASE_REPLACEMENTS = {2: HI_CASE2}
for _w, _ipg in IPG.items():
    CASE_REPLACEMENTS[_w] = f"qmv_fast_crossrow_affine4_g64_m<T, {_w}, {_ipg}, true@LVL@>("
LO_CASE_REPLACEMENTS: dict[int, str | None] = {2: None}
for _w in range(3, 10):
    LO_CASE_REPLACEMENTS[_w] = f"qmv_fast_crossrow_affine4_g64<T, {_w}@ECHO@@LVL@>("

CASE_ANCHORS = {w: _strip_markers(t) for w, t in CASE_REPLACEMENTS.items()}
LO_CASE_ANCHORS = {w: (_strip_markers(t) if t else None) for w, t in LO_CASE_REPLACEMENTS.items()}

CROSSROW_WIDTHS = range(2, 10)


def build_arm(base_text: str, crossrow_level: int, m1_level: int) -> str:
    """Patch the header for one Arm U cell. Level 0 leaves that family untreated."""
    text = apply_edits(base_text, STRUCT_EDITS)
    if crossrow_level:
        lvl = f", {crossrow_level}"
        # The >= 4096 tier is patched first: the two tiers share the `case 2:` body
        # verbatim, so the lower tier's anchor only becomes unique afterwards.
        tiers = ((CASE_ANCHORS, CASE_REPLACEMENTS), (LO_CASE_ANCHORS, LO_CASE_REPLACEMENTS))
        for anchors, replacements in tiers:
            for width in CROSSROW_WIDTHS:
                anchor = anchors[width]
                if anchor is None:
                    anchor = CASE_ANCHORS[2].split("\n")[-1]
                    replacement = CASE_REPLACEMENTS[2].split("\n")[-1]
                else:
                    replacement = replacements[width]
                replacement = replacement.replace(ECHO, ", 2").replace(LVL, lvl)
                if text.count(anchor) != 1:
                    raise SystemExit(
                        f"e48_perturb: width {width} anchor matched {text.count(anchor)} "
                        f"times: {anchor[:80]!r}"
                    )
                text = text.replace(anchor, replacement)
    text = apply_edits(text, M1_EDITS)
    text = text.replace("E42_M1_PASSES", str(m1_level))
    return text


def self_test(base_sha: str) -> None:
    base = base_header(base_sha)
    checks = 0
    for crossrow_level, m1_level in [(0, 0), (1, 2), (2, 4), (1, 0), (0, 2)]:
        text = build_arm(base, crossrow_level, m1_level)
        assert text != base
        # one repeat loop per treated kernel body: pair, wide, qmv_fast_impl
        assert text.count("for (int e42_pass = 0; e42_pass <= E42_PASSES; e42_pass++)") == 3
        assert f"qmv_fast_impl<T, group_size, bits, {m1_level}>(" in text
        # two untreated sites survive: the affine_qmv_fast fall-through for M != 1,
        # and gather_qmv_fast, which the dense scored path never dispatches
        assert text.count("qmv_fast_impl<T, group_size, bits>(") == 2
        checks += 4
        for w in CROSSROW_WIDTHS:
            treated = bool(crossrow_level)
            expect = CASE_REPLACEMENTS[w].replace(ECHO, ", 2").replace(LVL, f", {crossrow_level}")
            hi = expect if treated else CASE_ANCHORS[w]
            assert hi in text, (crossrow_level, w, "hi")
            lo = LO_CASE_REPLACEMENTS[w]
            if lo is not None:
                lo = lo.replace(ECHO, ", 2").replace(LVL, f", {crossrow_level}") if treated else _strip_markers(lo)
                assert lo in text, (crossrow_level, w, "lo")
            if treated:
                assert CASE_ANCHORS[w] not in text, (crossrow_level, w, "residual base cell")
            checks += 2
        # CrossrowGate parses inputs_per_group out of template argument 3
        for site in re.findall(r"qmv_fast_crossrow_affine4_g64<T, \d+[^>]*>", text):
            args = [a.strip() for a in site.split("<")[1].rstrip(">").split(",")]
            assert len(args) == 2 or args[2] == "2", site
            checks += 1
        # the accumulation expression, the reduction and the store are untouched
        assert "const float reduced = simd_sum(acc[r][m]);" in text
        assert "acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];" in text
        assert text.count("y[(first_m + m) * out_vec_size + out_row + r] =") == 1
        checks += 3
    # the two dosed arms and the null arm are all distinct source
    null = build_arm(base, 0, 0)
    assert build_arm(base, 1, 2) != build_arm(base, 2, 4) != null
    # the null arm carries the full structural churn but no extra passes anywhere
    assert "E42_PASSES = 0" in null and ", 1>(" not in null.split("switch (ntg.x)")[1][:4000]
    checks += 2
    print(f"E48 PERTURB SELF-TEST PASSED ({checks} checks)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossrow-level", type=int, default=0)
    ap.add_argument("--m1-level", type=int, default=0)
    ap.add_argument("--base-sha", default=BASE_SHA)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.base_sha)
        return 0
    if args.revert:
        revert(args.base_sha)
        print("e48_perturb: twins restored to base")
        return 0
    if args.crossrow_level < 0 or args.m1_level < 0:
        ap.error("levels must be >= 0")
    revert(args.base_sha)
    text = build_arm(base_header(args.base_sha), args.crossrow_level, args.m1_level)
    write_twins(text)
    print(
        f"e48_perturb: crossrow_level={args.crossrow_level} m1_level={args.m1_level} "
        "twins patched and locked"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
