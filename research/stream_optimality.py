#!/usr/bin/env python3
"""Is the shipped QMV dispatch table stream-optimal? Enumerated, from source.

WHY THIS IS A COMMITTED FILE AND NOT A SCRATCH SCRIPT
-----------------------------------------------------
The result below currently gates the entire remaining kernel axis of this
campaign, and for one session it existed only as a throwaway in /tmp with the
dispatch tables **hardcoded as literals**. That is the precise failure this
campaign keeps paying for: a constant quoted without its tree. So this version
reads every table from `kernels/quantized.h` at a NAMED rev, reads the NA
ceiling from the `static_assert` rather than assuming it, and self-tests.

THE RESULT
----------
An IPG is legal at width M iff `2 <= IPG <= NA_max` (the `_wide` helper's
static_assert) and `M % IPG != 1` (the `_m` helper's no-one-row-tail assert).
Weight streams are `ceil(M / IPG)`. Enumerating every legal IPG at every width,
under the ceiling this file READS from the tree rather than assumes:

    NA<=4   M=3 [3,4]  M=4 [4,2]  M=5 [3]  M=6 [3,4,2]  M=7 [4]  M=8 [4,3,2]
            M=9 [3]                       -- pre-E55 world, historical only
    NA<=5   only M=5 remains improvable   -- E55 took M=9

E55 raised the `_wide` static_assert to `NA <= 5` and moved `case 9` to
`<T,9,5,true>`, dropping width 9 from three weight streams to two. It measured
**-4.2952 %** on the candidate MTP leg against a +0.0497 % null, bitwise exact
over 512 tokens including post-EOS continuation. The shipped boundary set fell
from `[(4,5), (8,9)]` to `[(4,5)]`.

READ THIS BEFORE QUOTING THE FILE
---------------------------------
An earlier version of this docstring concluded that there is "no weight-stream
win available anywhere in the kernel" and that the only lever was gated on a
108-register ceiling, citing an E27 receipt of -0.3321 %. **All three claims
are now retracted.**

 1. E55 took exactly one of the cells this file called unreachable and won
    -4.2952 % with it.
 2. Ledger 193 retracted the E27 -0.3321 % receipt. One ranked run has a
    standard deviation of 0.756 % and a difference has 1.069 %, so a single
    -0.33 % row was never evidence of a regression.
 3. Legality is no longer the governing constraint; measured bandwidth is.
    E61 rung 1 measured the weight-stream bandwidth ladder directly on the
    scored kernel, peak 227.9 GB/s, all controls passed:

        NA   2       3       4       5       6      7
        GB/s 223.784 199.693 175.238 150.946 117.8  97.9

    So a stream removal pays only when the wider group's bandwidth clears the
    break-even the ladder sets, and that is an empirical question this file
    cannot answer. Ledger 199(D) records M=7 as a direct refutation: the model
    predicts -4.66 % and the measurement is +7.13 % SLOWER.
 4. The register ceiling cannot explain that ladder, by source construction.
    `affine_qmv_fast` is ONE `[[kernel]]` entry point (quantized.h:1869) and
    every width is a `case` of a RUNTIME `switch (ntg.x)` (:1922) that
    calls the `METAL_FUNC` helper at :1157. All widths inline into the same
    function, so the compiled register allocation, pipeline state object and
    resident simdgroup count are IDENTICAL at M=2 and M=9. Per-cell register
    counts from an isolated compile are NOT what the scored kernel runs.
    E61 rung 1b dosed +15 registers into the shared entry point at fixed
    routing and measured +0.3804 %, against a NA=5->6 step of -22 %.

WHAT THIS FILE IS FOR NOW
-------------------------
Two things, and nothing else.

 A. It enumerates which cells are stream-reducible under the live ceiling, as
    a CANDIDATE LIST for measurement. It does not price them.
 B. It is a DRIFT GUARD. The selftest pins the shipped IPG table, the NA
    ceiling and the boundary set to what the campaign currently believes. Any
    kernel change that moves them fails this gate on purpose, so that whoever
    merges it must revisit the stream claims and update this file in the same
    commit. `t55` (`case 5` -> `<T,5,5>`) and `t6` (`case 6` -> `<T,6,6>`) are
    both in flight and both will fire it.

THE SINGLE-FACTOR CONTRASTS
---------------------------
In any ceil-generated table, stream count and per-group row width move
together, so a boundary contrast cannot separate them. Two fixed-M contrasts
can, and in both the shared allocation provably cannot move because each uses
only `_wide<T,NA>` cells the shipped table already instantiates:

    A  <T,6,3> vs <T,6,4>   groups 3+3 vs 4+2   streams 2 BOTH SIDES
    B  <T,8,4> vs <T,8,3>   streams 2 -> 3      (A's positive control)

Usage:
  stream_optimality.py selftest
  stream_optimality.py report [REV...]     default: HEAD
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_dispatch_census as census  # noqa: E402

WIDTHS = range(3, 10)
# Single source of truth: the census module already owns this path. Hardcoding
# it here cost a selftest failure -- which is the correct outcome, since the
# NA-ceiling reader fails closed rather than defaulting to 4, but the lesson is
# to import the constant instead of retyping it.
QH = census.QH

# --- DRIFT-GUARD CONSTANTS -------------------------------------------------
# What the campaign currently believes about the shipped kernel, as of the
# merge of E55. Every one of these is READ back from the tree by `selftest`
# and compared, so a kernel change cannot silently invalidate a stream claim.
# When `t55` or `t6` lands, update these four together with the docstring.
NA_CEILING = 5
CANONICAL_TABLE = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5}
SHIPPED_BOUNDARIES = [(4, 5)]
# Reducible under the live ceiling.
IMPROVABLE_AT_CEILING = [5]
# Reducible if the `_wide` static_assert were raised by one. These are exactly
# the `t55` and `t6` candidates. A candidate list, NOT a prediction: M=7 was on
# the equivalent list one ceiling ago and measured +7.13 % SLOWER.
IMPROVABLE_AT_CEILING_PLUS_1 = [5, 6]

# thorfinn's E41 fit. SCOPED TO `04ad6bf1`, which is the entire point of the
# scope note in the census module: these are not universal constants.
E41_A, E41_B, E41_C = 16.432, 20.291, 11.798
E41_BASE = "04ad6bf11437c269df85a47e91faa769c74fe6da"
E41_MEASURED = {3: 72.811, 4: 82.722, 5: 96.217, 6: 128.890,
                7: 138.717, 8: 149.727, 9: 164.675}


def na_ceiling(rev):
    """Max NA allowed by the `_wide` helper's static_assert at `rev`.

    Read, not assumed: the live tree asserts `NA >= 2 && NA <= 4` and the E27
    window asserted `<= 5`, and that one character is the difference between
    "no stream win exists" and "two widths are improvable". Returns None if the
    assert cannot be found, so a parse failure is never silently a 4.
    """
    out = census.run(["git", "show", "%s:%s" % (rev, QH)])
    if out.returncode != 0:
        return None
    best = None
    for line in out.stdout.splitlines():
        if "static_assert" in line and "NA" in line and "<=" in line:
            tail = line.split("<=", 1)[1]
            digits = ""
            for ch in tail:
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break
            if digits:
                n = int(digits)
                # Take the widest assert seen; there is one in `_wide`.
                best = n if best is None else max(best, n)
    return best


def legal(M, ipg, na_max):
    """`2 <= IPG <= NA_max` and no one-row tail group."""
    return 2 <= ipg <= na_max and M % ipg != 1


def legal_ipgs(M, na_max):
    """Legal IPGs at M, ordered by resulting stream count (best first)."""
    opts = [i for i in range(2, na_max + 1) if legal(M, i, na_max)]
    return sorted(opts, key=lambda i: (math.ceil(M / i), i))


def min_streams(M, na_max):
    opts = legal_ipgs(M, na_max)
    return min(math.ceil(M / i) for i in opts) if opts else None


def groups(M, ipg):
    return [list(range(s, min(s + ipg, M))) for s in range(0, M, ipg)]


def na_cells(M, ipg):
    """`_wide<T,NA>` instantiations required by this (M, IPG) cell."""
    return sorted({len(g) for g in groups(M, ipg)})


def instantiated_cells(table):
    """Every NA cell the whole table instantiates somewhere, plus the <T,2>
    non-`_m` path which is always present."""
    used = {2}
    for M, ipg in table.items():
        used.update(na_cells(M, ipg))
    return sorted(used)


def optimality(table, na_max):
    """[(M, shipped_ipg, shipped_streams, min_streams, legal_ipgs, verdict)]"""
    rows = []
    for M in sorted(table):
        ipg = table[M]
        s = math.ceil(M / ipg)
        best = min_streams(M, na_max)
        rows.append((M, ipg, s, best, legal_ipgs(M, na_max),
                     "OPTIMAL" if best is not None and s == best
                     else "IMPROVABLE"))
    return rows


def predict(table):
    """thorfinn's E41 model evaluated on any table. Zero free parameters."""
    return {M: E41_A + E41_B * math.ceil(M / table[M]) + E41_C * M
            for M in table}


def first_diffs(T):
    ks = sorted(T)
    return [(ks[i], ks[i + 1], T[ks[i + 1]] - T[ks[i]])
            for i in range(len(ks) - 1)]


def report(revs):
    bad = 0
    for rev in revs:
        tbl = census.dispatch_table(rev)
        na = na_ceiling(rev)
        if tbl is None or na is None:
            print("UNUSABLE %s  (table=%s na_ceiling=%s)"
                  % (rev, "ok" if tbl else "MISSING", na))
            print("         a tree without both a dispatch table and a "
                  "readable NA assert cannot be judged; refusing to guess")
            bad += 1
            continue
        st = census.streams(tbl)
        print("=" * 74)
        print("%s   NA ceiling %d (read from static_assert)" % (rev, na))
        print("=" * 74)
        print("  IPG        : %s" % census.fmt(tbl))
        print("  streams    : %s" % census.fmt(st))
        print("  boundaries : %s" % (", ".join(
            "%d->%d" % (a, b) for a, b, _, _ in census.boundaries(st))
            or "none"))
        print("  NA cells instantiated: %s" % instantiated_cells(tbl))
        print()
        n_improv = 0
        for M, ipg, s, best, opts, verdict in optimality(tbl, na):
            if verdict != "OPTIMAL":
                n_improv += 1
            print("  M=%d  IPG %d -> %d streams   min %s   legal %-12s %s"
                  % (M, ipg, s, best, str(opts), verdict))
        print()
        if n_improv == 0:
            print("  => STREAM-MINIMAL AT ALL %d WIDTHS under NA<=%d."
                  % (len(tbl), na))
            print("     No weight-stream win exists without raising the bound.")
        else:
            print("  => %d width(s) improvable under NA<=%d." % (n_improv, na))
        print()
        T = predict(tbl)
        print("  E41 model on this table (coefficients scoped to %s):"
              % E41_BASE[:8])
        print("    T  : %s" % " ".join("%.3f" % T[M] for M in sorted(T)))
        print("    d1 : %s" % " ".join(
            "%.3f" % d for _, _, d in first_diffs(T)))
        d1 = first_diffs(T)
        amax = max(d1, key=lambda x: x[2])
        print("    argmax d1 : %d->%d  (%.3f)" % (amax[0], amax[1], amax[2]))
        print()
    return 1 if bad else 0


def selftest():
    fails = []

    # --- 1. Legality rules, on constructed inputs.
    if legal(7, 3, 4):
        fails.append("7 %% 3 == 1 must be illegal (one-row tail group)")
    if not legal(6, 4, 4):
        fails.append("6 %% 4 == 2 must be legal")
    if legal(5, 5, 4):
        fails.append("IPG 5 must be illegal under NA<=4")
    if not legal(5, 5, 5):
        fails.append("IPG 5 must be legal under NA<=5")
    if legal(4, 3, 4):
        fails.append("4 %% 3 == 1 must be illegal")

    # --- 2. DRIFT GUARD. These four expectations describe the post-E55 world.
    #        `t55` and `t6` will each fire this block on purpose; update the
    #        constants and the docstring in the SAME commit that lands them.
    ship = census.dispatch_table("HEAD")
    na = na_ceiling("HEAD")
    if ship is None:
        fails.append("HEAD dispatch table unreadable")
    elif na is None:
        fails.append("HEAD NA ceiling unreadable from static_assert")
    else:
        if na != NA_CEILING:
            fails.append("HEAD NA ceiling is %d, expected %d -- if this "
                         "changed deliberately every stream claim needs "
                         "revisiting" % (na, NA_CEILING))
        if ship != CANONICAL_TABLE:
            fails.append("shipped IPG table is %s, expected %s -- a kernel "
                         "change moved it; re-measure the bandwidth ladder "
                         "before quoting any stream claim"
                         % (dict(sorted(ship.items())),
                            dict(sorted(CANONICAL_TABLE.items()))))
        # Minimal under the ceiling the tree actually declares. Judging the
        # live table against a stale literal ceiling is what made this gate
        # fail six ways after E55.
        improv = [M for M, _, _, _, _, v in optimality(ship, na)
                  if v != "OPTIMAL"]
        if improv != IMPROVABLE_AT_CEILING:
            fails.append("under NA<=%d expected exactly %s improvable, got %s"
                         % (na, IMPROVABLE_AT_CEILING, improv))
        # Raising the ceiling by one opens the next candidate list. This is a
        # candidate list for MEASUREMENT, never a prediction of a win: M=7 was
        # on it and measured +7.13 % slower.
        improv_next = [M for M, _, _, _, _, v in optimality(ship, na + 1)
                       if v != "OPTIMAL"]
        if improv_next != IMPROVABLE_AT_CEILING_PLUS_1:
            fails.append("under NA<=%d expected exactly %s improvable, got %s"
                         % (na + 1, IMPROVABLE_AT_CEILING_PLUS_1, improv_next))

        # --- 3. Both contrasts need no new NA cell => allocation cannot move.
        used = instantiated_cells(ship)
        for M, alt in ((6, 4), (8, 3)):
            need = na_cells(M, alt)
            new = [n for n in need if n not in used]
            if new:
                fails.append("contrast M=%d IPG %d needs NEW cells %s; the "
                             "'allocation cannot move' claim is FALSE"
                             % (M, alt, new))
        # Contrast A must be stream-NEUTRAL, contrast B must be +1.
        if math.ceil(6 / 3) != math.ceil(6 / 4):
            fails.append("contrast A is not stream-neutral")
        if math.ceil(8 / 3) - math.ceil(8 / 4) != 1:
            fails.append("contrast B does not add exactly one stream")

    # --- 4. NEGATIVE CONTROL: a fabricated non-minimal table must be caught.
    #        Without this, optimality() returning "OPTIMAL" unconditionally
    #        would pass every check above.
    if ship is not None and na is not None:
        bad_table = dict(ship)
        bad_table[8] = 2          # 4 streams where 2 are legal
        want_bad = sorted(set(IMPROVABLE_AT_CEILING) | {8})
        v = [M for M, _, _, _, _, verdict in optimality(bad_table, na)
             if verdict != "OPTIMAL"]
        if v != want_bad:
            fails.append("negative control failed: a deliberately non-minimal "
                         "M=8 cell was not flagged (wanted %s, got %s)"
                         % (want_bad, v))

    # --- 5. The E41 fit must reproduce his residuals ON HIS BASE, and only
    #        there. This is what makes the scoping claim checkable.
    e27 = census.dispatch_table(E41_BASE)
    if e27 is None:
        fails.append("E41 base %s dispatch table unreadable" % E41_BASE[:8])
    else:
        T = predict(e27)
        resid = [E41_MEASURED[M] - T[M] for M in WIDTHS]
        mx = max(abs(r) for r in resid)
        if mx > 1.7:
            fails.append("E41 residuals do not reproduce on his base: "
                         "max|r| = %.3f, expected <= 1.7" % mx)
        bnd = census.boundaries(census.streams(e27))
        if [(a, b) for a, b, _, _ in bnd] != [(5, 6)]:
            fails.append("E41 base boundary is %s, expected exactly [(5, 6)]"
                         % [(a, b) for a, b, _, _ in bnd])
        if ship is not None:
            sb = [(a, b) for a, b, _, _ in census.boundaries(
                census.streams(ship))]
            if sb != SHIPPED_BOUNDARIES:
                fails.append("shipped boundaries are %s, expected %s"
                             % (sb, SHIPPED_BOUNDARIES))
            if sb == [(a, b) for a, b, _, _ in bnd]:
                fails.append("shipped and E41-base boundaries are identical; "
                             "the whole scope distinction would be vacuous")

    # --- 6. Break-even arithmetic quoted to students.
    for M, want in ((5, 21.20), (9, 14.20)):
        if ship is None:
            break
        s = math.ceil(M / ship[M])
        now = E41_A + E41_B * s + E41_C * M
        then = E41_A + E41_B * (s - 1) + E41_C * M
        tax = 100.0 * (now - then) / then
        if abs(tax - want) > 0.01:
            fails.append("break-even tax at M=%d is %.2f%%, quoted %.2f%%"
                         % (M, tax, want))

    if fails:
        print("SELFTEST FAIL (%d)" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("SELFTEST PASS: legality rules; NA ceiling %d and the shipped IPG "
          "table both read from source and unchanged; exactly %s improvable "
          "at the live ceiling and %s one above it; shipped boundaries %s; "
          "both contrasts allocation-neutral; E41 residuals reproduce on "
          "04ad6bf1 only; break-even taxes 21.20/14.20%%."
          % (NA_CEILING, IMPROVABLE_AT_CEILING, IMPROVABLE_AT_CEILING_PLUS_1,
             SHIPPED_BOUNDARIES))
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        return selftest()
    if mode == "report":
        return report(sys.argv[2:] or ["HEAD"])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
