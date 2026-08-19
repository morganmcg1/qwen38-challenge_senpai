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
Weight streams are `ceil(M / IPG)`. Enumerating every legal IPG at every width:

    NA<=4   M=3 [3,4]  M=4 [4,2]  M=5 [3]  M=6 [3,4,2]  M=7 [4]  M=8 [4,3,2]
            M=9 [3]
            => THE SHIPPED TABLE IS STREAM-MINIMAL AT ALL SEVEN WIDTHS
    NA<=5   only M=5 and M=9 become improvable -- exactly E27's two cells

So under the live bound there is **no weight-stream win available anywhere in
the kernel**. The only stream lever is raising the bound to NA=5 at M=5 and
M=9, which is what E27 did: correct per-width table, 125 regs, a 129 shared
allocation, and it **lost 0.3321 % of score**. Independently, of 476 rival
trees exactly one has a 5->6 boundary -- ours, `ca9251b8`, rejected.

Therefore the remaining kernel axis is gated on the 108-register ceiling, which
is what makes alphonse's E44 register gate decisive rather than optional.

A sharper way to say it, which `report` makes visible: E27's table is ALSO
stream-minimal under ITS OWN ceiling of 5. Both tables are optimal for their
bound. So E27's 0.3321 % loss cannot be attributed to a wrong table at all --
it is the price of the bound itself, i.e. of registers and occupancy. That
removes the last reading in which E27 failed by mis-tuning.

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

    # --- 2. The headline: shipped table minimal at all widths under NA<=4,
    #        and exactly M=5 and M=9 improvable once the bound is 5.
    ship = census.dispatch_table("HEAD")
    na = na_ceiling("HEAD")
    if ship is None:
        fails.append("HEAD dispatch table unreadable")
    elif na is None:
        fails.append("HEAD NA ceiling unreadable from static_assert")
    else:
        if na != 4:
            fails.append("HEAD NA ceiling is %d, expected 4 -- if this changed "
                         "deliberately every stream claim needs revisiting"
                         % na)
        improv = [M for M, _, _, _, _, v in optimality(ship, 4)
                  if v != "OPTIMAL"]
        if improv:
            fails.append("shipped table NOT stream-minimal under NA<=4 at %s "
                         "-- the 'no stream win exists' claim is FALSE" % improv)
        improv5 = [M for M, _, _, _, _, v in optimality(ship, 5)
                   if v != "OPTIMAL"]
        if improv5 != [5, 9]:
            fails.append("under NA<=5 expected exactly [5, 9] improvable, "
                         "got %s" % improv5)

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
    if ship is not None:
        bad_table = dict(ship)
        bad_table[8] = 2          # 4 streams where 2 are legal
        v = [M for M, _, _, _, _, verdict in optimality(bad_table, 4)
             if verdict != "OPTIMAL"]
        if v != [8]:
            fails.append("negative control failed: a deliberately non-minimal "
                         "M=8 cell was not flagged (got %s)" % v)

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
            if sb != [(4, 5), (8, 9)]:
                fails.append("shipped boundaries are %s, expected "
                             "[(4, 5), (8, 9)]" % sb)
            if sb == [(a, b) for a, b, _, _ in bnd]:
                fails.append("shipped and E41-base boundaries are identical; "
                             "the whole scope distinction would be vacuous")

    # --- 6. Break-even arithmetic quoted to students.
    for M, want in ((5, 21.20), (9, 12.43)):
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
    print("SELFTEST PASS: legality rules, shipped table minimal at all 7 "
          "widths under NA<=4 (read from source), exactly [5, 9] improvable "
          "under NA<=5, both contrasts allocation-neutral, E41 residuals "
          "reproduce on 04ad6bf1 only, break-even taxes 21.20/12.43%.")
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
