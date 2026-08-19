#!/usr/bin/env python3
"""Can the shipped QMV register ceiling be LOWERED, not merely held?

CORRECTED. The first version of this tool used a shipped kernel maximum of 104
and was wrong. Both students measured 108 (thorfinn E54 `shipped`; askeladd E55
`base`). The error was mine: I priced every cell on the bare r=4 affine ladder
`20 + 21*NA` and omitted a mixed-group term.

askeladd's E55 law supplies it and reproduces all six observed configurations
with zero residual and no parameter fitted by him:

    peak_live_regs = 20 + 21*max(NA) + 4*[two distinct NA group sizes]

The `+4` is read from a second field of the census itself -- `peak_live_values`
is 41 for every uniform group set and 45 for every mixed one, tracking `allocas`
1 against 2 -- so it is the cost of the second accumulator array, not a free
constant.

This also settles a disagreement between the two students. thorfinn published
the same +4 as an INSTRUMENT OVER-COUNT on mixed-NA cells, exact on
single-group cells. Under askeladd's law the instrument was right on both:
mixed cells genuinely cost +4 more. thorfinn's caveat should be retracted, and
his `<T,7,4>` "true 104" is really 108.

The structural conclusion is unchanged and in fact strengthened: 108 is a
legality floor pinned by M=7, no retabling can lower it, and thorfinn's r=2
row-block route fits under it with more headroom than I originally claimed.

Sources
  E55     reg = 20 + 21*max(NA) + 4*[mixed], six exact configurations
  E54     shipped kernel maximum 108, entry 163
  183(C)  r=2 affine ladder 16 + 15*NA
  E44     competing r=2 fit      15 + 17*NA
  183(B)  one library, one pipeline, one allocation for all M = 1..9
  quantized.h:1169  static_assert(M % IPG != 1)
  quantized.h:980   static_assert(NA >= 2 && NA <= 4)
"""

import argparse
import math

M_MIN, M_MAX = 3, 9
NA_MIN = 2

SHIPPED = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}

# E55 register law parameters. None of these were fitted here.
BASE_REGS, PER_NA, MIXED_PENALTY = 20, 21, 4

# E55's six exact census observations, keyed by the sorted distinct NA sizes.
CENSUS = {
    (3,): 83, (4,): 104, (5,): 125,
    (2, 3): 87, (3, 4): 108, (4, 5): 129,
}

MEASURED_SHIPPED_MAX = 108   # thorfinn E54 and askeladd E55 agree
MEASURED_E27_MAX = 129       # e27_full, and E55 candidate

# Competing r=2 intercepts; 187(L) carries both as a band.
LADDER_R2_183C = (16, 15)
LADDER_R2_E44 = (15, 17)


def regs(groups):
    return (BASE_REGS + PER_NA * max(groups)
            + (MIXED_PENALTY if len(set(groups)) > 1 else 0))


def groups_for(m, ipg):
    """M splits into floor(M/IPG) groups of IPG plus a tail of M % IPG."""
    tail = m % ipg
    g = [ipg] * (m // ipg)
    if tail:
        g.append(tail)
    return g


def legal(m, ipg):
    """quantized.h:1169 forbids a tail of 1; every group needs NA >= 2."""
    if not 1 <= ipg <= m or m % ipg == 1:
        return False
    return min(groups_for(m, ipg)) >= NA_MIN


def legal_ipgs(m):
    return [i for i in range(1, m + 1) if legal(m, i)]


def r2_regs(ladder, na):
    a, b = ladder
    return a + b * na


def report():
    print("=" * 74)
    print("1. The E55 register law against its six observations")
    print("=" * 74)
    for key in sorted(CENSUS, key=lambda k: (len(k), k)):
        obs, pred = CENSUS[key], regs(list(key))
        kind = "mixed" if len(key) > 1 else "uniform"
        print(f"   NA groups {str(list(key)):<10} {kind:<8}"
              f" obs {obs:>4}  pred {pred:>4}  resid {obs-pred:+d}")

    print()
    print("=" * 74)
    print("2. The shipped table, and the maximum I got wrong")
    print("=" * 74)
    mx = 0
    for m in range(M_MIN, M_MAX + 1):
        g = groups_for(m, SHIPPED[m])
        r = regs(g)
        mx = max(mx, r)
        print(f"   M={m} IPG={SHIPPED[m]} groups {str(g):<14}"
              f" {'mixed' if len(set(g)) > 1 else 'uniform':<8} regs {r:>4}")
    print(f"\n   shipped kernel maximum under the law = {mx}")
    print(f"   measured by thorfinn (E54) and askeladd (E55) = "
          f"{MEASURED_SHIPPED_MAX}")
    print("   my first version of this tool said 104, omitting the mixed term")

    print()
    print("=" * 74)
    print("3. Every legal configuration, and the cheapest reachable per width")
    print("=" * 74)
    floors = {}
    for m in range(M_MIN, M_MAX + 1):
        best = min(((regs(groups_for(m, i)), groups_for(m, i))
                    for i in legal_ipgs(m) if min(groups_for(m, i)) >= NA_MIN),
                   key=lambda t: t[0])
        floors[m] = best
        print(f"   M={m} legal IPG {str(legal_ipgs(m)):<22}"
              f" cheapest {str(best[1]):<14} regs {best[0]:>4}")

    floor = max(v[0] for v in floors.values())
    binding = [m for m, v in floors.items() if v[0] == floor]
    print(f"\n   lowest kernel maximum reachable by any legal retabling"
          f" = {floor}")
    print(f"   binding width(s) = {binding}")

    print()
    print("=" * 74)
    print("4. Verdict on lowering the ceiling")
    print("=" * 74)
    if floor >= MEASURED_SHIPPED_MAX:
        print("   The ceiling CANNOT be lowered by retabling. The shipped")
        print(f"   maximum {MEASURED_SHIPPED_MAX} already equals the legal"
              f" floor {floor},")
        print(f"   pinned by M={binding[0]}.")
        for i in range(NA_MIN, 4):
            m = binding[0]
            why = ("tail of 1" if m % i == 1
                   else f"tail {m % i} below NA_MIN")
            print(f"     M={m} at IPG={i}: {m} % {i} = {m % i} -> illegal"
                  f" ({why})")
        print("   M=4, M=6 and M=8 could each drop to 62, but that lowers")
        print("   nothing while adding a weight stream to each width.")

    print()
    print("=" * 74)
    print("5. The r=2 row-block route against that floor")
    print("=" * 74)
    print(f"   immovable shipped maximum          = {MEASURED_SHIPPED_MAX}")
    print(f"   <T,5,5> at r=4 (uniform, no mixed) = {regs([5])}"
          f"   RAISES the ceiling")
    for name, lad in (("183(C) 16+15*NA", LADDER_R2_183C),
                      ("E44    15+17*NA", LADDER_R2_E44)):
        v = r2_regs(lad, 5)
        head = MEASURED_SHIPPED_MAX - v
        print(f"   <T,5,5> at r=2, {name}   = {v:>4}"
              f"   FITS, headroom {head}")
    print()
    print("   Both r=2 fits sit below the floor, so 187(L) is robust to the")
    print("   intercept disagreement, and the headroom is larger than the")
    print("   original 104-based version of this tool reported.")

    print()
    print("=" * 74)
    print("6. What this adds")
    print("=" * 74)
    print("   The r=2 route is not one option that happens to fit. It is the")
    print("   only route that can ever fit, because 108 is a legality floor")
    print("   rather than a tuning choice, and because askeladd's law shows")
    print(f"   NO NA=5 table can read below {regs([5])} (only M=5 attains it,")
    print("   every other NA=5 width being mixed and therefore 129).")


def self_test():
    n = 0

    # 1. the law reproduces all six observations with zero residual
    for key, obs in CENSUS.items():
        assert regs(list(key)) == obs, key
    n += 1

    # 2. the shipped chooser IPG = ceil(M / ceil(M / 4)) reproduces the table
    for m in range(M_MIN, M_MAX + 1):
        assert math.ceil(m / math.ceil(m / 4)) == SHIPPED[m], m
    n += 1

    # 3. the law reproduces the MEASURED shipped maximum of 108, not 104
    got = max(regs(groups_for(m, SHIPPED[m])) for m in range(M_MIN, M_MAX + 1))
    assert got == MEASURED_SHIPPED_MAX == 108, got
    n += 1

    # 4. and it reproduces E27's measured 129 from the M=9 cell alone
    assert regs(groups_for(9, 5)) == MEASURED_E27_MAX == 129
    n += 1

    # 5. M=7 has no legal configuration below 108 -- the load-bearing claim
    opts = [regs(groups_for(7, i)) for i in legal_ipgs(7)]
    assert min(opts) == 108, opts
    assert 7 % 2 == 1 and 7 % 3 == 1
    n += 1

    # 6. the legal floor over all widths equals the shipped maximum
    floors = [min(regs(groups_for(m, i)) for i in legal_ipgs(m))
              for m in range(M_MIN, M_MAX + 1)]
    assert max(floors) == MEASURED_SHIPPED_MAX, floors
    n += 1

    # 7. both r=2 fits sit strictly below the floor, with real headroom
    for lad in (LADDER_R2_183C, LADDER_R2_E44):
        assert r2_regs(lad, 5) < MEASURED_SHIPPED_MAX, lad
    assert r2_regs(LADDER_R2_183C, 5) == 91
    assert r2_regs(LADDER_R2_E44, 5) == 100
    n += 1

    # 8. no NA=5 table can reach the shipped maximum: 125 is the minimum and
    #    only the uniform M=5 cell attains it
    na5 = {m: regs(groups_for(m, 5)) for m in range(5, 10) if legal(m, 5)}
    assert min(na5.values()) == 125
    assert [m for m, v in na5.items() if v == 125] == [5], na5
    assert all(v > MEASURED_SHIPPED_MAX for v in na5.values()), na5
    n += 1

    # 9. the tail-of-1 rule really does exclude M=6 at IPG=5, as E55 states
    assert not legal(6, 5)
    n += 1

    print(f"self-test OK: {n} checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    self_test() if a.self_test else report()
