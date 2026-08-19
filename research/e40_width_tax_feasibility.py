#!/usr/bin/env python3
"""E40 deliverable 3: the joint-consistency test on a per-width tax.

QUESTION (from the brief): is there any `(per-M tax vector, width distribution)`
pair simultaneously consistent with beagle +0.363 % and medicine +0.088 %?  If
the feasible set is empty, no per-width tax model explains the data and the
mechanism is not width-CELL-specific.  If it is non-empty, report which widths
it implicates.

THE MODEL, and why it is exact rather than a fit
------------------------------------------------
Item 123 establishes that every top rival has bit-identical acceptance to ours,
and the plateau rows share our `effective_mean_draft_len` to 12 significant
figures.  Identical acceptance means the SAME per-round draft counts, so the two
trees traverse the SAME width sequence and differ only in the cost of each
width.  Write `n_M` for the number of rounds our row spends at verify width M
(M = 1 + draft count), `c'_M` for the plateau's per-round cost at that width and
`tau_M = c_M / c'_M - 1` for our relative cost at that width.  Then the leg
deficit is exactly a weighted mean of the per-width taxes:

    deficit = sum_M n_M c_M / sum_M n_M c'_M - 1 = sum_M w_M tau_M   (first order)
    w_M = n_M c'_M / sum_j n_j c'_j            (time share, not round share)

WHAT THE PUBLISHED FIELDS PIN EXACTLY
-------------------------------------
Per prompt, `effective_mean_draft_len` is a 12-decimal rational D/R whose
reduction recovers the integer round count R and draft count D (the brief's
`beagle R=107 D=485`).  `non_drafting_round_count` pins `n_1` exactly.  So the
feasible set of width distributions is the polytope

    n >= 0,   sum_M n_M = R,   sum_M n_M (M-1) = D,   n_1 = nd

which is 2 equalities after fixing n_1, so its VERTICES have support size <= 2.
`sum w tau` is linear-fractional in n, so its extrema over the polytope are
attained at those vertices: enumerating all width PAIRS gives the EXACT
achievable interval of the deficit for any tau.  No optimisation, no fit.

THE CORRECTION THAT CHANGES THIS QUESTION
-----------------------------------------
The brief quotes "E27's own M-table (... M6 1.0150 ...)".  E27's actual measured
M=6 ratio is 1.0032 (research/results/qwen38-r1-e27-m5-weight-stream-cliff.md
:212).  1.0150 is the M=6 ratio from ledger item 129's table, which is E33's
row-blocking arm -- a DIFFERENT experiment, FALSIFIED, and not in the shipped
5-file surface.  E27's own report calls M = 3,4,6,7,8 "the five untouched widths
... all land within +-0.5 %, which sets the noise floor" (:222).  So the "1.50 %
per-M6 tax" the brief sets out to bound was never E27's number, and the bound
the brief derives from medicine (<= 0.29 %) is CONSISTENT with E27's real M=6
value rather than in tension with it.

Zero GPU.  Board dump plus two committed local cost tables.
"""
import itertools
import json
import statistics
import sys
from fractions import Fraction

HEAD = '559b24eb'
ORDER = ['c1ec5866', '4b9e88cd', '3b10cb4d', '919318e1',
         '00142a44', 'a2ea8b60', '192fb621', 'ea82dcb5']
NAME = dict(zip(ORDER, ['plutarch', 'drama', 'travel', 'beagle',
                        'medicine', 'essays', 'republic', 'botany']))
OURS = '2b0c36a078'
PLATEAU = ['ef42e043', '1cb1f43a72', 'e267db8c80',
           '0cbaf6a7f7', 'c0e34afd85', '9cd3be9b99']
WIDE = ['919318e1', '00142a44', 'a2ea8b60', '192fb621', 'ea82dcb5']
BANKABLE = ['919318e1', '00142a44']          # the 4th/5th order statistics
MAXM = 9                                     # mtp_max_draft_depth = 8 => M <= 9

# research/results/qwen38-r1-e27-m5-weight-stream-cliff.md:207-216.
# C_round(M) in ms, recomputed from each run's vendored.json as
# sum_shapes calls_per_verify x seconds_per_call.  M4 Pro, 128/64, residency off.
E27_BASE = {1: 64.549, 2: 65.628, 3: 72.993, 4: 83.115, 5: 120.683,
            6: 128.865, 7: 139.078, 8: 149.355, 9: 186.233}
E27_CAND = {1: 59.979, 2: 64.707, 3: 73.136, 4: 83.072, 5: 96.423,
            6: 129.280, 7: 139.007, 8: 150.110, 9: 164.900}
# The E33 table the brief's 1.0150 actually comes from (ledger item 129), kept
# here only so the mix-up is reproducible rather than asserted.
E33_M6_RATIO = 130.781 / 128.843

# Which widths E27 rewrote.  quantized.h case 5: <T,5,3> -> <T,5,5>; case 9:
# <T,9,3> -> <T,9,5>.  Cases 3,4,6,7,8 are byte-identical in both trees, so the
# ONLY route by which E27 can cost anything there is the register allocation of
# the single enclosing kernel (see research/e40_cell_air.sh).
CHANGED_CELLS = (5, 9)


def load(path):
    raw = json.load(open(path))
    rows = raw['submissions'] if isinstance(raw, dict) else raw
    out = {}
    for r in rows:
        pp = ((r.get('officialMetrics') or {}).get('per_prompt') or [])
        if not pp:
            continue
        if {(p.get('head_provenance_sha256') or '')[:8] for p in pp} != {HEAD}:
            continue
        d = {(p.get('prompt_sha256') or '')[:8]: p for p in pp}
        sha = (r.get('submissionCommitSha') or '')
        if len(d) != 8 or not sha:
            continue
        out[sha[:12]] = (d, r)
    return out


def recover_counts(mean_draft, decode_tokens=512):
    """R and D from the published 12-decimal mean draft length."""
    f = Fraction(mean_draft).limit_denominator(decode_tokens)
    return f.denominator, f.numerator


def vertices(R, D, nd, maxm=MAXM):
    """Every support-<=2 width distribution consistent with (R, D, n_1 = nd).

    Returns a list of {M: rounds} dicts.  These are the vertices of the polytope,
    so any linear-fractional functional of the distribution attains its extremes
    here.
    """
    out = []
    R2, D2 = R - nd, D            # rounds and drafts left after the n_1 = nd rounds
    if R2 < 0:
        return out
    lo = 2
    if R2 == 0:
        return [{1: nd}] if D2 == 0 else []
    for a in range(lo, maxm + 1):
        for b in range(a, maxm + 1):
            # x rounds at width a, y at width b: x + y = R2, x(a-1) + y(b-1) = D2
            if a == b:
                if (a - 1) * R2 == D2:
                    d = {a: R2}
                    if nd:
                        d[1] = nd
                    out.append(d)
                continue
            y = Fraction(D2 - (a - 1) * R2, (b - a))
            x = R2 - y
            if x < 0 or y < 0:
                continue
            d = {a: float(x), b: float(y)}
            if nd:
                d[1] = nd
            out.append(d)
    return out


def solve3(cols, rhs):
    """Exact 3x3 solve by Cramer's rule; None when the columns are dependent."""
    def det(c0, c1, c2):
        return (c0[0] * (c1[1] * c2[2] - c1[2] * c2[1])
                - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
                + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1]))
    a, b, c = cols
    d = det(a, b, c)
    if d == 0:
        return None
    return [det(rhs, b, c) / d, det(a, rhs, c) / d, det(a, b, rhs) / d]


def deficit_of(dist, tau, cost):
    """sum_M w_M tau_M with time shares w from the plateau-side cost table."""
    num = sum(n * cost[M] * tau[M] for M, n in dist.items())
    den = sum(n * cost[M] for M, n in dist.items())
    return 100.0 * num / den


def interval(dist_list, tau, cost):
    vals = [deficit_of(d, tau, cost) for d in dist_list]
    return min(vals), max(vals), dist_list[vals.index(min(vals))], dist_list[vals.index(max(vals))]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rows_live.json'
    rows = load(path)
    ours = next(k for k in rows if k.startswith(OURS))
    plat = [k for k in rows if any(k.startswith(p) for p in PLATEAU)]
    assert len(plat) == 6

    mtp = {k: {p: rows[k][0][p]['mtp_seconds_per_token_mean'] for p in ORDER} for k in rows}
    ref = {p: statistics.median(mtp[k][p] for k in plat) for p in ORDER}
    deficit = {p: 100.0 * (mtp[ours][p] / ref[p] - 1.0) for p in ORDER}

    print('=' * 78)
    print('EXACT ROUND / DRAFT / WIDTH-DISTRIBUTION RECOVERY (our ranked row)')
    print('=' * 78)
    facts = {}
    print('%-9s %-18s %5s %5s %5s %5s %8s %8s %6s'
          % ('prompt', 'mean_draft', 'R', 'D', 'A', 'nd', 'meanM', 'alpha', 'verts'))
    for p in ORDER:
        pp = rows[ours][0][p]
        md = pp['effective_mean_draft_len']
        R, D = recover_counts(md)
        nd = pp['non_drafting_round_count']
        A = 512 - R
        vs = vertices(R, D, nd)
        facts[p] = dict(R=R, D=D, A=A, nd=nd, verts=vs, meanM=1 + md)
        print('%-9s %-18.12f %5d %5d %5d %5d %8.4f %8.4f %6d'
              % (NAME[p], md, R, D, A, nd, 1 + md, A / D if D else 0, len(vs)))
    print('  R + A = 512 for every prompt: %s'
          % all(facts[p]['R'] + facts[p]['A'] == 512 for p in ORDER))
    print("  brief's beagle R=107 D=485 A=405 alpha=.8351 reproduced: %s"
          % (facts['919318e1']['R'] == 107 and facts['919318e1']['D'] == 485
             and facts['919318e1']['A'] == 405))
    print("  brief's medicine R=99 D=472 A=413 alpha=.8750 reproduced: %s"
          % (facts['00142a44']['R'] == 99 and facts['00142a44']['D'] == 472
             and facts['00142a44']['A'] == 413))

    print()
    print('=' * 78)
    print("THE BRIEF'S M-TABLE IS E33's, NOT E27's")
    print('=' * 78)
    print('  %-4s %-12s %-12s %-10s %-10s' % ('M', 'E27 base', 'E27 cand', 'E27 ratio', 'brief'))
    brief = {1: 0.9292, 2: 0.9860, 3: 1.0020, 4: 0.9995, 5: 0.7990,
             6: 1.0150, 7: 0.9995, 8: 1.0051, 9: 0.8854}
    for M in range(1, MAXM + 1):
        r = E27_CAND[M] / E27_BASE[M]
        flag = '' if abs(r - brief[M]) < 5e-4 else '   <-- MISMATCH'
        print('  %-4d %-12.3f %-12.3f %-10.4f %-10.4f%s'
              % (M, E27_BASE[M], E27_CAND[M], r, brief[M], flag))
    print('  E33 (ledger 129) M=6 ratio = %.4f, which is the brief\'s 1.0150.'
          % E33_M6_RATIO)
    print('  E33 is FALSIFIED (ledger 129) and is NOT in the shipped 5-file surface,')
    print('  so no 1.50 %% M=6 tax exists in the tree that produced our ranked row.')

    tau_e27 = {M: E27_CAND[M] / E27_BASE[M] - 1.0 for M in range(1, MAXM + 1)}
    cost = dict(E27_BASE)          # plateau-side per-width cost, ms

    print()
    print('=' * 78)
    print('F1. IS E27\'s OWN MEASURED TAX VECTOR CONSISTENT WITH THE TWO LEGS?')
    print('=' * 78)
    print('  achievable deficit interval over every support-<=2 width distribution')
    print('  %-9s %9s %10s %10s  %s' % ('prompt', 'observed', 'min', 'max', 'in range?'))
    ok_all = True
    for p in ORDER:
        lo, hi, dlo, dhi = interval(facts[p]['verts'], tau_e27, cost)
        inr = lo - 1e-9 <= deficit[p] <= hi + 1e-9
        ok_all &= inr or p not in WIDE
        print('  %-9s %+9.4f %+10.4f %+10.4f  %s'
              % (NAME[p], deficit[p], lo, hi, 'YES' if inr else 'NO'))
    print()
    for p in BANKABLE:
        lo, hi, dlo, dhi = interval(facts[p]['verts'], tau_e27, cost)
        sol = [d for d in facts[p]['verts']
               if abs(deficit_of(d, tau_e27, cost) - deficit[p]) < 0.02]
        print('  %s: widths that reproduce the observed %+0.4f %% within 0.02 pp:'
              % (NAME[p], deficit[p]))
        if sol:
            for d in sol:
                print('      %s   predicted %+0.4f %%'
                      % ({k: round(v, 1) for k, v in sorted(d.items())},
                         deficit_of(d, tau_e27, cost)))
        else:
            print('      none at support 2; the observed value needs an interior mix')
        print('      widest-deficit vertex %s -> %+0.4f %%'
              % ({k: round(v, 1) for k, v in sorted(dhi.items())}, hi))
        print('      lowest-deficit vertex %s -> %+0.4f %%'
              % ({k: round(v, 1) for k, v in sorted(dlo.items())}, lo))

    print()
    print('=' * 78)
    print('F2. THE SOURCE-STRUCTURAL FAMILY (what the 4 shipped kernel lines imply)')
    print('=' * 78)
    print('  E27 rewrote only cells M=5 and M=9.  Everywhere else our kernel text is')
    print('  byte-identical to the plateau\'s, so tau_M there can come ONLY from the')
    print('  register allocation of the single enclosing kernel.  Model:')
    print('      tau_M = rho                     for M not in {5, 9}')
    print('      tau_5 = rho - beta5, tau_9 = rho - beta9,  beta >= 0')
    print('  Fit rho to the three narrow legs, then ask what beta the wide legs need.')
    narrow = ['4b9e88cd', '3b10cb4d']         # drama, travel; plutarch latch excluded
    print()
    print('  %-9s %8s %10s %10s' % ('prompt', 'meanM', 'observed', 'rho-only range'))
    for p in narrow + WIDE:
        tau_flat = {M: 1.0 for M in range(1, MAXM + 1)}   # unit rho
        lo, hi, _, _ = interval(facts[p]['verts'], tau_flat, cost)
        print('  %-9s %8.4f %+10.4f  x rho in [%.3f, %.3f]'
              % (NAME[p], facts[p]['meanM'], deficit[p], lo / 100, hi / 100))
    print('  A flat rho gives EVERY prompt the same deficit rho, so the two narrow')
    print('  legs (%+0.4f, %+0.4f) force rho ~ 0 and then the five wide legs at'
          % (deficit['4b9e88cd'], deficit['3b10cb4d']))
    print('  +0.23..+0.51 %% cannot be produced at all.  A width-INDEPENDENT')
    print('  register tax is REFUTED by our own narrow legs.')
    print()
    print('  So the register ceiling cannot be the whole mechanism.  Admit a')
    print('  width-dependent register penalty tau_M = rho * g(M) and ask what g')
    print('  the data force.  Two one-parameter shapes, both fitted on the two')
    print('  bankable legs and cross-validated on the other three wide legs:')
    for label, g in (('g(M) = M-1  (per-row)', lambda M: M - 1),
                     ('g(M) = 1[M>=6] (threshold at the 2nd weight pass)',
                      lambda M: 1.0 if M >= 6 else 0.0),
                     ('g(M) = 1[M>=4]', lambda M: 1.0 if M >= 4 else 0.0),
                     ('g(M) = launched TGs = M', lambda M: float(M))):
        tau_unit = {M: g(M) for M in range(1, MAXM + 1)}
        print()
        print('    %s' % label)
        print('      %-9s %10s %-28s' % ('prompt', 'observed', 'rho needed (range)'))
        needs = {}
        for p in narrow + WIDE:
            lo, hi, _, _ = interval(facts[p]['verts'], tau_unit, cost)
            if abs(hi) < 1e-12 and abs(lo) < 1e-12:
                needs[p] = None
                print('      %-9s %+10.4f  %s' % (NAME[p], deficit[p], 'no leverage (g=0 everywhere feasible)'))
                continue
            cand = sorted(x for x in (deficit[p] / lo if abs(lo) > 1e-12 else None,
                                      deficit[p] / hi if abs(hi) > 1e-12 else None)
                          if x is not None)
            needs[p] = (cand[0], cand[-1])
            print('      %-9s %+10.4f  rho in [%.4f, %.4f]'
                  % (NAME[p], deficit[p], cand[0], cand[-1]))
        live = [needs[p] for p in narrow + WIDE if needs[p]]
        if live:
            lo = max(x[0] for x in live)
            hi = min(x[1] for x in live)
            print('      intersection over all seven usable legs: %s'
                  % ('[%.4f, %.4f]  FEASIBLE' % (lo, hi) if lo <= hi else 'EMPTY'))
            lb = [needs[p] for p in BANKABLE if needs[p]]
            lo2 = max(x[0] for x in lb)
            hi2 = min(x[1] for x in lb)
            print('      intersection over beagle+medicine only:   %s'
                  % ('[%.4f, %.4f]  FEASIBLE' % (lo2, hi2) if lo2 <= hi2 else 'EMPTY'))

    print()
    print('=' * 78)
    print('F3. FREE TAX VECTOR WITH THE SOURCE SIGN CONSTRAINTS')
    print('=' * 78)
    print('  tau_5 <= 0 and tau_9 <= 0 (we removed a weight stream at those two')
    print('  widths and E27 measured -20.1 %% and -11.5 %% there); tau_M >= 0')
    print('  elsewhere (identical kernel text, plus a raised register ceiling).')
    print('  With 7 free non-negative taus and 8 leg equations the system is')
    print('  under-determined, so the question is only whether the SIGNS are')
    print('  attainable.  Necessary condition, checked exactly per prompt:')
    tau_pos = {M: (0.0 if M in CHANGED_CELLS else 1.0) for M in range(1, MAXM + 1)}
    for p in ORDER:
        lo, hi, _, dhi = interval(facts[p]['verts'], tau_pos, cost)
        print('  %-9s observed %+8.4f %%   max attainable at unit rho on the'
              ' unchanged cells %+8.4f  (%s)'
              % (NAME[p], deficit[p], hi,
                 {k: round(v, 1) for k, v in sorted(dhi.items())}))
    print()
    print('  Every prompt can attain an arbitrary positive deficit by scaling rho,')
    print('  so a free per-width tax vector is trivially consistent and therefore')
    print('  says nothing.  The informative statements are F1 and F2.')

    print()
    print('=' * 78)
    print('F4. THE FALSIFIABLE WIDTH-CENSUS PREDICTION (for askeladd to check)')
    print('=' * 78)
    print('  Under E27\'s measured tau the only cells with tau > 0 are M=3,6,8')
    print('  (+0.20, +0.32, +0.51 %) while M=1,2,5,9 are all NEGATIVE.  So a')
    print('  positive observed deficit forces time-share OFF M=5 and M=9.')
    print('  Adding "deficit == observed" to the polytope keeps it LINEAR after')
    print('  clearing the denominator:  sum_M n_M c_M (tau_M - obs) = 0.  With')
    print('  three equalities the vertices have support <= 3, so enumerating')
    print('  every width triple and solving exactly in rationals gives the')
    print('  TRUE maximum decode-time share on {5, 9}.  A measured histogram')
    print('  above that bound falsifies "E27 kernel tax alone" for that leg.')
    tau27 = {M: Fraction(E27_BASE[M]).limit_denominator(10 ** 9)
             / Fraction(E27_CAND[M]).limit_denominator(10 ** 9) - 1
             for M in E27_BASE}
    cost27 = {M: Fraction(E27_CAND[M]).limit_denominator(10 ** 9) for M in E27_CAND}
    for p in BANKABLE:
        f, obs = facts[p], Fraction(deficit[p] / 100.0).limit_denominator(10 ** 12)
        R, D, nd = f['R'], f['D'], f['nd']
        widths = [m for m in range(2, MAXM + 1)]
        # n_1 = nd is pinned; fold its contribution into the right-hand sides.
        rhs = [Fraction(R - nd), Fraction(D),
               -Fraction(nd) * cost27[1] * (tau27[1] - obs)]
        best, feasible = None, 0
        for tri in itertools.combinations(widths, 3):
            cols = [[Fraction(1), Fraction(m - 1), cost27[m] * (tau27[m] - obs)]
                    for m in tri]
            sol = solve3(cols, rhs)
            if sol is None or any(v < 0 for v in sol):
                continue
            feasible += 1
            dist = {m: sol[i] for i, m in enumerate(tri)}
            if nd:
                dist[1] = Fraction(nd)
            den = sum(n * cost27[m] for m, n in dist.items())
            if den == 0:
                continue
            sh = sum(n * cost27[m] for m, n in dist.items() if m in (5, 9)) / den
            if best is None or sh > best[0]:
                best = (sh, dist)
        if best is None:
            print('  %-9s observed %+7.4f %%  INFEASIBLE: no non-negative width mix'
                  % (NAME[p], deficit[p]))
            print('             reproduces this deficit under E27\'s measured tau.')
        else:
            print('  %-9s observed %+7.4f %%  %d feasible vertices, max decode-time'
                  % (NAME[p], deficit[p], feasible))
            print('             share on M in {5,9} = %.2f %%   at %s'
                  % (100.0 * float(best[0]),
                     {m: round(float(v), 1) for m, v in sorted(best[1].items()) if v}))

    print()
    print('=' * 78)
    print('WHAT THE FEASIBLE SET IMPLICATES, IN BANKABLE-LEG TERMS')
    print('=' * 78)
    print('  Only beagle (919318e1) and medicine (00142a44) can move the score.')
    print('  beagle  observed %+0.4f %%   value of closing it +0.1752 %% of score'
          % deficit['919318e1'])
    print('  medicine observed %+0.4f %%  value of closing it +0.0455 %% of score'
          % deficit['00142a44'])
    print('  Their mean widths differ by only %.3f rows (%.4f vs %.4f) yet their'
          % (facts['00142a44']['meanM'] - facts['919318e1']['meanM'],
             facts['919318e1']['meanM'], facts['00142a44']['meanM']))
    print('  deficits differ by %.3f pp, and medicine -- the WIDER prompt -- has the'
          % (deficit['919318e1'] - deficit['00142a44']))
    print('  SMALLER deficit.  Under E27\'s measured tax vector that inversion is')
    print('  exactly what a difference in width MIX predicts: M=5 and M=9 are the')
    print('  two cells where our tree is FASTER (-20.1 %%, -11.5 %%), so a prompt')
    print('  with more mass on 5 or 9 shows a smaller deficit at a larger mean.')


if __name__ == '__main__':
    main()
