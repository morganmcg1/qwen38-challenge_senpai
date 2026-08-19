#!/usr/bin/env python3
"""Ledger item 148: is our MTP-leg deficit against the board plateau real, and where
does its score value live?

DATA PROVENANCE
---------------
Input is the full ranked-board dump, one JSON array of submission rows, fetched from

    GET https://api.yukon.org/api/benchmarks/
        5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?limit=2000

which is the only endpoint that returns complete `officialMetrics.per_prompt` for every
row.  Pass the saved file as argv[1] (default /tmp/rows_live.json).  Zero GPU: this is
pure re-analysis of numbers the organizers already published.

WHY THE REFERENCE CLASS MATTERS (ledger 149)
--------------------------------------------
Two wrong noise models were used before this one, and both understated the effect:

  * the 94-row *serial*-leg per-prompt sd (0.2054 %) is the right sigma for comparing
    sessions days apart and the WRONG one for rows submitted the same evening;
  * residuals against the 94-row cohort *median* mix "we beat the median tree" into
    "we trail the plateau" and produce a meaningless number.

The correct comparison is against the contemporaneous plateau: six rows submitted
2026-08-18 16:59-23:52, all with `effective_mean_draft_len` identical to 4 dp on every
prompt, i.e. six independent submissions of what is effectively one tree.  Our row
landed 22:44, inside that window.  Their mutual per-prompt scatter is the noise floor
for this comparison, and it is measured here rather than assumed.

The load-bearing conclusion needs no sigma at all: the three narrow prompts are a
control INSIDE OUR OWN ROW.  Whatever luck our session drew, it moved them by ~0, so it
cannot be what moves the five wide prompts by ~0.33 %.

SCORE IDENTITY (verified on all 94 matched rows, 0 mismatches)
-------------------------------------------------------------
    raw_p = serial_seconds_per_token_mean / mtp_seconds_per_token_mean
    score = mean of the 4th and 5th order statistics of the eight raw_p values

Only the two prompts occupying those slots can move the score.  For us those are
beagle and medicine, which is why a 34-sigma deficit on essays is worth exactly zero.
"""
import json
import math
import statistics
import sys

HEAD = '559b24eb'  # declared head_provenance_sha256 prefix; score claims need this
ORDER = ['c1ec5866', '4b9e88cd', '3b10cb4d', '919318e1',
         '00142a44', 'a2ea8b60', '192fb621', 'ea82dcb5']
NAME = dict(zip(ORDER, ['plutarch', 'drama', 'travel', 'beagle',
                        'medicine', 'essays', 'republic', 'botany']))
OURS = '2b0c36a078'
# The six rows scoring above us on the declared head, all submitted 2026-08-18.
PLATEAU = ['ef42e043', '1cb1f43a72', 'e267db8c80',
           '0cbaf6a7f7', 'c0e34afd85', '9cd3be9b99']
NARROW = ['c1ec5866', '4b9e88cd', '3b10cb4d']          # draftlen <= 2.7
WIDE = ['919318e1', '00142a44', 'a2ea8b60', '192fb621', 'ea82dcb5']  # >= 4.5


def load(path):
    rows = json.load(open(path))
    if isinstance(rows, dict):
        for k in ('submissions', 'data', 'items', 'results'):
            if k in rows:
                rows = rows[k]
                break
    out = {}
    for r in rows:
        pp = ((r.get('officialMetrics') or {}).get('per_prompt') or [])
        d = {}
        ok = True
        for p in pp:
            if not (p.get('head_provenance_sha256') or '').startswith(HEAD):
                ok = False
                break
            d[(p.get('prompt_sha256') or '')[:8]] = p
        if not ok or len(d) != 8:
            continue
        sha = (r.get('submissionCommitSha') or '')
        if not sha:
            continue
        out[sha[:12]] = (d, r.get('solverUsername'), r.get('officialScore'))
    return out


def score(serial, mtp):
    rp = sorted(serial[k] / mtp[k] for k in ORDER)
    return (rp[3] + rp[4]) / 2.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rows_live.json'
    rows = load(path)
    ours_key = next(k for k in rows if k.startswith(OURS))
    plat = [k for k in rows if any(k.startswith(p) for p in PLATEAU)]
    assert len(plat) == 6, 'expected 6 plateau rows, got %d' % len(plat)
    print('head-matched rows: %d   plateau rows: %d   ours: %s'
          % (len(rows), len(plat), ours_key))

    mtp = {k: {p: rows[k][0][p]['mtp_seconds_per_token_mean'] for p in ORDER} for k in rows}
    ser = {p: rows[ours_key][0][p]['serial_seconds_per_token_mean'] for p in ORDER}
    ref = {p: statistics.median(mtp[k][p] for k in plat) for p in ORDER}

    def resid(k, p):
        return 100.0 * (mtp[k][p] / ref[p] - 1.0)

    print()
    print('%-9s %8s %10s %11s %9s' % ('prompt', 'draftlen', 'deficit_%', 'plateau_sd', 'sigma'))
    for p in ORDER:
        sd = statistics.pstdev([resid(k, p) for k in plat])
        r = resid(ours_key, p)
        dl = rows[ours_key][0][p]['effective_mean_draft_len']
        sig = ('%+9.2f' % (r / sd)) if sd > 1e-6 else '      n/a'
        print('%-9s %8.4f %+10.4f %11.4f %s' % (NAME[p], dl, r, sd, sig))

    print()
    for lbl, grp in (('narrow (draftlen <= 2.7)', NARROW), ('wide (draftlen >= 4.5)', WIDE)):
        vals = [resid(ours_key, p) for p in grp]
        print('%-26s mean %+.4f %%   signs %s'
              % (lbl, statistics.fmean(vals), ''.join('+' if v > 0 else '-' for v in vals)))
    print('  -> the narrow group is a within-row control: it rules out common-mode,')
    print('     thermal, box and session explanations without using any sigma estimate.')

    # inversion contrast, calibrated on the plateau's own scatter
    bm = [resid(k, '919318e1') - resid(k, '00142a44') for k in plat]
    sd_bm = statistics.pstdev(bm)
    ours_bm = resid(ours_key, '919318e1') - resid(ours_key, '00142a44')
    print()
    print('(beagle - medicine) contrast: ours %+.4f %%  plateau sd %.4f %%  => %.2f sigma'
          % (ours_bm, sd_bm, ours_bm / sd_bm))

    # ---- score value: only the 4th and 5th order statistics can move ----
    base = score(ser, mtp[ours_key])
    assert abs(base - rows[ours_key][2]) < 1e-8, 'score identity failed'
    print()
    print('score identity reproduces official row to %.1e' % abs(base - rows[ours_key][2]))
    print()
    print('SCORE VALUE OF CLOSING EACH LEG DEFICIT TO PLATEAU MEDIAN, ONE AT A TIME:')
    tot = 0.0
    for p in ORDER:
        m2 = dict(mtp[ours_key])
        m2[p] = min(m2[p], ref[p])
        g = 100.0 * (score(ser, m2) / base - 1.0)
        tot += g
        print('   %-9s leg %+7.3f %%  ->  score %+.4f %%' % (NAME[p], resid(ours_key, p), g))
    m_all = {p: min(mtp[ours_key][p], ref[p]) for p in ORDER}
    print('   ALL EIGHT AT ONCE               ->  score %+.4f %%  (sum of singles %+.4f)'
          % (100.0 * (score(ser, m_all) / base - 1.0), tot))

    print()
    print('LADDER (what a change must deliver; crown gap is +0.5193 %%):')
    for imp in (0.417, 0.640, 1.0, 2.0):
        m_b = dict(mtp[ours_key])
        m_b['919318e1'] *= (1 - imp / 100.0)
        m_j = dict(mtp[ours_key])
        for p in ('919318e1', '00142a44'):
            m_j[p] *= (1 - imp / 100.0)
        print('   -%.3f %%   beagle only -> %+.4f %%     beagle+medicine -> %+.4f %%'
              % (imp, 100.0 * (score(ser, m_b) / base - 1.0),
                 100.0 * (score(ser, m_j) / base - 1.0)))

    # ---- ledger 149: the item-124 identity, proved numerically ----
    print()
    print('LEDGER 149 CHECK -- "consistent across N rivals" is N=1 when the N share a term.')
    print('For each prompt, sd(rival - us) must EQUAL sd(rival) because -us is constant:')
    bad = 0
    for p in ORDER:
        a = statistics.pstdev([resid(k, p) for k in plat])
        b = statistics.pstdev([resid(k, p) - resid(ours_key, p) for k in plat])
        same = abs(a - b) < 1e-12
        bad += (not same)
        print('   %-9s sd(rival)=%.6f  sd(rival-us)=%.6f  equal=%s' % (NAME[p], a, b, same))
    print('   mismatches: %d (expected 0)' % bad)

    # ---- shape: not per-round, not per-token ----
    dl = [rows[ours_key][0][p]['effective_mean_draft_len'] for p in ORDER]
    dpct = [resid(ours_key, p) for p in ORDER]
    dms = [1000.0 * (mtp[ours_key][p] - ref[p]) for p in ORDER]

    def pearson(a, b):
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = math.sqrt(sum((x - ma) ** 2 for x in a))
        db = math.sqrt(sum((y - mb) ** 2 for y in b))
        return num / (da * db)

    print()
    print('SHAPE: corr(draftlen, deficit_pct) = %+.4f' % pearson(dl, dpct))
    print('       corr(1/(1+draftlen), deficit_ms) = %+.4f'
          % pearson([1.0 / (1.0 + d) for d in dl], dms))
    print('       -> positive would mean a fixed per-ROUND host cost; it is negative.')
    print('       deficit_ms range %+.4f .. %+.4f ms' % (min(dms), max(dms)))
    print('       -> a fixed per-TOKEN cost would be flat; it is not.')
    print('       Conclusion: the deficit scales with draft/verify BATCH WIDTH.')


if __name__ == '__main__':
    main()
