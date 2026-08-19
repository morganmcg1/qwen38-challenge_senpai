#!/usr/bin/env python3
"""E40 step 0: attack the instrument before building anything on it.

The E40 brief rests entirely on ledger item 148, which is re-derived by
`research/board_plateau_deficit.py`.  The brief's own instruction is to attack
that tool first, because if its plateau-scatter estimate is wrong the whole
assignment is void.  This script is the attack.  Zero GPU: pure re-analysis of
the published board dump.

    curl -s -H "Authorization: Bearer $YUKON_API_TOKEN" \
      'https://api.yukon.org/api/benchmarks/5d1ee4d7-80bd-4555-b182-6505f26ef495/submissions?limit=2000' \
      > /tmp/rows_live.json
    python3 research/e40_instrument_audit.py /tmp/rows_live.json

FIVE INDEPENDENT ATTACKS, each of which could void item 148:

A. COHORT SIZE.  Item 148 says 94 head-matched rows; the tool's own filter now
   yields a different number.  Reconcile it exactly and name what is dropped.

B. SELECTION ON THE OUTCOME.  The plateau is defined as "the six rows scoring
   above us".  Score is a function of the per-prompt MTP legs, so conditioning
   on score > ours biases those legs fast, and the bias is concentrated on the
   4th/5th order statistics -- which for us are beagle and medicine, the two
   prompts the brief is about.  This is the attack with real teeth.  It is
   answered by rebuilding the reference class WITHOUT using score at all:
   same-evening rows carrying the same draft-length fingerprint.

C. REPLICATE INDEPENDENCE.  A 0.0149 % between-row sd on essays over six rows
   is only a noise floor if the six are six independent measurements.  If any
   two rows share a run (identical published seconds/token), the sd is a
   duplication artifact and every sigma in item 148 is inflated.

D. ESTIMATOR BIAS.  `pstdev` of residuals about the plateau MEDIAN, computed
   from the same six rows, is a population sd about an in-sample centre on
   n = 6.  Report the sample-sd and leave-one-out versions so the sigmas can be
   read at their least favourable.

E. THE WITHIN-ROW CONTROL, RESTATED AS A TEST.  Item 148's load-bearing claim
   needs no sigma: narrow prompts move ~0, wide prompts move ~0.33 %.  Test it
   as a permutation of the narrow/wide labels over our own eight residuals,
   which uses no plateau sd at all.
"""
import itertools
import json
import statistics
import sys

HEAD = '559b24eb'
ORDER = ['c1ec5866', '4b9e88cd', '3b10cb4d', '919318e1',
         '00142a44', 'a2ea8b60', '192fb621', 'ea82dcb5']
NAME = dict(zip(ORDER, ['plutarch', 'drama', 'travel', 'beagle',
                        'medicine', 'essays', 'republic', 'botany']))
OURS = '2b0c36a078'
PLATEAU = ['ef42e043', '1cb1f43a72', 'e267db8c80',
           '0cbaf6a7f7', 'c0e34afd85', '9cd3be9b99']
NARROW = ['c1ec5866', '4b9e88cd', '3b10cb4d']
WIDE = ['919318e1', '00142a44', 'a2ea8b60', '192fb621', 'ea82dcb5']


def load(path):
    """Same filter as board_plateau_deficit.load, plus the drop accounting."""
    raw = json.load(open(path))
    rows = raw['submissions'] if isinstance(raw, dict) else raw
    kept, drops = {}, {'head_mismatch': 0, 'no_per_prompt': 0,
                       'dup_prompt': 0, 'no_commit_sha': 0}
    for r in rows:
        pp = ((r.get('officialMetrics') or {}).get('per_prompt') or [])
        if not pp:
            drops['no_per_prompt'] += 1
            continue
        if {(p.get('head_provenance_sha256') or '')[:8] for p in pp} != {HEAD}:
            drops['head_mismatch'] += 1
            continue
        d = {(p.get('prompt_sha256') or '')[:8]: p for p in pp}
        if len(d) != 8:
            drops['dup_prompt'] += 1
            continue
        sha = (r.get('submissionCommitSha') or '')
        if not sha:
            drops['no_commit_sha'] += 1
            continue
        kept[sha[:12]] = (d, r)
    return kept, drops, len(rows)


def score(serial, mtp):
    rp = sorted(serial[k] / mtp[k] for k in ORDER)
    return (rp[3] + rp[4]) / 2.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rows_live.json'
    rows, drops, total = load(path)
    ours = next(k for k in rows if k.startswith(OURS))
    plat = [k for k in rows if any(k.startswith(p) for p in PLATEAU)]
    assert len(plat) == 6, 'expected the 6 named plateau rows, got %d' % len(plat)

    mtp = {k: {p: rows[k][0][p]['mtp_seconds_per_token_mean'] for p in ORDER}
           for k in rows}
    ser = {p: rows[ours][0][p]['serial_seconds_per_token_mean'] for p in ORDER}

    print('=' * 78)
    print('A. COHORT ACCOUNTING  (item 148 claims 94 head-matched rows)')
    print('=' * 78)
    print('  rows in dump                        %d' % total)
    print('  kept by the item-148 filter         %d' % len(rows))
    for k, v in sorted(drops.items()):
        print('  dropped: %-26s %d' % (k, v))
    nosha = drops['no_commit_sha']
    print('  -> %d rows are all-8 on head %s but carry no submissionCommitSha,'
          % (nosha, HEAD))
    print('     so the head-matched population is %d and the analysable cohort %d.'
          % (len(rows) + nosha, len(rows)))
    print('     The plateau six and our row are unaffected: the cohort size enters')
    print('     item 148 only through the 94-row serial-leg sigma, which item 148')
    print('     itself discards as the WRONG reference class.  No sigma in the')
    print('     brief moves. Recorded as a citation drift, not an error.')

    print()
    print('=' * 78)
    print('B. SELECTION ON THE OUTCOME  (the attack with teeth)')
    print('=' * 78)
    ts = {k: (rows[k][1].get('createdAt') or '') for k in rows}
    fp = {k: tuple(round(rows[k][0][p]['effective_mean_draft_len'], 4) for p in ORDER)
          for k in rows}
    ours_fp = fp[ours]
    same_day = [k for k in rows if ts[k][:10] == ts[ours][:10]]
    # Reference class chosen WITHOUT looking at score: same calendar day, and the
    # same draft-length fingerprint to 4 dp on all eight prompts (item 148's own
    # "identical work" criterion), excluding ourselves.
    twins = [k for k in same_day if fp[k] == ours_fp and k != ours]
    print('  our row %s  submitted %s  score %.6f'
          % (ours, ts[ours], rows[ours][1]['officialScore']))
    print('  rows sharing our submission DAY                      %d' % len(same_day))
    print('  ... AND our 8-prompt draft-length fingerprint to 4dp  %d' % len(twins))
    above = [k for k in twins if rows[k][1]['officialScore'] > rows[ours][1]['officialScore']]
    below = [k for k in twins if rows[k][1]['officialScore'] <= rows[ours][1]['officialScore']]
    print('      of which score above us %d, at-or-below us %d' % (len(above), len(below)))
    print('      named plateau six are a subset of the twins: %s'
          % all(k in twins for k in plat))
    # Was the plateau six cherry-picked?  Enumerate every cohort row above us.
    allabove = sorted((k for k in rows
                       if rows[k][1]['officialScore'] > rows[ours][1]['officialScore']),
                      key=lambda k: -rows[k][1]['officialScore'])
    print()
    print('      EVERY cohort row scoring above us (%d), and whether item 148 used it:'
          % len(allabove))
    for k in allabove:
        r = rows[k][1]
        print('        %-14s %-18s %-10.6f %-24s %-8s %s'
              % (k, r.get('solverUsername'), r['officialScore'], r.get('createdAt'),
                 r.get('status'), 'PLATEAU' if k in plat else 'excluded'))

    def resid_table(ref_keys, label):
        ref = {p: statistics.median(mtp[k][p] for k in ref_keys) for p in ORDER}
        out = {p: 100.0 * (mtp[ours][p] / ref[p] - 1.0) for p in ORDER}
        n = len(ref_keys)
        print()
        print('  reference class: %s  (n=%d)' % (label, n))
        print('  %-9s %8s %10s %11s %8s' % ('prompt', 'draftlen', 'deficit_%', 'ref_sd_%', 'sigma'))
        for p in ORDER:
            rs = [100.0 * (mtp[k][p] / ref[p] - 1.0) for k in ref_keys]
            sd = statistics.stdev(rs) if n > 2 else float('nan')
            dl = rows[ours][0][p]['effective_mean_draft_len']
            sig = ('%+8.2f' % (out[p] / sd)) if sd == sd and sd > 1e-6 else '     n/a'
            print('  %-9s %8.4f %+10.4f %11.4f %s' % (NAME[p], dl, out[p], sd, sig))
        nm = statistics.fmean([out[p] for p in NARROW])
        wm = statistics.fmean([out[p] for p in WIDE])
        print('  narrow mean %+.4f %%   wide mean %+.4f %%   wide-narrow %+.4f %%'
              % (nm, wm, wm - nm))
        return out

    print()
    print('  B1. item 148 says all seven rows have effective_mean_draft_len')
    print('      "identical to four decimal places on every prompt". Test it:')
    for k in plat:
        diff = [NAME[p] for i, p in enumerate(ORDER) if fp[k][i] != ours_fp[i]]
        print('      %-14s %-18s %s'
              % (k, rows[k][1].get('solverUsername'),
                 'identical 8/8' if not diff else 'DIFFERS on ' + ','.join(diff)))
    for p in ORDER:
        vs = {round(rows[k][0][p]['effective_mean_draft_len'], 4) for k in plat + [ours]}
        if len(vs) > 1:
            print('      -> %s draftlen values across the seven: %s'
                  % (NAME[p], sorted(vs)))

    r_plateau = resid_table(plat, 'the named plateau six (SELECTED ON SCORE)')
    resid_table(twins, 'all same-day draft-length twins (NO score selection)')

    # B2. A reference class selected on a variable with PROVEN zero score value:
    # closeness to us on the three narrow legs.  Item 148 and item 146 both show
    # narrow legs cannot move the 4th/5th order statistic, so conditioning on
    # them cannot smuggle in the outcome. If narrow-matched rows still beat us on
    # the wide legs, selection on score is not what produces the deficit.
    def narrow_dist(k):
        return max(abs(100.0 * (mtp[k][p] / mtp[ours][p] - 1.0)) for p in NARROW[1:])
    pool = [k for k in twins if narrow_dist(k) < 0.25]
    print()
    print('  B2. rows within 0.25 %% of us on BOTH scoreless narrow legs')
    print('      (drama, travel; plutarch excluded because its latch state varies)')
    print('      pool size %d, of which %d score above us and %d at or below'
          % (len(pool),
             sum(1 for k in pool if rows[k][1]['officialScore'] > rows[ours][1]['officialScore']),
             sum(1 for k in pool if rows[k][1]['officialScore'] <= rows[ours][1]['officialScore'])))
    if len(pool) >= 3:
        resid_table(pool, 'narrow-leg-matched rows (selection orthogonal to score)')

    # B3. A reference class selected on the SERIAL leg, which is a nominally
    # identical computation in every submission and therefore a box-speed proxy
    # rather than a candidate-code proxy.
    def serial_dist(k):
        return max(abs(100.0 * (rows[k][0][p]['serial_seconds_per_token_mean'] / ser[p] - 1.0))
                   for p in ORDER)
    spool = [k for k in twins if serial_dist(k) < 0.60]
    print()
    print('  B3. rows within 0.60 %% of us on ALL EIGHT serial legs (box-speed match)')
    print('      pool size %d, of which %d score above us and %d at or below'
          % (len(spool),
             sum(1 for k in spool if rows[k][1]['officialScore'] > rows[ours][1]['officialScore']),
             sum(1 for k in spool if rows[k][1]['officialScore'] <= rows[ours][1]['officialScore'])))
    if len(spool) >= 3:
        resid_table(spool, 'serial-leg-matched rows (selection orthogonal to MTP code)')

    # B4. The plateau set is a post-hoc choice: 11 cohort rows now score above us
    # and item 148 used 6.  Robustness to that choice.
    window = [k for k in allabove
              if '2026-08-18T16:59' <= ts[k] <= '2026-08-18T23:53']
    print()
    print('  B4. robustness to the plateau DEFINITION')
    resid_table(allabove, 'ALL %d cohort rows above us' % len(allabove))
    resid_table(window, 'all %d rows above us inside the 16:59-23:52 window' % len(window))

    print()
    print('  MDE of each reference class on the wide-minus-narrow contrast')
    print('  (paired design, per-prompt sd pooled over the wide legs, n = wide legs)')
    sys.path.insert(0, 'research')
    import e39_mde
    for label, keys in (('plateau six', plat), ('narrow-matched', pool),
                        ('serial-matched', spool), ('all-above', allabove),
                        ('same-day twins', twins)):
        ref = {p: statistics.median(mtp[k][p] for k in keys) for p in ORDER}
        sds = []
        for p in WIDE:
            rs = [100.0 * (mtp[k][p] / ref[p] - 1.0) for k in keys]
            sds.append(statistics.stdev(rs))
        s = e39_mde.pooled_sd(*sds)
        print('    %-16s n=%-3d pooled wide sd %7.4f %%   MDE(normal) %7.4f %%'
              '   MDE(exact) %7.4f %%'
              % (label, len(keys), s,
                 e39_mde.mde(s, len(WIDE), 'paired'),
                 e39_mde.mde_exact(s, len(WIDE), 'paired')))

    print()
    print('=' * 78)
    print('C. ARE THE SIX PLATEAU ROWS SIX INDEPENDENT MEASUREMENTS?')
    print('=' * 78)
    print('  %-14s %-18s %-10s %-22s %s'
          % ('sha12', 'solver', 'score', 'createdAt', 'status'))
    for k in sorted(plat, key=lambda k: -rows[k][1]['officialScore']):
        r = rows[k][1]
        print('  %-14s %-18s %-10.6f %-22s %s'
              % (k, r.get('solverUsername'), r['officialScore'],
                 r.get('createdAt'), r.get('status')))
    dupes = 0
    for a, b in itertools.combinations(plat, 2):
        if all(mtp[a][p] == mtp[b][p] for p in ORDER):
            dupes += 1
            print('  DUPLICATE RUN: %s and %s publish identical seconds/token on 8/8'
                  % (a, b))
        shared = sum(mtp[a][p] == mtp[b][p] for p in ORDER)
        if 0 < shared < 8:
            print('  partial tie: %s vs %s share %d/8 exact values' % (a, b, shared))
    print('  identical-on-all-8 pairs: %d (expected 0 for 6 independent runs)' % dupes)
    print('  distinct solvers: %d of 6'
          % len({rows[k][1].get('solverUsername') for k in plat}))
    print('  distinct submission commits: %d of 6' % len(set(plat)))

    print()
    print('=' * 78)
    print('D. ESTIMATOR BIAS IN THE PLATEAU SD')
    print('=' * 78)
    print('  %-9s %10s %10s %10s %10s %10s'
          % ('prompt', 'deficit_%', 'pstdev', 'stdev', 'sigma_p', 'sigma_s'))
    worst = []
    for p in ORDER:
        ref = statistics.median(mtp[k][p] for k in plat)
        rs = [100.0 * (mtp[k][p] / ref - 1.0) for k in plat]
        sp, ss = statistics.pstdev(rs), statistics.stdev(rs)
        dv = r_plateau[p]
        if ss > 1e-6:
            worst.append((NAME[p], dv / sp, dv / ss))
            print('  %-9s %+10.4f %10.4f %10.4f %+10.2f %+10.2f'
                  % (NAME[p], dv, sp, ss, dv / sp, dv / ss))
        else:
            print('  %-9s %+10.4f %10.4f %10.4f %10s %10s'
                  % (NAME[p], dv, sp, ss, 'n/a', 'n/a'))
    print('  pstdev/stdev ratio is sqrt(5/6) = %.4f, so item 148 quotes sigmas'
          % (5.0 / 6.0) ** 0.5)
    print('  about 9.5 %% too large.  beagle at the sample sd is %+.2f sigma.'
          % [w[2] for w in worst if w[0] == 'beagle'][0])
    # leave-one-out on the plateau: does any single row carry the beagle sigma?
    print()
    print('  LEAVE-ONE-OUT on beagle (drop each plateau row in turn):')
    for drop in plat:
        keep = [k for k in plat if k != drop]
        ref = statistics.median(mtp[k]['919318e1'] for k in keep)
        dv = 100.0 * (mtp[ours]['919318e1'] / ref - 1.0)
        rs = [100.0 * (mtp[k]['919318e1'] / ref - 1.0) for k in keep]
        sd = statistics.stdev(rs)
        print('    without %-12s deficit %+.4f %%  sd %.4f  sigma %+.2f'
              % (drop, dv, sd, dv / sd))

    print()
    print('=' * 78)
    print('E. THE WITHIN-ROW CONTROL AS AN EXACT TEST (uses no plateau sd)')
    print('=' * 78)
    vals = [r_plateau[p] for p in ORDER]
    obs = (statistics.fmean([r_plateau[p] for p in WIDE])
           - statistics.fmean([r_plateau[p] for p in NARROW]))
    # exact permutation over which 5 of the 8 prompts are labelled wide
    perms = list(itertools.combinations(range(8), 5))
    stats = []
    for c in perms:
        w = [vals[i] for i in c]
        n = [vals[i] for i in range(8) if i not in c]
        stats.append(statistics.fmean(w) - statistics.fmean(n))
    ge = sum(1 for s in stats if s >= obs - 1e-12)
    print('  observed wide-minus-narrow contrast   %+.4f %%' % obs)
    print('  exact permutation p (one-sided)       %d/%d = %.4f'
          % (ge, len(perms), ge / len(perms)))
    print('  -> with 8 prompts the smallest attainable p is 1/56 = 0.0179, so this')
    print('     test can never be more than moderately significant.  It is a')
    print('     LABEL-ORDERING check, and its value is that the wide/narrow split')
    print('     is also the draft-length ORDERING: the 5 largest draftlen prompts')
    print('     are exactly the 5 largest deficits?  %s'
          % (sorted(ORDER, key=lambda p: rows[ours][0][p]['effective_mean_draft_len'])[-5:]
             == sorted(ORDER, key=lambda p: r_plateau[p])[-5:]))
    ranks = sorted(ORDER, key=lambda p: rows[ours][0][p]['effective_mean_draft_len'])
    print('  draftlen order : %s' % ' < '.join(NAME[p] for p in ranks))
    print('  deficit order  : %s'
          % ' < '.join(NAME[p] for p in sorted(ORDER, key=lambda p: r_plateau[p])))

    print()
    print('=' * 78)
    print('SCORE IDENTITY AND THE ORDER-STATISTIC GATE (unchanged check)')
    print('=' * 78)
    base = score(ser, mtp[ours])
    assert abs(base - rows[ours][1]['officialScore']) < 1e-8, 'score identity failed'
    print('  score identity reproduces our official row to %.1e'
          % abs(base - rows[ours][1]['officialScore']))
    rp = sorted((ser[p] / mtp[ours][p], NAME[p]) for p in ORDER)
    print('  our per-prompt raw_p in order:')
    for i, (v, n) in enumerate(rp):
        mark = '  <-- 4th/5th order statistic (the only bankable prompts)' \
            if i in (3, 4) else ''
        print('    %d  %-9s %.6f%s' % (i + 1, n, v, mark))


if __name__ == '__main__':
    main()
