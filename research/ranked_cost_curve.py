"""Recover the ranked M5 per-round cost curve from official Yukon board data.

Method
------
`officialMetrics.per_prompt[i].effective_mean_draft_len` is the exact rational
`total_drafts / total_rounds`, where the denominator counts non-drafting rounds
too. `Fraction(dl).limit_denominator()` therefore recovers the ranked round count
for that prompt. With the fixed 512-token decode window and
`mtp_seconds_per_token_mean`, that gives the ranked cost of one round at a known
verify width `M = effective_mean_draft_len + 1`.

The curve is two independent lines with a step at the `G = ceil(M / 4)` group
boundary. The three-parameter form `a + b*G + c*M` is NOT identifiable on ranked
data because the two per-row slopes differ; a naive least-squares fit with free
round counts returns b = -20,374.7 by choosing round counts that equalise accept
rates. Fit two lines instead.

CAVEAT, carry it on every use
-----------------------------
The round counts in ROUNDS below are INFERRED, not measured. They are pinned by
the maximal-tokens-per-round rule (tokens/round <= M) plus a plausibility
constraint on the implied accept rate. drama is the one judgement call: 168
rounds would make drama dearer than travel at a lower verify width, which is
impossible. The internal validation is that four parameters fit eight prompts to
1.3 % and stay stable across 50 independent official runs.

Input
-----
/tmp/yukon-board/full.json, as written by the board refresh helper.

See senpai/campaign-ledger.md entry 243 and
research/CURRENT_RESEARCH_STATE.md section 0d.
"""

import json, statistics

full = json.load(open('/tmp/yukon-board/full.json'))
rows = None
for k in ('submissions', 'rows', 'data', 'items'):
    if isinstance(full, dict) and k in full:
        rows = full[k]
        break
rows = [r for r in rows if isinstance(r, dict)]
PROMPTS = {'919318e1': 'beagle', '192fb621': 'botany', '4b9e88cd': 'drama',
           'a2ea8b60': 'essays', '00142a44': 'medicine', 'c1ec5866': 'plutarch',
           'ea82dcb5': 'republic', '3b10cb4d': 'travel'}
def pp(r):
    om = r.get('officialMetrics') or {}
    return {PROMPTS.get(e['prompt_sha256'][:8], e['prompt_sha256'][:8]): e
            for e in (om.get('per_prompt') or [])}
T = 512

# Round counts fixed by the maximal-tokens-per-round rule, except drama where
# 168 makes drama dearer than travel at a lower width, which is impossible.
ROUNDS = {'plutarch': 487, 'drama': 252, 'travel': 212, 'beagle': 110,
          'republic': 93, 'essays': 92, 'medicine': 90, 'botany': 81}
G1 = ['plutarch', 'drama', 'travel']
G2 = ['beagle', 'republic', 'essays', 'medicine', 'botany']

def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b

scored = [r for r in rows if len(pp(r)) == 8 and r.get('officialScore')]
print(f'runs with all eight prompts: {len(scored)}')

recs = []
for r in scored:
    e = pp(r)
    if any(e[k]['effective_mean_draft_len'] + 1.0 > 9.0 for k in e):
        continue
    # require the reference schedule so widths are comparable
    if abs(e['beagle']['effective_mean_draft_len'] - 4.381818181818182) > 1e-9:
        continue
    d = {}
    ok = True
    for k in ROUNDS:
        R = ROUNDS[k]
        M = e[k]['effective_mean_draft_len'] + 1.0
        ru = T * e[k]['mtp_seconds_per_token_mean'] / R * 1e6
        d[k] = (M, ru)
    a1, c1 = fit([d[k][0] for k in G1], [d[k][1] for k in G1])
    a2, c2 = fit([d[k][0] for k in G2], [d[k][1] for k in G2])
    res2 = [(d[k][1] - (a2 + c2 * d[k][0])) / d[k][1] for k in G2]
    recs.append((r['id'][:8], r['officialScore'], a1, c1, a2, c2, max(abs(v) for v in res2), d))

print(f'runs on the reference schedule: {len(recs)}')
print()
def q(v, p):
    s = sorted(v)
    return s[int(p * (len(s) - 1))]
for name, idx in [('a1 intercept G=1', 2), ('c1 per row G=1', 3),
                  ('a2 intercept G=2', 4), ('c2 per row G=2', 5),
                  ('max |resid| G=2', 6)]:
    v = [x[idx] for x in recs]
    print(f'{name:20s} median {statistics.median(v):11.1f}   p10 {q(v,0.10):11.1f}   p90 {q(v,0.90):11.1f}   '
          f'cv {100*statistics.pstdev(v)/abs(statistics.fmean(v)):5.2f} %')

med = lambda i: statistics.median([x[i] for x in recs])
A1, C1, A2, C2 = med(2), med(3), med(4), med(5)
print()
print(f'RANKED CURVE (median over {len(recs)} official runs, harness=ranked, M5)')
print(f'  G=1, M=1..4 : round_us = {A1:9.1f} + {C1:8.1f} * M')
print(f'  G=2, M=5..8 : round_us = {A2:9.1f} + {C2:8.1f} * M')
LOCAL = {1: 64445.4, 2: 69775.5, 3: 74778.4, 4: 86237.4, 5: 126103.1,
         6: 137842.6, 7: 150431.4, 8: 163957.1, 9: 204028.5}
print()
print(f"{'M':>3s}{'G':>3s}{'ranked_us':>11s}{'local_us':>11s}{'ratio':>8s}{'rank marg':>11s}{'loc marg':>11s}")
prev = prevl = None
for M in range(1, 9):
    G = 1 if M <= 4 else 2
    rk = (A1 + C1 * M) if G == 1 else (A2 + C2 * M)
    lc = LOCAL[M]
    mr = '' if prev is None else f'{rk-prev:11.1f}'
    ml = '' if prevl is None else f'{lc-prevl:11.1f}'
    print(f'{M:3d}{G:3d}{rk:11.1f}{lc:11.1f}{lc/rk:8.3f}{mr:>11s}{ml:>11s}')
    prev, prevl = rk, lc
s4 = (A2 + C2 * 5) - (A1 + C1 * 4)
print()
print(f'  M=4 -> M=5 boundary step  ranked {s4:9.1f} us = {100*s4/(A1+C1*4):5.2f} %')
print(f'                            local  {LOCAL[5]-LOCAL[4]:9.1f} us = {100*(LOCAL[5]-LOCAL[4])/LOCAL[4]:5.2f} %')
print()
print('=== composition of the ranked beagle round (M = 5.3818) ===')
Mb = 5.381818181818182
rb = A2 + C2 * Mb
print(f'  round               {rb:9.1f} us')
print(f'  intercept a + 2b    {A2:9.1f} us  = {100*A2/rb:5.1f} %')
print(f'  per-row c * M       {C2*Mb:9.1f} us  = {100*C2*Mb/rb:5.1f} %')
lb = LOCAL[5] + 0.381818 * (LOCAL[6] - LOCAL[5])
print(f'  local round         {lb:9.1f} us   machine ratio {lb/rb:5.3f}')
print(f'  local a + 2b        {10920 + 2*27377:9.1f} us  = {100*(10920+2*27377)/lb:5.1f} %')
print(f'  local c * M         {10268*Mb:9.1f} us  = {100*10268*Mb/lb:5.1f} %')
