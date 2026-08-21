"""Rank every official board run that ran the EXACT crown draft schedule.

Selection. A run enters the cohort only when its `effective_mean_draft_len` is
bit-identical on all eight prompts to the crown `8819b108`. Inside that cohort
the draft trajectory is the same round for round, so two runs can differ only in
what a round costs. That removes the schedule as a confounder and turns the
board into a controlled experiment on round cost.

Decomposition. Fit the five G=2 prompts, CENTERED on the width centroid
M = 6.1723, so the level and the slope are orthogonal and separately identified:

    round_us(M) = L + S * (M - Mbar)

Never fit the raw intercept. A five-point line over M in [5.38, 7.15]
extrapolated back to M = 0 has enormous leverage, so the raw intercept and the
raw slope see-saw: a run that draws low noise on botany reads as a low slope and
a high intercept with no mechanism behind it. Four of the lowest raw slopes on
the board belong to schedule-only runs with mediocre scores.

What is and is not identified (n = 54, 2026-08-21):

    L : sd 0.90 % against a noise floor near 0.09 %  -> identified, about 10x
    S : sd 2.73 % against a within-run se near 205 us -> NOT identified

So a mechanism that moves the width-independent part of the round is
confirmable by one official run. A mechanism that moves only the per-row verify
cost is not: one run resolves that slope to about +/- 2.8 %.

Run `research/board_per_prompt.py fetch` first to refresh /tmp/yukon-board.
"""
import json, math, statistics

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
T = 512
SERIAL = {'beagle': 0.037990310, 'botany': 0.037996382, 'drama': 0.037994558,
          'essays': 0.037995792, 'medicine': 0.037994454, 'plutarch': 0.037993623,
          'republic': 0.037994423, 'travel': 0.038002279}
ROUNDS = {'plutarch': 487, 'drama': 252, 'travel': 212, 'beagle': 110,
          'republic': 93, 'essays': 92, 'medicine': 90, 'botany': 81}
CROWN_DL = {'beagle': 4.381818181818182, 'botany': 6.148148148148148,
            'drama': 2.2976190476190474, 'essays': 5.086956521739131,
            'medicine': 5.2555555555555555, 'plutarch': 0.1540041067761807,
            'republic': 4.989247311827957, 'travel': 2.6556603773584904}
G2 = ['beagle', 'republic', 'essays', 'medicine', 'botany']
MS = [CROWN_DL[k] + 1.0 for k in G2]
MBAR = sum(MS) / len(MS)
SXX = sum((m - MBAR) ** 2 for m in MS)

recs = []
for r in rows:
    om = r.get('officialMetrics') or {}
    per = om.get('per_prompt') or []
    if len(per) != 8 or not r.get('officialScore'):
        continue
    d, ok = {}, True
    for e in per:
        name = PROMPTS.get(e['prompt_sha256'][:8])
        dl, spt = e.get('effective_mean_draft_len'), e.get('mtp_seconds_per_token_mean')
        if name is None or dl is None or not spt or abs(dl - CROWN_DL[name]) > 1e-12:
            ok = False
            break
        d[name] = T * spt / ROUNDS[name] * 1e6
    if not ok or len(d) != 8:
        continue
    ys = [d[k] for k in G2]
    L = sum(ys) / len(ys)
    S = sum((m - MBAR) * (y - L) for m, y in zip(MS, ys)) / SXX
    resid = [y - (L + S * (m - MBAR)) for m, y in zip(MS, ys)]
    sig = math.sqrt(sum(e * e for e in resid) / (len(ys) - 2))
    sf = sorted(SERIAL[k] / (d[k] * ROUNDS[k] / T / 1e6) for k in d)
    recs.append(dict(id=r['id'][:8], user=r.get('solverUsername'), sf=0.5 * (sf[3] + sf[4]),
                     L=L, S=S, seS=sig / math.sqrt(SXX), sig=sig,
                     pub=r['officialScore'], beagle=d['beagle'], essays=d['essays'],
                     note=' '.join((r.get('note') or '').split())[:46]))

crown = next(x for x in recs if x['id'] == '8819b108')
ours = next(x for x in recs if x['id'] == 'cb8aeefb')
print(f'same-schedule cohort n = {len(recs)}   G2 widths {[round(m,3) for m in MS]}'
      f'   centroid M = {MBAR:.4f}')
Lv = sorted(x['L'] for x in recs)
Sv = sorted(x['S'] for x in recs)
print(f'level L  at centroid : median {statistics.median(Lv):9.1f}  sd {statistics.pstdev(Lv):7.1f}'
      f' ({100*statistics.pstdev(Lv)/statistics.median(Lv):5.2f} %)')
print(f'slope S  per row     : median {statistics.median(Sv):9.1f}  sd {statistics.pstdev(Sv):7.1f}'
      f' ({100*statistics.pstdev(Sv)/statistics.median(Sv):5.2f} %)')
print(f'median within-run se(S) from residuals: {statistics.median([x["seS"] for x in recs]):8.1f}'
      f'   -> slope spread is {statistics.pstdev(Sv)/statistics.median([x["seS"] for x in recs]):.2f}x its own noise')
print()
print('=== ranked by identified LEVEL at the G=2 centroid (lowest = fastest) ===')
print(f"{'#':>3s}{'id':>9s}{'user':>16s}{'serfree':>9s}{'L':>9s}{'vs crown':>10s}"
      f"{'S':>8s}{'se(S)':>7s}  note")
for i, x in enumerate(sorted(recs, key=lambda z: z['L'])[:18], 1):
    print(f"{i:3d}{x['id']:>9s}{str(x['user'])[:15]:>16s}{x['sf']:9.5f}{x['L']:9.1f}"
          f"{100*(x['L']/crown['L']-1):9.2f}%{x['S']:8.1f}{x['seS']:7.1f}  {x['note']}")
print()
print('=== arm C signature check: is the cb8aeefb - 8819b108 delta a per-draft head cut? ===')
dL = ours['L'] - crown['L']
dS = ours['S'] - crown['S']
print(f'  delta level at centroid M={MBAR:.3f} : {dL:9.1f} us  ({100*dL/crown["L"]:+.3f} %)')
print(f'  delta slope  per verify row        : {dS:9.1f} us  (+/- {crown["seS"]:.0f})')
print(f'  delta beagle round  M=5.382        : {ours["beagle"]-crown["beagle"]:9.1f} us'
      f'  ({100*(ours["beagle"]/crown["beagle"]-1):+.3f} %)')
print(f'  delta essays round  M=6.087        : {ours["essays"]-crown["essays"]:9.1f} us'
      f'  ({100*(ours["essays"]/crown["essays"]-1):+.3f} %)')
print()
print('  A marginal-per-draft head cut of dh1 predicts   d(round) = dh0 + dh1*(M-2).')
dh1 = dS
dh0_b = (ours['beagle'] - crown['beagle']) - dh1 * (MS[0] - 2.0)
dh0_e = (ours['essays'] - crown['essays']) - dh1 * (MS[2] - 2.0)
print(f'    dh1 = dS = {dh1:8.1f} us/draft   ->  implied dh0 from beagle {dh0_b:7.1f},'
       f' from essays {dh0_e:7.1f}')
print(f'  local arm C: 2274.95 -> 1674.65 us/draft  = -600.3 us/draft (-26.39 %)')
print(f'  RANKED marginal per-draft head cut  = {-dh1:6.1f} us  -> absolute transfer'
      f' {(-dh1)/600.3:5.3f}')
print()
print('=== the lottery, measured from repeat submissions of ONE unchanged tree ===')
print('a group is one solver whose runs all sit within 0.10 % of each other on the')
print('identified level L, so the tree is effectively unchanged across the group.')
print(f"{'solver':>16s}{'n':>3s}{'L spread':>10s}{'pub mean':>10s}{'pub sd':>9s}"
      f"{'pub min':>10s}{'pub max':>10s}{'max-mean':>10s}  ids")
by_user = {}
for x in recs:
    by_user.setdefault(x['user'], []).append(x)
for user, xs in sorted(by_user.items()):
    xs = sorted(xs, key=lambda z: z['L'])
    for i in range(len(xs)):
        grp = [y for y in xs if abs(y['L'] / xs[i]['L'] - 1) < 0.0010]
        if len(grp) >= 3 and grp[0] is xs[i]:
            p = [y['pub'] for y in grp]
            mp = statistics.mean(p)
            print(f'{str(user)[:15]:>16s}{len(grp):3d}'
                  f"{100*(grp[-1]['L']/grp[0]['L']-1):9.3f}%{mp:10.5f}"
                  f'{statistics.stdev(p):9.5f}{min(p):10.5f}{max(p):10.5f}'
                  f'{100*(max(p)/mp-1):9.3f}%  ' + ' '.join(y['id'] for y in grp))
print()
print('=== per-row verify cost: what IS it? ===')
c = crown['S']
print(f'  ranked per-row slope                 {c:9.1f} us/row')
print(f'  trunk FLOP per verify row            {48.7:9.1f} GFLOP  (24.35 G params, 2 flop each)')
print(f'  => effective rate                    {48.7e9/(c*1e-6)/1e12:9.2f} TFLOP/s')
print(f'  weight bytes are read ONCE per round, shared across all M rows, so they')
print(f'  cannot be in the slope. The slope is arithmetic + per-row activations.')
act = 64 * (5120 * 2 + 5120 * 2)   # rough per-layer per-row activation r+w, bf16
print(f'  rough per-row activation traffic     {act/1e6:9.2f} MB  ->'
       f' {act/(c*1e-6)/1e9:6.1f} GB/s  (far below DRAM)')
