import json

b = json.load(open('/tmp/yukon-board/full.json'))['submissions']
d = {r.get('id', '')[:8]: r for r in b}
names = {'919318e1': 'beagle', '192fb621': 'botany', '4b9e88cd': 'drama', 'a2ea8b60': 'essays',
         '00142a44': 'medicine', 'c1ec5866': 'plutarch', 'ea82dcb5': 'republic', '3b10cb4d': 'travel'}


def cand(sid):
    pp = d[sid]['officialMetrics']['per_prompt']
    return {names[x['prompt_sha256'][:8]]: x['mtp_seconds_per_token_mean'] for x in pp}


def raws(sid):
    pp = d[sid]['officialMetrics']['per_prompt']
    return {names[x['prompt_sha256'][:8]]: x['raw_ratio_of_means'] for x in pp}


def medpair(sid):
    r = raws(sid)
    s = sorted(r.items(), key=lambda kv: kv[1])
    m = d[sid]['officialScore']
    return {s[3][0]: 0.5 * s[3][1] / m, s[4][0]: 0.5 * s[4][1] / m}, s


print('=' * 78)
print('SORTED ORDER CHECK ON EVERY TREE WITH FULL PER-PROMPT DATA')
print('=' * 78)
ids = ['623e77af', 'b6cb0fea', 'dacf7005', '0b8602e1', '02742bf0', 'ed608e64',
       '08b67f12', '0b2f0014', '115c5c50', 'c24f1755']
for sid in ids:
    if sid not in d or not d[sid].get('officialMetrics'):
        print('%-9s no metrics' % sid)
        continue
    w, s = medpair(sid)
    order = ' '.join(k[:4] for k, _ in s)
    ident = 0.5 * (s[3][1] + s[4][1])
    print('%-9s %-14s %s  ident %.8f  pub %.8f  %s'
          % (sid, d[sid].get('solverUsername', '')[:13], order, ident,
             d[sid]['officialScore'],
             'OK' if abs(ident - d[sid]['officialScore']) < 1e-8 else 'MISMATCH'))

print()
print('=' * 78)
print('NEW PAIRS, CANDIDATE LEG, MEDIAN-PAIR WEIGHTED (positive = slower)')
print('=' * 78)
pairs = [
    ('F189 tight grid                ', '02742bf0', 'ed608e64'),
    ('F192 probe 0.25->0.15 tight    ', 'ed608e64', '08b67f12'),
    ('F194 one-pass table under tight', 'ed608e64', '0b2f0014'),
    ('F197 prefill swizzle (nulltest)', 'ed608e64', '115c5c50'),
    ('F198 table+probe15+E122bM2     ', 'ed608e64', 'c24f1755'),
]
F83 = {'beagle': 0.4862, 'medicine': 0.2508, 'essays': 0.1598, 'botany': 0.0124,
       'republic': 0.0100, 'drama': 0.0, 'travel': 0.0, 'plutarch': 0.0}
print('%-33s %9s %9s %9s %9s' % ('mechanism', 'F83', 'medpair', 'unweight', 'pubmedian'))
for label, a, bb in pairs:
    ca, cb = cand(a), cand(bb)
    w, _ = medpair(a)
    dl = {k: 100 * (cb[k] / ca[k] - 1) for k in ca}
    f = sum(F83[k] * dl[k] for k in dl)
    mp = sum(w.get(k, 0.0) * dl[k] for k in dl)
    un = sum(dl.values()) / 8
    pm = 100 * (d[bb]['officialScore'] / d[a]['officialScore'] - 1)
    print('%-33s %+9.4f %+9.4f %+9.4f %+9.4f' % (label, f, mp, un, pm))

print()
print('=' * 78)
print('PREDICTION FOR 572b2cc4  (our 623e77af tree + tight grid, one word)')
print('=' * 78)
# route 1: from our own tree
tight_no_table = -4.0462   # medpair candidate leg, 02742bf0 -> ed608e64
p_tight = +0.2649          # table penalty under tight, ed608e64 -> 0b2f0014
p_wide = -0.2032           # table benefit under wide, our own F167 ranked receipt
delta_ours = tight_no_table + p_tight - p_wide
mult = 1.0 / (1.0 + delta_ours / 100.0)
print('  candidate-leg delta on our tree  %+.4f %%' % delta_ours)
print('  route 1  3.52085227 x %.6f = %.5f' % (mult, 3.52085227 * mult))
# route 2: from ed608e64, adding back our two deficits
deficit = p_tight + 0.2603   # table penalty + probe 0.25 vs 0.15
print('  route 2  3.68172016 x %.6f = %.5f'
      % (1.0 / (1.0 + deficit / 100.0), 3.68172016 / (1.0 + deficit / 100.0)))

print()
print('=' * 78)
print('TONIGHT-COMPOSITION LADDER FROM 572b2cc4 PREDICTED 3.6600')
print('=' * 78)
base = 3.6600
for label, g in [('pb6 tier 1.45          +2.4683 %', 0.024683),
                 ('probe 0.25 -> 0.10     +0.4848 %', 0.004848),
                 ('table onePass67 ->  .shipped', 0.002649 / (1 - 0.002649))]:
    base *= (1 + g)
    print('  %-36s -> %.5f' % (label, base))
print('  crown now 3.69071883')
