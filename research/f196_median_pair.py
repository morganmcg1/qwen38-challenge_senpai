import json, itertools

b = json.load(open('/tmp/yukon-board/full.json'))['submissions']
d = {r.get('id', '')[:8]: r for r in b}
names = {'919318e1': 'beagle', '192fb621': 'botany', '4b9e88cd': 'drama', 'a2ea8b60': 'essays',
         '00142a44': 'medicine', 'c1ec5866': 'plutarch', 'ea82dcb5': 'republic', '3b10cb4d': 'travel'}
F83 = {'beagle': 0.4862, 'medicine': 0.2508, 'essays': 0.1598, 'botany': 0.0124,
       'republic': 0.0100, 'drama': 0.0, 'travel': 0.0, 'plutarch': 0.0}


def raws(sid):
    pp = d[sid]['officialMetrics']['per_prompt']
    return {names[x['prompt_sha256'][:8]]: x['raw_ratio_of_means'] for x in pp}


ids = ['623e77af', 'b6cb0fea', 'dacf7005', '0b8602e1', '02742bf0', 'ed608e64', '08b67f12', '0b2f0014']
ids = [i for i in ids if i in d]

print('%-9s %-9s %10s %10s %10s %10s' % ('A', 'B', 'actual', 'medpair', 'F83raw', 'err_pair'))
print('-' * 62)
rows = []
for a, bb in itertools.combinations(ids, 2):
    ra, rb = raws(a), raws(bb)
    ma, mb = d[a]['officialScore'], d[bb]['officialScore']
    actual = 100 * (mb / ma - 1)
    # median-pair model: median = mean of the two straddling values, identified on A
    sa = sorted(ra.items(), key=lambda kv: kv[1])
    p4, p5 = sa[3][0], sa[4][0]
    pred_pair = 100 * (((rb[p4] + rb[p5]) / 2) / ma - 1)
    # F83 model on raw ratios
    predf = sum(F83[k] * (rb[k] / ra[k] - 1) for k in F83) * 100
    rows.append((abs(actual - pred_pair), abs(actual - predf)))
    print('%-9s %-9s %+10.4f %+10.4f %+10.4f %+10.4f' % (a, bb, actual, pred_pair, predf, actual - pred_pair))

import statistics
print('-' * 62)
print('n pairs %d' % len(rows))
print('mean |err| median-pair model  %.4f pp' % statistics.mean(r[0] for r in rows))
print('mean |err| F83 raw model      %.4f pp' % statistics.mean(r[1] for r in rows))
print('max  |err| median-pair model  %.4f pp' % max(r[0] for r in rows))
print('max  |err| F83 raw model      %.4f pp' % max(r[1] for r in rows))

print()
print('=== marginal weights on the live crown 08b67f12 ===')
r = raws('08b67f12')
m = d['08b67f12']['officialScore']
s = sorted(r.items(), key=lambda kv: kv[1])
p4, p5 = s[3], s[4]
print('4th %s %.5f   5th %s %.5f   median %.5f' % (p4[0], p4[1], p5[0], p5[1], m))
print('d(median)/d(%s) in relative terms = 0.5 * %.5f/%.5f = %.4f' % (p4[0], p4[1], m, 0.5 * p4[1] / m))
print('d(median)/d(%s) in relative terms = 0.5 * %.5f/%.5f = %.4f' % (p5[0], p5[1], m, 0.5 * p5[1] / m))
print('every other prompt                                    = 0.0000')
print()
print('headroom before the ordering changes:')
print('  %s must rise %.2f %% to reach %s' % (p4[0], 100 * (p5[1] / p4[1] - 1), p5[0]))
print('  %s must rise %.2f %% to reach %s' % (p5[0], 100 * (s[5][1] / p5[1] - 1), s[5][0]))
print()
print('ceiling if %s alone is raised without limit: median -> %.5f = +%.2f %%'
      % (p4[0], (s[4][1] + s[5][1]) / 2, 100 * (((s[4][1] + s[5][1]) / 2) / m - 1)))
print('if %s is raised to equal %s: median -> %.5f = +%.2f %%'
      % (p4[0], p5[0], p5[1], 100 * (p5[1] / m - 1)))
