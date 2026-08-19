"""E33: per-shape attribution of the M=6 regression and the ceiling on shape gating."""
import json

M = 6


def per_shape(tag):
    d = json.load(open('.mlxfast-private/qmv-curve/%s/vendored.json' % tag))
    out = {}
    for s in d['shapes']:
        r = next(x for x in s['rows'] if x['m'] == M)
        out[s['name']] = dict(n=s['n'], k=s['k'], calls=s['calls_per_verify'],
                              spc=r['seconds_per_call'],
                              cost=s['calls_per_verify'] * r['seconds_per_call'])
    return out


base, cand = per_shape('e33-base-r1'), per_shape('e33-cand-r1')
tot_b = sum(v['cost'] for v in base.values())
tot_c = sum(v['cost'] for v in cand.values())
net = tot_c - tot_b

print('C_round(6): base %.4f ms  cand %.4f ms  ratio %.4f  net %+.4f ms'
      % (tot_b * 1e3, tot_c * 1e3, tot_c / tot_b, net * 1e3))
print()
hdr = '%-34s %7s %6s %9s %9s %8s %9s %7s'
print(hdr % ('shape', 'n', 'calls', 'base_ms', 'cand_ms', 'ratio', 'delta_ms', '%net'))
for k in sorted(base, key=lambda x: -abs(cand[x]['cost'] - base[x]['cost'])):
    b, c = base[k], cand[k]
    d = (c['cost'] - b['cost']) * 1e3
    print(hdr % (k, b['n'], b['calls'], '%.4f' % (b['cost'] * 1e3), '%.4f' % (c['cost'] * 1e3),
                 '%.4f' % (c['spc'] / b['spc']), '%+.4f' % d, '%+.1f' % (100 * d / (net * 1e3))))

wins = sum(min(0.0, cand[k]['cost'] - base[k]['cost']) for k in base)
loss = sum(max(0.0, cand[k]['cost'] - base[k]['cost']) for k in base)
print()
print('total wins %+.4f ms   total losses %+.4f ms   net %+.4f ms'
      % (wins * 1e3, loss * 1e3, net * 1e3))

print()
print('Shape-gated follow-up: apply ROWS_PER_SIMD=2 only where n >= threshold')
for t in (5121, 14336, 16480, 34816, 98336):
    g = sum(cand[k]['cost'] if base[k]['n'] >= t else base[k]['cost'] for k in base)
    picked = sorted(k for k in base if base[k]['n'] >= t)
    print('  n >= %6d : ratio %.4f (%+.2f %%)  blocks %d shapes' % (t, g / tot_b, 100 * (g / tot_b - 1), len(picked)))

oracle = sum(min(cand[k]['cost'], base[k]['cost']) for k in base)
print()
print('ORACLE (per-shape best, unimplementable upper bound): ratio %.4f (%+.2f %%)'
      % (oracle / tot_b, 100 * (oracle / tot_b - 1)))
print('control-band on this instrument (max |ratio-1| over untreated widths): 0.46 %')
