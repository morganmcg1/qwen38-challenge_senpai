import json, math, statistics

rows = json.load(open('/tmp/yukon-board/full.json'))['submissions']
NAMES = {'919318e1': 'beagle', '192fb621': 'botany', '4b9e88cd': 'drama',
         'a2ea8b60': 'essays', '00142a44': 'medicine', 'c1ec5866': 'plutarch',
         'ea82dcb5': 'republic', '3b10cb4d': 'travel'}
W = {'plutarch': -0.3852, 'drama': 0.0215, 'travel': 0.4945, 'beagle': 0.2068,
     'medicine': -0.1480, 'republic': -0.0917, 'essays': -0.0041, 'botany': -0.0939}
ORDER = ['plutarch', 'drama', 'travel', 'beagle', 'republic', 'essays', 'medicine', 'botany']

out = []
for r in rows:
    om = r.get('officialMetrics') or {}
    pp = om.get('per_prompt')
    if not pp or len(pp) != 8:
        continue
    d = {}
    ok = True
    for e in pp:
        n = NAMES.get(e['prompt_sha256'][:8])
        if n is None or not e.get('mtp_seconds_per_token_mean'):
            ok = False; break
        d[n] = e
    if not ok or len(d) != 8:
        continue
    idx = sum(W[n] * 100.0 * math.log(d[n]['mtp_seconds_per_token_mean']) for n in W)
    out.append((r.get('createdAt') or '', (r.get('id') or '')[:8], r.get('solverUsername'),
                r.get('officialScore'), idx, d['plutarch']['mtp_seconds_per_token_mean'],
                tuple(round(d[n]['effective_mean_draft_len'], 6) for n in ORDER)))

out.sort()
recent = [o for o in out if o[0] >= '2026-08-21T18:00']
print('rows with full per-prompt metrics: %d   since 2026-08-21T18:00Z: %d' % (len(out), len(recent)))
print()
print('%-22s %-9s %-14s %10s %9s %11s' % ('created', 'id', 'solver', 'score', 'modeidx', 'plutarch'))
for o in recent:
    print('%-22s %-9s %-14s %10s %9.4f %11.8f' % (o[0][:19], o[1], (o[2] or '')[:14],
                                                  ('%.6f' % o[3]) if o[3] else 'None', o[4], o[5]))

print()
print('=== mode index histogram, bin 0.10, all %d rows ===' % len(out))
h = {}
for o in out:
    b = round(o[4] * 10) / 10.0
    h[b] = h.get(b, 0) + 1
for b in sorted(h):
    print('%7.1f %s %d' % (b, '#' * min(h[b], 70), h[b]))
