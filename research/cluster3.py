import json, sys, math, statistics

rows = json.load(open('/tmp/yukon-board/full.json'))['submissions']
by = {}
for r in rows:
    sid = (r.get('id') or '')[:8]
    om = r.get('officialMetrics') or {}
    if om.get('per_prompt'):
        by[sid] = r
NAMES = {'919318e1': 'beagle', '192fb621': 'botany', '4b9e88cd': 'drama',
         'a2ea8b60': 'essays', '00142a44': 'medicine', 'c1ec5866': 'plutarch',
         'ea82dcb5': 'republic', '3b10cb4d': 'travel'}
ORDER = ['plutarch', 'drama', 'travel', 'beagle', 'republic', 'essays', 'medicine', 'botany']


def pp(sid):
    d = {}
    for e in by[sid]['officialMetrics']['per_prompt']:
        d[NAMES.get(e['prompt_sha256'][:8], '?')] = e
    return d


ids = sys.argv[1:]
print('%-9s %-14s %10s  %s' % ('id', 'solver', 'score', '  '.join('%9s' % n[:9] for n in ORDER)))
vecs = {}
for sid in ids:
    d = pp(sid)
    v = [d[n]['mtp_seconds_per_token_mean'] for n in ORDER]
    vecs[sid] = v
    print('%-9s %-14s %10.6f  %s' % (sid, (by[sid].get('solverUsername') or '')[:14],
                                     by[sid]['officialScore'],
                                     '  '.join('%9.7f' % x for x in v)))
print()
print('pairwise mean |pct diff| per prompt (candidate leg)')
print('%-9s %s' % ('', '  '.join('%9s' % s for s in ids)))
for a in ids:
    line = []
    for b in ids:
        d = [abs(vecs[a][i] - vecs[b][i]) / vecs[a][i] * 100 for i in range(8)]
        line.append('%9.4f' % statistics.mean(d))
    print('%-9s %s' % (a, '  '.join(line)))
print()
print('pairwise MAX |pct diff| over the 8 prompts')
for a in ids:
    line = []
    for b in ids:
        d = [abs(vecs[a][i] - vecs[b][i]) / vecs[a][i] * 100 for i in range(8)]
        line.append('%9.4f' % max(d))
    print('%-9s %s' % (a, '  '.join(line)))
