import json, sys, math, statistics

rows = json.load(open('/tmp/yukon-board/full.json'))['submissions']
by = {}
for r in rows:
    sid = (r.get('id') or '')[:8]
    if r.get('officialMetrics') and r['officialMetrics'].get('per_prompt'):
        by[sid] = r

NAMES = {'919318e1': 'beagle', '192fb621': 'botany', '4b9e88cd': 'drama',
         'a2ea8b60': 'essays', '00142a44': 'medicine', 'c1ec5866': 'plutarch',
         'ea82dcb5': 'republic', '3b10cb4d': 'travel'}
ORDER = ['plutarch', 'drama', 'travel', 'beagle', 'republic', 'essays', 'medicine', 'botany']
TOK = 512


def pp(sid):
    d = {}
    for e in by[sid]['officialMetrics']['per_prompt']:
        d[NAMES.get(e['prompt_sha256'][:8], e['prompt_sha256'][:8])] = e
    return d


a, b = sys.argv[1], sys.argv[2]
da, db = pp(a), pp(b)
print('%-9s %6s %6s %7s %7s | %10s %10s %9s %9s' % (
    'prompt', 'R', 'draftR', 'Mbar', 'nondrf', 'A us/rd', 'B us/rd', 'sav/rd', 'sav/drfrd'))
tot = []
for n in ORDER:
    ea, eb = da[n], db[n]
    Mbar = ea['effective_mean_draft_len']
    # rounds: tokens / (Mbar + 1)
    R = TOK / (Mbar + 1.0)
    nd = ea.get('non_drafting_round_count')
    ra = ea['mtp_seconds_per_token_mean'] * TOK * 1e6 / R
    rb = eb['mtp_seconds_per_token_mean'] * TOK * 1e6 / R
    drf = R - nd if nd is not None else float('nan')
    sav = ra - rb
    print('%-9s %6.1f %6.1f %7.4f %7s | %10.1f %10.1f %9.1f %9.1f' % (
        n, R, drf, Mbar, str(nd), ra, rb, sav, sav * R / drf if drf and drf > 0 else float('nan')))
    if drf and drf > 0:
        tot.append(sav * R / drf)
print('mean saving per DRAFTING round: %.1f us   sd %.1f  (mode effect is ~820 us)' % (
    statistics.mean(tot), statistics.pstdev(tot)))
