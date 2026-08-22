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
# F76 mode index weights
W = {'plutarch': -0.3852, 'drama': 0.0215, 'travel': 0.4945, 'beagle': 0.2068,
     'medicine': -0.1480, 'republic': -0.0917, 'essays': -0.0041, 'botany': -0.0939}
# F83 marginal weights
MW = {'beagle': 0.4862, 'medicine': 0.2508, 'essays': 0.1598, 'botany': 0.0124,
      'republic': 0.0100, 'plutarch': 0.0, 'drama': 0.0, 'travel': 0.0}


def pp(sid):
    r = by[sid]
    d = {}
    for e in r['officialMetrics']['per_prompt']:
        n = NAMES.get(e['prompt_sha256'][:8], e['prompt_sha256'][:8])
        d[n] = e
    return r, d


def modeindex(d):
    return sum(W[n] * 100.0 * math.log(d[n]['mtp_seconds_per_token_mean']) for n in W)


def median8(vals):
    v = sorted(vals)
    return 0.5 * (v[3] + v[4])


ids = sys.argv[1:]
for sid in ids:
    if sid not in by:
        print(sid, 'MISSING'); continue
    r, d = pp(sid)
    print('%s %-14s %.8f %-10s src=%s mode=%.4f' % (
        sid, r.get('solverUsername'), r.get('officialScore') or float('nan'),
        r.get('promotionStatus') or r.get('status'), (r.get('promotedSourceRef') or '')[:8],
        modeindex(d)))

if len(ids) == 2:
    a, b = ids
    ra, da = pp(a); rb, db = pp(b)
    print()
    print('%-9s %10s %10s %8s | %10s %10s %8s | %7s %7s' % (
        'prompt', 'A mtp', 'B mtp', 'cand%', 'A ser', 'B ser', 'ser%', 'A Mbar', 'B Mbar'))
    cand = []; ser = []
    sfA = []; sfB = []
    for n in ['plutarch', 'drama', 'travel', 'beagle', 'republic', 'essays', 'medicine', 'botany']:
        ea, eb = da[n], db[n]
        c = (ea['mtp_seconds_per_token_mean'] - eb['mtp_seconds_per_token_mean']) / ea['mtp_seconds_per_token_mean'] * 100
        s = (ea['serial_seconds_per_token_mean'] - eb['serial_seconds_per_token_mean']) / ea['serial_seconds_per_token_mean'] * 100
        cand.append(c); ser.append(s)
        print('%-9s %10.8f %10.8f %+8.4f | %10.8f %10.8f %+8.4f | %7.4f %7.4f' % (
            n, ea['mtp_seconds_per_token_mean'], eb['mtp_seconds_per_token_mean'], c,
            ea['serial_seconds_per_token_mean'], eb['serial_seconds_per_token_mean'], s,
            ea['effective_mean_draft_len'], eb['effective_mean_draft_len']))
    print('cand leg  B faster by %+.4f %%  sd %.4f' % (statistics.mean(cand), statistics.pstdev(cand)))
    print('ser  leg  B faster by %+.4f %%  sd %.4f' % (statistics.mean(ser), statistics.pstdev(ser)))
    # schedule identity
    same = all(abs(da[n]['effective_mean_draft_len'] - db[n]['effective_mean_draft_len']) < 1e-9 for n in da)
    print('schedule identical on all eight prompts:', 'YES' if same else 'NO')
    # serial-free: use A's serial for both
    for n in da:
        sfA.append(da[n]['serial_seconds_per_token_mean'] / da[n]['mtp_seconds_per_token_mean'])
        sfB.append(da[n]['serial_seconds_per_token_mean'] / db[n]['mtp_seconds_per_token_mean'])
    print('serial-free (A serial pinned)  A %.8f  B %.8f  gap %+.4f %%' % (
        median8(sfA), median8(sfB), (median8(sfB) - median8(sfA)) / median8(sfA) * 100))
    print('published  A %.8f  B %.8f  gap %+.4f %%' % (
        ra['officialScore'], rb['officialScore'],
        (rb['officialScore'] - ra['officialScore']) / ra['officialScore'] * 100))
    print('F83-marginal-weighted cand delta %+.4f %%' % sum(
        MW[n] * ((da[n]['mtp_seconds_per_token_mean'] - db[n]['mtp_seconds_per_token_mean'])
                 / da[n]['mtp_seconds_per_token_mean'] * 100) for n in MW))
