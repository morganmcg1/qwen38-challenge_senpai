import json

b = json.load(open('/tmp/yukon-board/full.json'))['submissions']
d = {r.get('id', '')[:8]: r for r in b}
names = {'919318e1': 'beagle', '192fb621': 'botany', '4b9e88cd': 'drama', 'a2ea8b60': 'essays',
         '00142a44': 'medicine', 'c1ec5866': 'plutarch', 'ea82dcb5': 'republic', '3b10cb4d': 'travel'}
F83 = {'beagle': 0.4862, 'medicine': 0.2508, 'essays': 0.1598, 'botany': 0.0124,
       'republic': 0.0100, 'drama': 0.0, 'travel': 0.0, 'plutarch': 0.0}


def cand(sid):
    pp = d[sid]['officialMetrics']['per_prompt']
    return {names[x['prompt_sha256'][:8]]: x['mtp_seconds_per_token_mean'] for x in pp}


def raws(sid):
    pp = d[sid]['officialMetrics']['per_prompt']
    return {names[x['prompt_sha256'][:8]]: x['raw_ratio_of_means'] for x in pp}


def weights(sid):
    r = raws(sid)
    s = sorted(r.items(), key=lambda kv: kv[1])
    m = d[sid]['officialScore']
    return {s[3][0]: 0.5 * s[3][1] / m, s[4][0]: 0.5 * s[4][1] / m}


pairs = [
    ('F189 tight launch grid',            '02742bf0', 'ed608e64'),
    ('F192 probe 0.25->0.15, wide base',  'b6cb0fea', '02742bf0'),
    ('F192 probe 0.25->0.15, tight base', 'ed608e64', '08b67f12'),
    ('F194 one-pass table under tight',   'ed608e64', '0b2f0014'),
]

print('%-36s %10s %10s %10s' % ('mechanism', 'F83 cand', 'medpair', 'shift pp'))
print('-' * 70)
for label, a, bb in pairs:
    ca, cb = cand(a), cand(bb)
    w = weights(a)
    dl = {k: 100 * (cb[k] / ca[k] - 1) for k in ca}
    f83 = sum(F83[k] * dl[k] for k in dl)
    mp = sum(w.get(k, 0.0) * dl[k] for k in dl)
    print('%-36s %+10.4f %+10.4f %+10.4f' % (label, f83, mp, mp - f83))

print()
print('per-prompt candidate deltas, positive = slower')
print('%-36s %s' % ('mechanism', ' '.join('%9s' % n for n in
      ['beagle', 'essays', 'medicine', 'republic', 'botany'])))
for label, a, bb in pairs:
    ca, cb = cand(a), cand(bb)
    dl = {k: 100 * (cb[k] / ca[k] - 1) for k in ca}
    print('%-36s %s' % (label, ' '.join('%+9.4f' % dl[n] for n in
          ['beagle', 'essays', 'medicine', 'republic', 'botany'])))

print()
print('=== implied ranked median gain, sign-flipped (candidate faster -> median up) ===')
for label, a, bb in pairs:
    ca, cb = cand(a), cand(bb)
    w = weights(a)
    dl = {k: 100 * (cb[k] / ca[k] - 1) for k in ca}
    mp = sum(w.get(k, 0.0) * dl[k] for k in dl)
    f83 = sum(F83[k] * dl[k] for k in dl)
    print('%-36s  medpair %+7.4f   F83 %+7.4f   published-median actual %+7.4f'
          % (label, -mp, -f83, 100 * (d[bb]['officialScore'] / d[a]['officialScore'] - 1)))
