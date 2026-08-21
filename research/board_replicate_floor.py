"""Resolution floor of one ranked pair, measured on byte-identical scored trees.

Statistic = mean of the 4th and 5th sorted raw_p (the published score rule).
Mode is inferred by the empty band alphonse found in the drafting-leg gap.

Inputs, both produced outside this repository:

    /tmp/yukon-board/full.json    the whole Yukon board payload
    /tmp/yukon-board/treedigest.json
        {submission id: output of
         `git ls-tree <upstream/submissions/<id>> Sources Vendor
          mtp-head.manifest.json`}

Two runs are replicates when all three subtree object ids agree and all eight
`effective_mean_draft_len` values agree to 1e-3, so the scored code and the
draft schedule were identical and only the run differed.
"""
import json, itertools, statistics

payload = json.load(open('/tmp/yukon-board/full.json'))
rows = payload
for k in ('submissions', 'rows', 'data', 'items'):
    if isinstance(rows, dict) and k in rows:
        rows = rows[k]
        break
rows = [r for r in rows if isinstance(r, dict)]
dig = json.load(open('/tmp/yukon-board/treedigest.json'))

PLUT = 'c1ec5866'


def perprompt(r):
    om = r.get('officialMetrics') or {}
    pp = (om.get('per_prompt') if isinstance(om, dict) else None) or []
    out = {}
    for e in pp:
        if not isinstance(e, dict):
            continue
        s = e.get('prompt_sha256') or e.get('promptSha256')
        if not s:
            continue
        out[s[:8]] = e
    return out


cand = []
for r in rows:
    sid = r.get('id') or r.get('submissionId')
    if sid not in dig:
        continue
    sc = r.get('officialScore')
    if not isinstance(sc, (int, float)):
        continue
    pp = perprompt(r)
    if len(pp) != 8:
        continue
    try:
        ratios = {k: v['raw_ratio_of_means'] for k, v in pp.items()}
        lens = {k: v['effective_mean_draft_len'] for k, v in pp.items()}
        spt = {k: v['mtp_seconds_per_token_mean'] for k, v in pp.items()}
        serial = {k: v['serial_seconds_per_token_mean'] for k, v in pp.items()}
    except KeyError:
        continue
    cand.append(dict(id=sid, tree=dig[sid], ratios=ratios, lens=lens, spt=spt,
                     serial=serial, score=sc, created=r.get('createdAt') or ''))

print('rows with digest and 8 prompts:', len(cand))

groups = {}
for c in cand:
    groups.setdefault(c['tree'], []).append(c)

pairs = []
for tree, g in groups.items():
    if len(g) < 2:
        continue
    for a, b in itertools.combinations(g, 2):
        if any(abs(a['lens'][k] - b['lens'][k]) > 1e-3 for k in a['lens']):
            continue
        pairs.append((a, b))

print('byte-identical replicate pairs with matching draft lengths:', len(pairs))


boardmean = {}
for k in cand[0]['spt']:
    boardmean[k] = statistics.mean(c['serial'][k] for c in cand)


def stat(c):
    v = sorted(c['ratios'].values())
    return (v[3] + v[4]) / 2.0


def statfree(c):
    v = sorted(boardmean[k] / c['spt'][k] for k in c['spt'])
    return (v[3] + v[4]) / 2.0


def draftgap(a, b):
    ks = [k for k in a['spt'] if k != PLUT]
    return statistics.mean(abs(a['spt'][k] / b['spt'][k] - 1.0) for k in ks) * 100.0


tab = []
for a, b in pairs:
    tab.append((draftgap(a, b), abs(stat(a) / stat(b) - 1.0) * 100.0,
                (statfree(a) / statfree(b) - 1.0) * 100.0, a['id'][:8], b['id'][:8]))
tab.sort()
print()
print('  draft-leg gap %   |published d| %   serial-free d %   pair')
for t in tab:
    print(f'  {t[0]:12.3f}    {t[1]:12.4f}   {t[2]:+10.4f}   {t[3]} {t[4]}')

same = [t for t in tab if t[0] < 0.7]
cross = [t for t in tab if t[0] >= 0.7]
print()
print('same-mode pairs :', len(same))
if same:
    a = sorted(t[1] for t in same)
    print('  median |score delta| % :', round(statistics.median(a), 4))
    print('  p75                    :', round(a[int(0.75 * (len(a) - 1))], 4))
    print('  max                    :', round(max(a), 4))
    s = [t[2] for t in same]
    print('  serial-free |median| % :', round(statistics.median(sorted(abs(x) for x in s)), 4))
    print('  serial-free max %      :', round(max(abs(x) for x in s), 4))
    print('  signed sd %            :', round(statistics.pstdev(s), 4))
    print('  implied per-run sd %   :', round(statistics.pstdev(s) / (2 ** 0.5), 4))
print('cross-mode pairs:', len(cross))
if cross:
    a = sorted(t[1] for t in cross)
    print('  median |score delta| % :', round(statistics.median(a), 4))
