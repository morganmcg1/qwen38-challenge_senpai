import json, math, statistics

B = json.load(open('/tmp/yukon-board/full.json'))
rows = B['submissions']
by = {}
for r in rows:
    sid = (r.get('id') or '')[:8]
    m = r.get('officialMetrics') or {}
    pp = m.get('per_prompt')
    if not pp:
        continue
    by[sid] = r

NAMES = {'919318e1':'beagle','192fb621':'botany','4b9e88cd':'drama','a2ea8b60':'essays',
         '00142a44':'medicine','c1ec5866':'plutarch','ea82dcb5':'republic','3b10cb4d':'travel'}
W83 = {'beagle':0.4862,'medicine':0.2508,'essays':0.1598,'botany':0.0124,'republic':0.0100,
       'plutarch':0.0,'drama':0.0,'travel':0.0}
MODEW = {'plutarch':-0.3852,'drama':0.0215,'travel':0.4945,'beagle':0.2068,
         'medicine':-0.1480,'republic':-0.0917,'essays':-0.0041,'botany':-0.0939}

def vec(sid):
    r = by[sid]
    d = {}
    for e in r['officialMetrics']['per_prompt']:
        n = NAMES.get(e['prompt_sha256'][:8])
        if n:
            d[n] = e
    return r, d

def summary(sid):
    r, d = vec(sid)
    cand = {n: e['mtp_seconds_per_token_mean'] for n, e in d.items()}
    ser = {n: e['serial_seconds_per_token_mean'] for n, e in d.items()}
    idx = sum(MODEW[n]*100*math.log(cand[n]) for n in cand)
    return dict(sid=sid, user=r.get('solverUsername'), score=r.get('officialScore'),
                status=r.get('status'), src=(r.get('promotedSourceRef') or '')[:8],
                cand=cand, ser=ser, mode=idx,
                cmean=statistics.fmean(cand.values()),
                smean=statistics.fmean(ser.values()),
                eff={n: e['effective_mean_draft_len'] for n, e in d.items()},
                nd={n: e['non_drafting_round_count'] for n, e in d.items()})

def compare(a, b, label):
    A, Bb = summary(a), summary(b)
    print('='*100)
    print(label)
    print(f'  {a} {A["user"]:<16} score {A["score"]} {A["status"]:<9} src {A["src"]} mode {A["mode"]:+.4f}')
    print(f'  {b} {Bb["user"]:<16} score {Bb["score"]} {Bb["status"]:<9} src {Bb["src"]} mode {Bb["mode"]:+.4f}')
    print(f'  mode delta {Bb["mode"]-A["mode"]:+.4f}  (one state step = 1.000, same-state sd 0.116)')
    print(f'  {"prompt":<10} {"cand A":>12} {"cand B":>12} {"cand d%":>9} {"ser d%":>9} {"eff A":>8} {"eff B":>8} {"ndA":>5} {"ndB":>5}')
    tot = 0.0
    f83 = 0.0
    for n in ['beagle','medicine','essays','botany','republic','plutarch','drama','travel']:
        if n not in A['cand'] or n not in Bb['cand']:
            continue
        dc = (Bb['cand'][n]/A['cand'][n]-1)*100
        ds = (Bb['ser'][n]/A['ser'][n]-1)*100
        tot += dc
        f83 += -dc*W83[n]
        print(f'  {n:<10} {A["cand"][n]:12.8f} {Bb["cand"][n]:12.8f} {dc:+9.4f} {ds:+9.4f} '
              f'{A["eff"][n]:8.4f} {Bb["eff"][n]:8.4f} {A["nd"][n]:5d} {Bb["nd"][n]:5d}')
    print(f'  unweighted 8-prompt candidate mean delta  {(Bb["cmean"]/A["cmean"]-1)*100:+.4f} %  '
          f'(se of an 8-prompt mean = 0.0187 %, F154)')
    print(f'  F83-weighted candidate SPEEDUP (B faster = positive) {f83:+.4f} %')
    print(f'  serial 8-prompt mean delta                {(Bb["smean"]/A["smean"]-1)*100:+.4f} %  '
          f'(pure lottery, se 0.0533 %)')
    print(f'  published median delta                    '
          f'{(Bb["score"]/A["score"]-1)*100:+.4f} %')

# nagaral's own claimed isolated pair for the E87 probe-select restore
compare('2d02ef0b', '555cb9e3', 'NAGARAL CLAIMED ISOLATED PAIR: without -> with E87 probe-select (both on a0f8588)')
# francip's promotion of the probe-select
compare('4d3a03aa', 'bc070b7b', 'FRANCIP: 4d3a03aa -> bc070b7b (claimed +0.72 % probe-select promotion)')
# the new crown against the old crown
compare('48423d09', '0b8602e1', 'CROWN MOVE: 48423d09 -> 0b8602e1 (three restorations)')
# ours against the new crown
compare('0c6191b7', '0b8602e1', 'OUR BEST CANDIDATE 0c6191b7 vs THE NEW CROWN 0b8602e1')
