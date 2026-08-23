import json, statistics as st, collections

B = json.load(open('/tmp/board4.json'))['submissions']
NAME = {'c1ec5866':'plutarch','4b9e88cd':'drama','3b10cb4d':'travel','919318e1':'beagle',
        '00142a44':'medicine','ea82dcb5':'republic','a2ea8b60':'essays','192fb621':'botany'}
def pname(k):
    for pre,n in NAME.items():
        if k.startswith(pre): return n
    return None
ORDER=['plutarch','drama','travel','beagle','medicine','republic','essays','botany']

recs=[]
for s in B:
    om=s.get('officialMetrics') or {}
    pp=om.get('per_prompt') or []
    sc=s.get('officialScore')
    if len(pp)!=8 or sc is None or om.get('mtp_depth')!=8: continue
    row={}; ok=True
    for v in pp:
        n=pname(v.get('prompt_sha256',''))
        if n is None or not v.get('mtp_seconds_per_token_mean'): ok=False; break
        row[n]=dict(q=v.get('effective_mean_draft_len'), mtp=v['mtp_seconds_per_token_mean'],
                    head=(v.get('head_provenance_sha256') or 'none')[:8])
    if ok: recs.append({'id':s['id'][:8],'sc':sc,'p':row,'user':s.get('solverUsername','')})

sub=[r for r in recs if r['p']['beagle']['head']=='559b24eb']
print('depth-8 receipts on the shipped head 559b24eb:', len(sub))

print()
print('=== REVEALED PREFERENCE: what schedule setting do the BEST solvers choose? ===')
print('   (bias-free: no min-of-bin, just the distribution of q among high scorers)')
for thr in (3.0,3.3,3.5,3.6,3.65,3.7):
    v=[r for r in sub if r['sc']>=thr]
    if len(v)<3: continue
    c=collections.Counter(round(r['p']['beagle']['q'],3) for r in v)
    tot=len(v)
    top=' | '.join('q=%.3f %d%%'%(q,round(100*k/tot)) for q,k in c.most_common(4))
    print('  score >= %.2f  n=%4d  distinct users %2d  ->  %s' % (thr,tot,len(set(r['user'] for r in v)),top))

print()
print('=== the two big bins, compared fairly (comparable n) ===')
for qv in (4.382,4.533):
    v=[r for r in sub if abs(r['p']['beagle']['q']-qv)<0.01]
    m=sorted(r['p']['beagle']['mtp'] for r in v)
    print('  q=%.3f  n=%3d  min %.7f  p05 %.7f  p25 %.7f  median %.7f  best score %.5f'
          % (qv,len(v),m[0],m[int(.05*len(m))],m[int(.25*len(m))],st.median(m),
             max(r['sc'] for r in v)))

print()
print('=== the small high-q bins: could they win with more draws? ===')
print('   %8s %5s %12s %12s %10s %-16s' % ('q','n','min mtp','vs q=4.382','best score','best-mtp holder'))
ref=min(r['p']['beagle']['mtp'] for r in sub if abs(r['p']['beagle']['q']-4.382)<0.01)
bins=collections.defaultdict(list)
for r in sub: bins[round(r['p']['beagle']['q'],3)].append(r)
for q in sorted(bins):
    v=bins[q]
    if len(v)<2 or q<4.3: continue
    b=min(v,key=lambda r:r['p']['beagle']['mtp'])
    print('   %8.3f %5d %12.7f %+11.2f %% %10.5f %-16s' % (q,len(v),b['p']['beagle']['mtp'],
          100*(b['p']['beagle']['mtp']/ref-1), b['sc'], b['user']))

print()
print('=== same question, weighted-prompt view: which q wins on the 5 SCORING prompts? ===')
W={'beagle':0.4741,'medicine':0.1951,'essays':0.1658,'republic':0.0963,'botany':0.0508}
print('   %8s %5s %14s %10s' % ('beagle q','n','best wtd mtp','vs 4.382'))
def wmtp(r): return sum(W[n]*r['p'][n]['mtp'] for n in W)/sum(W.values())
refw=min(wmtp(r) for r in sub if abs(r['p']['beagle']['q']-4.382)<0.01)
for q in sorted(bins):
    v=bins[q]
    if len(v)<2 or q<4.3: continue
    bw=min(wmtp(r) for r in v)
    print('   %8.3f %5d %14.7f %+9.2f %%' % (q,len(v),bw,100*(bw/refw-1)))

print()
print('=== who are the top 10 receipts overall, and what q do they run? ===')
print('   %-9s %-16s %8s %8s %10s' % ('id','user','score','beagle q','beagle mtp'))
for r in sorted(recs,key=lambda r:-r['sc'])[:10]:
    print('   %-9s %-16s %8.5f %8.3f %10.7f' % (r['id'],r['user'],r['sc'],r['p']['beagle']['q'],r['p']['beagle']['mtp']))
