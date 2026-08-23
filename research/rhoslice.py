import json
B = json.load(open('/tmp/board4.json'))['submissions']
NAME = {'c1ec5866':'plutarch','4b9e88cd':'drama','3b10cb4d':'travel','919318e1':'beagle',
        '00142a44':'medicine','ea82dcb5':'republic','a2ea8b60':'essays','192fb621':'botany'}
ORDER=['plutarch','drama','travel','beagle','medicine','republic','essays','botany']
W={'beagle':0.4741,'medicine':0.1951,'essays':0.1658,'republic':0.0963,'botany':0.0508,
   'travel':0.0037,'plutarch':0.0033,'drama':0.0019}
def pname(k):
    for pre,n in NAME.items():
        if k.startswith(pre): return n
recs={}
for s in B:
    om=s.get('officialMetrics') or {}; pp=om.get('per_prompt') or []
    if len(pp)!=8: continue
    row={}; ok=True
    for v in pp:
        n=pname(v.get('prompt_sha256',''))
        if n is None or not v.get('mtp_seconds_per_token_mean'): ok=False; break
        row[n]=dict(q=v.get('effective_mean_draft_len') or 0.0, mtp=v['mtp_seconds_per_token_mean'],
                    nd=v.get('non_drafting_round_count') or 0)
    if ok: recs[s['id'][:8]]={'sc':s.get('officialScore'),'p':row,'user':s.get('solverUsername','')}

def slice_rho(rec, rho):
    """rho = 8h/s.  Then h = rho*s/8 and  1+a_p = s*(1 + rho*q_p/8)/mtp_p.
       Feasible s range from 0<=a_p<=q_p and the nd/depth-8 constraints."""
    lo, hi = 0.0, 1.0
    for n in ORDER:
        d=rec['p'][n]; k=(1.0+rho*d['q']/8.0)/d['mtp']       # 1+a_p = s*k
        lo=max(lo, 1.0/k)                                     # a_p >= 0
        hi=min(hi, (1.0+d['q'])/k)                            # a_p <= q_p
    # plutarch depth<=8 constraint: q*N <= 8*(N-nd), N=512/(1+a)
    d=rec['p']['plutarch']; k=(1.0+rho*d['q']/8.0)/d['mtp']
    if d['nd']>0 and d['q']<8:
        Nmax=8.0*d['nd']/(8.0-d['q'])                         # N >= Nmax
        hi=min(hi, (512.0/Nmax)/k)                            # 1+a <= 512/Nmax
    if lo>hi: return None
    return lo,hi

print('=== WHAT EDWARD\'S ANSWER BUYS: fix rho = 8h/s, and the board pins everything else ===')
print('    rho is the cost model\'s own normalised depth price. FINDING 303: its constructors hold 8h=1.44.')
print()
r=recs['5a9f130a']
print('%6s | %-19s | %-17s | %-15s | %8s | %10s' %
      ('rho','s (fixed/round)','a_beagle','beagle rounds','draft%','tot rounds'))
print('-'*96)
for rho in (0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.72,0.80,1.00,1.20,1.44,1.58):
    sl=slice_rho(r,rho)
    if sl is None:
        print('%6.2f | INFEASIBLE' % rho); continue
    slo,shi=sl
    d=r['p']['beagle']
    a=[(s*(1.0+rho*d['q']/8.0)/d['mtp'])-1.0 for s in (slo,shi)]
    # drafting share of decode time = h*q/(s+h*q) = (rho*q/8)/(1+rho*q/8)
    ds=(rho*d['q']/8.0)/(1.0+rho*d['q']/8.0)
    tot=[sum(512.0/((s*(1.0+rho*r['p'][n]['q']/8.0)/r['p'][n]['mtp'])) for n in ORDER) for s in (shi,slo)]
    print('%6.2f | %.6f-%.6f | %6.3f - %6.3f  | %5.0f - %5.0f   | %7.1f%% | %4.0f-%4.0f' %
          (rho,slo,shi,a[0],a[1],512/(1+a[1]),512/(1+a[0]),100*ds,tot[0],tot[1]))

print()
print('    draft% = share of CANDIDATE-LEG decode seconds spent on proposed draft rows, on beagle')
print('    (= rho*q/8 / (1+rho*q/8); prompt-specific because q differs)')
print()
print('=== the same drafting share on every prompt, at three candidate rho values ===')
print('%-9s %6s' % ('prompt','q'), ''.join('%10s'%('rho=%.2f'%x) for x in (0.30,0.72,1.44)))
for n in ORDER:
    q=r['p'][n]['q']
    print('%-9s %6.3f' % (n,q), ''.join('%9.1f%%'%(100*(x*q/8)/(1+x*q/8)) for x in (0.30,0.72,1.44)))
wq=sum(W[n]*r['p'][n]['q'] for n in ORDER)/sum(W.values())
print('%-9s %6.3f' % ('WEIGHTED',wq), ''.join('%9.1f%%'%(100*(x*wq/8)/(1+x*wq/8)) for x in (0.30,0.72,1.44)))

print()
print('=== CROSS-CHECK: does fabinulleins\' MEASURED zero-draft s=0.0313628 fall in our s-slice? ===')
for rho in (0.0,0.30,0.72,1.44,1.58):
    sl=slice_rho(r,rho)
    if sl: print('   rho=%.2f  our s in [%.6f, %.6f]   fab s=0.0313628 %s'
                 % (rho,sl[0],sl[1],'INSIDE -> not decisive' if sl[0]<=0.0313628<=sl[1] else
                    ('ABOVE our range -> WE ARE FASTER' if 0.0313628>sl[1] else 'BELOW -> we are slower')))
