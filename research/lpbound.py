import json
B = json.load(open('/tmp/board4.json'))['submissions']
NAME = {'c1ec5866':'plutarch','4b9e88cd':'drama','3b10cb4d':'travel','919318e1':'beagle',
        '00142a44':'medicine','ea82dcb5':'republic','a2ea8b60':'essays','192fb621':'botany'}
ORDER=['plutarch','drama','travel','beagle','medicine','republic','essays','botany']
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
                    nd=v.get('non_drafting_round_count') or 0, head=(v.get('head_provenance_sha256') or 'x')[:8])
    if ok: recs[s['id'][:8]]={'sc':s.get('officialScore'),'p':row,'user':s.get('solverUsername','')}

# Feasibility system for one receipt.
#   per-round model:  mtp_p * (1 + a_p) = R_p = s + h*q_p      (s,h shared across prompts)
#   definitional:     h >= 0            (a proposed draft row cannot make a round faster)
#                     0 <= a_p <= q_p   (cannot accept more drafts than were proposed)
#                     N_p = 512/(1+a_p) ; q_p*N_p <= 8*(N_p - nd_p)  (max depth 8)
#                     q_p*N_p >= 1*(N_p - nd_p)                      (a drafting round proposes >=1)
# Given (s,h) every a_p is determined, so the feasible set is an intersection of half-planes in (s,h).
def feasible(rec, s, h):
    for n in ORDER:
        d=rec['p'][n]; q=d['q']; mtp=d['mtp']; nd=d['nd']
        a=(s+h*q)/mtp-1.0
        if a < -1e-12 or a > q+1e-12: return False
        N=512.0/(1.0+a)
        if nd>N+1e-9: return False
        if q*N > 8.0*(N-nd)+1e-9: return False
        if nd>0 and q*N < (N-nd)-1e-9: return False
    return True

def solve(tag, rec, verbose=True):
    lo_s,hi_s=0.0,0.06; lo_h,hi_h=0.0,0.02
    pts=[]
    NS,NH=1200,900
    for i in range(NS+1):
        s=lo_s+(hi_s-lo_s)*i/NS
        for j in range(NH+1):
            h=lo_h+(hi_h-lo_h)*j/NH
            if feasible(rec,s,h): pts.append((s,h))
    if not pts:
        print(tag,'INFEASIBLE'); return
    ss=[p[0] for p in pts]; hh=[p[1] for p in pts]
    print('%s  n_feasible=%d' % (tag,len(pts)))
    print('   s in [%.7f, %.7f]   h in [%.7f, %.7f]   8h/s in [%.3f, %.3f]'
          % (min(ss),max(ss),min(hh),max(hh),
             8*min(hh)/max(ss), 8*max(hh)/max(min(ss),1e-9)))
    if not verbose: return
    print('   %-9s %7s %10s %10s %10s %8s %8s %8s %8s' %
          ('prompt','q','mtp','a_lo','a_hi','N_lo','N_hi','R_lo','R_hi'))
    for n in ORDER:
        d=rec['p'][n]
        A=[(s+h*d['q'])/d['mtp']-1.0 for (s,h) in pts]
        alo,ahi=min(A),max(A)
        print('   %-9s %7.3f %10.7f %10.4f %10.4f %8.1f %8.1f %8.5f %8.5f'
              % (n,d['q'],d['mtp'],alo,ahi,512/(1+ahi),512/(1+alo),
                 d['mtp']*(1+alo),d['mtp']*(1+ahi)))
    # published-gain sensitivity: a 1% cut in R at fixed a
    tot_lo=sum(512.0/(1.0+max((min(( (s+h*rec['p'][n]['q'])/rec['p'][n]['mtp']-1.0) for (s,h) in pts),0))) for n in ORDER)
    print('   TOTAL ROUNDS over 8 prompts in [%d, %d]  (512 tokens each, 4096 tokens total)'
          % (sum(int(512/(1+max((s+h*rec['p'][n]['q'])/rec['p'][n]['mtp']-1.0 for (s,h) in pts))) for n in ORDER),
             sum(int(512/(1+min((s+h*rec['p'][n]['q'])/rec['p'][n]['mtp']-1.0 for (s,h) in pts))) for n in ORDER)))

print('=== RIGOROUS FEASIBLE SET  (only definitional constraints: h>=0, 0<=a<=q, depth<=8, nd) ===')
print()
for wid in ('5a9f130a','ec24d591','ac89ef87'):
    if wid in recs:
        r=recs[wid]
        solve('--- %s %s sc=%.5f' % (wid,r['user'],r['sc']), r)
        print()
