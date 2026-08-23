import json, datetime as dt, collections
B = json.load(open('/tmp/board4.json'))['submissions']
def ts(x):
    if not x: return None
    try: return dt.datetime.fromisoformat(x.replace('Z','+00:00'))
    except Exception: return None

print('=== our own submissions ===')
mine=[s for s in B if s.get('solverUsername')=='morganmcg1']
mine.sort(key=lambda s: s.get('createdAt') or '')
print('%-9s %-11s %-19s %-19s %10s %10s %-9s' %
      ('id','status','created','updated','dur(min)','score','commit'))
for s in mine[-14:]:
    c=ts(s.get('createdAt')); u=ts(s.get('updatedAt') or s.get('completedAt'))
    d=((u-c).total_seconds()/60) if (c and u) else float('nan')
    print('%-9s %-11s %-19s %-19s %10.1f %10s %-9s' %
          (s['id'][:8], s.get('status') or '?', str(c)[:19], str(u)[:19], d,
           ('%.5f'%s['officialScore']) if s.get('officialScore') else '-',
           (s.get('submissionCommitSha') or '')[:8]))

print()
print('=== distribution of validation duration for COMPLETED submissions, whole board ===')
ds=[]
for s in B:
    if (s.get('status') or '') in ('validating','queued','pending'): continue
    c=ts(s.get('createdAt')); u=ts(s.get('updatedAt') or s.get('completedAt'))
    if c and u:
        m=(u-c).total_seconds()/60
        if 0<m<600: ds.append(m)
ds.sort()
if ds:
    def pct(p): return ds[min(len(ds)-1,int(p*len(ds)))]
    print('  n=%d  min %.1f  p10 %.1f  p25 %.1f  median %.1f  p75 %.1f  p90 %.1f  p95 %.1f  p99 %.1f  max %.1f'
          % (len(ds),ds[0],pct(.10),pct(.25),pct(.50),pct(.75),pct(.90),pct(.95),pct(.99),ds[-1]))
    print('  fraction over 60 min: %.1f%%   over 80 min: %.1f%%   over 100 min: %.1f%%'
          % (100*sum(1 for x in ds if x>60)/len(ds), 100*sum(1 for x in ds if x>80)/len(ds),
             100*sum(1 for x in ds if x>100)/len(ds)))

print()
print('=== anything else currently non-terminal on the board (queue ahead of us)? ===')
live=[s for s in B if (s.get('status') or '') in ('validating','queued','pending','running')]
live.sort(key=lambda s: s.get('createdAt') or '')
for s in live:
    print('  %-9s %-14s %-11s created %s' % (s['id'][:8], s.get('solverUsername',''), s.get('status'), (s.get('createdAt') or '')[:19]))
print('  n live in cached snapshot:', len(live), '(snapshot is stale; use it only for queue shape)')

print()
print('=== status vocabulary seen ===')
print(' ', collections.Counter((s.get('status') or '?') for s in B).most_common())
