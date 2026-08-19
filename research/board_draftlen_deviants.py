import json
import statistics as st
from collections import Counter

rows = json.load(open('/tmp/rows_live.json'))
sel = []
for r in rows:
    om = r.get('officialMetrics') or {}
    pp = om.get('per_prompt')
    if (isinstance(pp, list) and len(pp) == 8 and r.get('officialScore') is not None
            and all(isinstance(p.get('head_provenance_sha256'), str)
                    and p['head_provenance_sha256'].startswith('559b24eb') for p in pp)):
        sel.append(r)

def tab(r):
    return {p['prompt_sha256'][:8]: p for p in r['officialMetrics']['per_prompt']}

prompts = sorted(tab(sel[0]).keys())
order = sorted(prompts, key=lambda q: st.median(
    [tab(r)[q]['serial_seconds_per_token_mean'] / tab(r)[q]['mtp_seconds_per_token_mean']
     for r in sel]))
NAMES = ["plutarch", "drama", "travel", "beagle", "medicine", "essays", "republic", "botany"]
bea, med = order[3], order[4]

print('medicine draftlen counts:',
      Counter(round(tab(r)[med]['effective_mean_draft_len'], 4) for r in sel).most_common(6))
print('beagle   draftlen counts:',
      Counter(round(tab(r)[bea]['effective_mean_draft_len'], 4) for r in sel).most_common(6))

o = next(r for r in sel if str(r.get('submissionCommitSha', '')).startswith('2b0c36a078'))
print('ours: medicine draftlen', tab(o)[med]['effective_mean_draft_len'],
      ' beagle draftlen', tab(o)[bea]['effective_mean_draft_len'])

j = next(r for r in sel if str(r.get('submissionCommitSha', '')).startswith('f422c5a1'))
print('jonathan308 draftlens:', [round(tab(j)[q]['effective_mean_draft_len'], 4) for q in order])
print('jonathan308 mtp ms   :', [round(tab(j)[q]['mtp_seconds_per_token_mean'] * 1000, 3) for q in order])
print('cohort median mtp ms :', [round(st.median([tab(r)[q]['mtp_seconds_per_token_mean']
                                                 for r in sel]) * 1000, 3) for q in order])
print('prompt order         :', NAMES)
print('jonathan308 status:', j.get('status'), '| reason:', repr(j.get('rejectionReason'))[:160])
print('jonathan308 note:', repr(j.get('note'))[:500])

# how many matched rows deviate from the modal draft length on ANY prompt?
modal = {q: Counter(round(tab(r)[q]['effective_mean_draft_len'], 4) for r in sel).most_common(1)[0][0]
         for q in order}
odd = []
for r in sel:
    diffs = [NAMES[i] for i, q in enumerate(order)
             if abs(round(tab(r)[q]['effective_mean_draft_len'], 4) - modal[q]) > 1e-4]
    if diffs:
        odd.append((str(r.get('submissionCommitSha'))[:8], r.get('solverUsername'),
                    round(r['officialScore'], 6), diffs))
print()
print('rows deviating from modal draft length on any prompt:', len(odd), 'of', len(sel))
for t in sorted(odd, key=lambda x: -x[2])[:15]:
    print('  ', t[0], str(t[1])[:16].ljust(16), t[2], t[3])
