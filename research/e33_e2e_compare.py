"""E33 deliverable (d): both legs, reported separately, from the ABBA arms.

Every arm must carry stale_metallib_warnings=0; benchmark-qwen-mtp.sh does not
rebuild mlx.metallib, so an arm without that gate may have timed the other
arm's kernel.
"""
import json
import os
import statistics as st

ROOT = '.mlxfast-private/e33-e2e'
ORDER = ['base-3', 'cand-2', 'cand-3', 'base-4']  # ABBA
ARMS = {'base-3': 'base', 'base-4': 'base', 'cand-2': 'cand', 'cand-3': 'cand'}


def ident(tag):
    out = {}
    path = os.path.join(ROOT, tag, 'identity.txt')
    for line in open(path):
        if '=' in line:
            k, _, v = line.strip().partition('=')
            out[k.strip()] = v.strip()
    return out


rows = []
for tag in ORDER:
    p = os.path.join(ROOT, tag, 'score.json')
    if not os.path.exists(p):
        print('MISSING %s' % p)
        continue
    m = json.load(open(p))['metrics']
    i = ident(tag)
    rows.append(dict(tag=tag, arm=ARMS[tag], m=m, i=i))

print('%-8s %-5s %8s %8s %6s %12s %12s %9s' %
      ('tag', 'arm', 'entryC', 'exitC', 'stale', 'serial_s/tok', 'mtp_s/tok', 'speedup'))
for r in rows:
    print('%-8s %-5s %8.3f %8.3f %6s %12.8f %12.8f %9.6f' %
          (r['tag'], r['arm'], float(r['i']['gpu_temp_c_entry']), float(r['i']['gpu_temp_c_exit']),
           r['i'].get('stale_metallib_warnings', '?'),
           r['m']['serial_seconds_per_token'], r['m']['mtp_seconds_per_token'],
           r['m']['mtp_decode_speedup']))

bad = [r['tag'] for r in rows if r['i'].get('stale_metallib_warnings') != '0']
if bad:
    print('\nINVALID ARMS (stale metallib): %s' % bad)

print()
for leg, key in (('SERIAL leg (M=1, global control)', 'serial_seconds_per_token'),
                 ('MTP leg', 'mtp_seconds_per_token')):
    b = [r['m'][key] for r in rows if r['arm'] == 'base']
    c = [r['m'][key] for r in rows if r['arm'] == 'cand']
    if not b or not c:
        continue
    mb, mc = st.mean(b), st.mean(c)
    print('%-34s base %.8f  cand %.8f  cand/base %.5f (%+.3f %%)'
          % (leg, mb, mc, mc / mb, 100 * (mc / mb - 1)))
    print('%-34s   base arms %s' % ('', ['%.8f' % x for x in b]))
    print('%-34s   cand arms %s' % ('', ['%.8f' % x for x in c]))

print()
print('Correctness and provenance (must be identical across all arms):')
for k in ('all_tokens_matched', 'residual_divergence_count', 'public_drift_tripwire_passed',
          'effective_mean_draft_len', 'accepted_draft_rate', 'mtp_depth', 'decode_tokens',
          'head_provenance_sha256', 'uses_pinned_mtp_head'):
    vals = {r['tag']: r['m'].get(k) for r in rows}
    uniq = set(map(str, vals.values()))
    print('  %-30s %s   %s' % (k, 'IDENTICAL' if len(uniq) == 1 else 'DIFFERS',
                               next(iter(uniq)) if len(uniq) == 1 else vals))

temps = [float(r['i']['gpu_temp_c_entry']) for r in rows]
print()
print('entry-temperature spread across arms: %.3f C (%.3f .. %.3f)'
      % (max(temps) - min(temps), min(temps), max(temps)))
print('cool_gate_passed_real_gate=false ; gate_qualified_for_timing=false')
