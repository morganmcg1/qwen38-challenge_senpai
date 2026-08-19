"""Cross-arm golden diagnostic: candidate timed against the BASE's reference rows.

Strictly stronger than each arm matching rows it generated itself.
"""
import json

b = json.load(open('/tmp/e33_diag/timed-base.json'))
c = json.load(open('/tmp/e33_diag/timed-cand.json'))

KEYS = [
    'all_tokens_matched', 'parity_all_ok', 'residual_divergence_count',
    'round_count', 'effective_mean_draft_len', 'effective_max_draft_len',
    'effective_draft_lengths', 'non_drafting_round_count',
    'accepted_draft_total', 'rejected_draft_total', 'declared_rows_total',
    'reference_checked_row_total', 'rejected_rows_reference_checked',
    'emitted_token_total', 'target_tail_total', 'target_cache_offset_final',
    'verify_block_replayed_round_count', 'max_rejected_tail_logit_delta',
    'first_divergence_index', 'seed_token_count', 'decode_token_count',
    'uses_native_mtp_head',
]

print('%-38s %-26s %-26s %s' % ('field', 'base 4e5dc2b', 'cand 20bdd25', 'same'))
bad = []
for k in KEYS:
    vb, vc = b.get(k, '<absent>'), c.get(k, '<absent>')
    same = vb == vc
    if not same:
        bad.append(k)
    print('%-38s %-26s %-26s %s'
          % (k, vb, vc, 'YES' if same else '*** NO ***'))

print()
print('DIFFERING LEDGER FIELDS: %s' % (bad or 'none'))
print()
print('traced run -- diagnostic only, NOT a timing arm')
print('  parent_measured_seconds_per_token base %.8f cand %.8f'
      % (b['parent_measured_seconds_per_token'],
         c['parent_measured_seconds_per_token']))
print('  decode_seconds        base %.4f cand %.4f'
      % (b['decode_seconds'], c['decode_seconds']))
print('  seed_prefill_seconds  base %.4f cand %.4f'
      % (b['seed_prefill_seconds'], c['seed_prefill_seconds']))

for name in ('golden', 'timed-base', 'timed-cand'):
    print('%-12s provenance: %s'
          % (name, open('/tmp/e33_diag/%s.provenance' % name).read().strip()))
