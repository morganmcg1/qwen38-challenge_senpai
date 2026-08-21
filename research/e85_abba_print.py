#!/usr/bin/env python3
"""Print one E85 ABBA stats bundle as a compact human-readable report.

    usage: research/e85_abba_print.py SESSION_DIR/stats.json
"""

import json
import sys

d = json.load(open(sys.argv[1]))

print(f"arms {d['arm_a']} vs {d['arm_b']}   legs {d['legs']}")
for arm in (d["arm_a"], d["arm_b"]):
    k = f"arm_{arm}"
    a = d[k]
    print(f"  {arm:5s} mtp={a['mtp_s_per_tok_mean']:.8f} sd={a['mtp_s_per_tok_sd']:.8f}"
          f"  serial={a['serial_s_per_tok_mean']:.8f}"
          f"  ratio={a['ratio_mean']:.4f}"
          f"  dlen={a['mean_draft_len_mean']:.9f}"
          f"  acc={a['accepted_rate_mean']:.9f}"
          f"  matched={a['all_matched']}")
print(f"  drafts_per_token={d['drafts_per_token']:.4f}"
      f"  temp_in_spread={d['temp_in_spread_c']:.2f}C")

print("\nARM CONTRAST on absolute candidate seconds per token"
      "  (negative = the arms are faster)")
for tag in ("block", "ols", "cov"):
    print(f"  [{tag:5s}] {d[tag + '_effect_us_per_token']:+8.2f} us/token"
          f"  CI95 [{d[tag + '_effect_us_per_token_ci95_lo']:+8.2f},"
          f" {d[tag + '_effect_us_per_token_ci95_hi']:+8.2f}]"
          f"  {d[tag + '_effect_pct_of_mtp']:+.4f}%"
          f"  {d[tag + '_effect_us_per_draft']:+8.2f} us/draft"
          f"  {d[tag + '_effect_us_per_buffer']:+7.2f} us/buffer"
          f"  CI95 [{d[tag + '_effect_us_per_buffer_ci95_lo']:+7.2f},"
          f" {d[tag + '_effect_us_per_buffer_ci95_hi']:+7.2f}]")
for tag in ("ols", "cov"):
    m = d[f"{tag}_mtp_s_per_tok"]
    print(f"    {tag:4s} t={m['t']:+.3f} dof={m['dof']}"
          f" resid_sd={m['resid_sd'] * 1e6:.2f}us"
          f" drift_per_leg={m['drift_per_leg'] * 1e6:+.2f}us")
blk = d["block_mtp_s_per_tok"]
print("    block values us =", [round(v * 1e6, 2) for v in blk["values"]])

s = d["ols_serial_s_per_tok"]
print("\nCONTROL: the unchanged depth-0 serial leg, measured in the same runs")
print(f"  contrast {s['effect'] * 1e6:+.2f} us/token"
      f"  ({d['control_serial_effect_pct']:+.4f}%)"
      f"  t={s['t']:+.3f}"
      f"  resid_sd={s['resid_sd'] * 1e6:.2f}us"
      f"  drift_per_leg={s['drift_per_leg'] * 1e6:+.2f}us")

print("\nBEHAVIOUR (must not regress)")
print(f"  effective_mean_draft_len contrast {d['ols_mean_draft_len']['effect']:+.3e}")
print(f"  accepted_draft_rate      contrast {d['ols_accepted_rate']['effect']:+.3e}")

print(f"\nsaving_us_per_buffer = {d['saving_us_per_buffer']:+.2f}"
      f"  CI95 {[round(v, 2) for v in d['saving_us_per_buffer_ci95']]}")
print("VERDICT:", d["verdict"])
