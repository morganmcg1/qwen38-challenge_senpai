#!/usr/bin/env python3
"""Compact report for a buffer-tax slope JSON produced by e85_tax_slope.py."""
import json
import sys

d = json.load(open(sys.argv[1]))
dpt = d["drafts_per_token"]

print(f"legs {d['legs']}  levels {d['tax_levels']}  matched {d['all_tokens_matched']}")
print(f"draft_len_invariant {d['draft_len_invariant']}  "
      f"accepted_rate_invariant {d['accepted_rate_invariant']}")
print(f"temp_in spread {d['temp_in_spread']:.2f} C  warm_legs {d.get('warm_legs')}  "
      f"warm_floor {d.get('warm_floor_c', float('nan')):.1f} C")
print(f"drafts_per_token {dpt:.6f}")
print()
for k, v in d["per_level"].items():
    print(f"K={k:<4} legs={v['legs']}  mtp={v['mtp_s_per_tok_mean']:.8f} "
          f"sd={v['mtp_s_per_tok_sd']:.8f}  serial={v['serial_s_per_tok_mean']:.8f}")
print()
print("linearity residual us:",
      {k: round(v, 1) for k, v in d["linearity_residual_us"].items()})
print()
hdr = (f"{'estimator':<14}{'us/tok/unit':>12}{'ci_lo':>10}{'ci_hi':>10}"
       f"{'us/buffer':>11}{'ci_lo':>9}{'ci_hi':>9}{'t':>8}{'dof':>5}{'resid_us':>10}")
print(hdr)
for n, e in d["estimators"].items():
    print(f"{n:<14}{e['slope_us_per_token_per_unit']:>12.3f}"
          f"{e['ci95_lo_us_per_token_per_unit']:>10.3f}"
          f"{e['ci95_hi_us_per_token_per_unit']:>10.3f}"
          f"{e['slope_us_per_buffer']:>11.3f}"
          f"{e['ci95_lo_us_per_buffer']:>9.3f}"
          f"{e['ci95_hi_us_per_buffer']:>9.3f}"
          f"{e.get('t', float('nan')):>8.2f}{str(e.get('dof')):>5}"
          f"{1e6 * e.get('resid_sd', 0):>10.1f}")
print()
print("pass slopes us/buffer:",
      [round(s * 1e6 / dpt, 3) for s in d["estimators"]["pass"]["slopes"]])
print("verdict:", d["verdict"])
print(f"claim_excluded_by_ci {d['claim_excluded_by_ci']}  "
      f"overprediction_factor {d['overprediction_factor']:.1f}")
