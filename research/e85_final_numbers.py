#!/usr/bin/env python3
"""Exact figures quoted in the E85 terminal result."""
from __future__ import annotations

import json
import pathlib

ART = pathlib.Path("research/e85-artifacts")

strat = json.loads((ART / "stratified.json").read_text())
pairs = json.loads((ART / "round-pairs.json").read_text())

for name in ("clean", "contaminated"):
    arm = strat["strata"][name]["arm"]
    null = strat["strata"][name]["null_serial"]
    print(f"[{name}] legs={strat['strata'][name]['legs']}")
    print(f"  arm  delta {arm['delta_us_per_token']:+.2f} us/tok "
          f"({arm['pct_of_base']:+.4f} %) CI "
          f"[{arm['ci95_lo_us_per_token']:+.1f},{arm['ci95_hi_us_per_token']:+.1f}] "
          f"t={arm['t']:+.2f}")
    print(f"  null delta {null['delta_us_per_token']:+.2f} us/tok "
          f"({null['pct_of_base']:+.4f} %) t={null['t']:+.2f}")
    print(f"  mean position base={arm['mean_position_base']:.2f} "
          f"ab={arm['mean_position_treat']:.2f}")
    print(f"  base_mean={arm['base_mean']:.8f} treat_mean={arm['treat_mean']:.8f}")

allv = strat["all_legs"]
print(f"[all] arm {allv['arm']['delta_us_per_token']:+.2f} us/tok "
      f"({allv['arm']['pct_of_base']:+.4f} %) "
      f"null {allv['null_serial']['delta_us_per_token']:+.2f} us/tok "
      f"({allv['null_serial']['pct_of_base']:+.4f} %)")
print(f"heterogeneity t={strat['heterogeneity']['t']:+.2f}")
print(f"contaminated draw {strat['contaminated_draw']}")
print(f"threshold {strat['threshold_us_per_round']:.0f} us/round")

pe = pairs["paired_effect_clustered"]
pn = pairs["session_null_clustered"]
print(f"paired clustered {pe['median_us_per_token']:+.2f} "
      f"t-CI [{pe['t_ci95_lo_us_per_token']:+.1f},{pe['t_ci95_hi_us_per_token']:+.1f}]")
print(f"paired null clustered {pn['median_us_per_token']:+.2f}")
print(f"variance ratio {pairs['cluster_variance_check']['ratio']:.1f}x")
print(f"leg-total/paired {pairs['leg_total_over_paired_ratio']:.3f}")
