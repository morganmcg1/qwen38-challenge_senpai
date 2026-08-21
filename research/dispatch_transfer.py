"""Calibrate the ranked transfer of a dispatch-count deletion.

Thorfinn measured E101 chain C locally. Rival 9612d3ba shipped the same
mechanism on the ranked M5 on top of 51b9bf85. Both sides are now known, so the
Finding 22 LATENCY multiplier of 2.40 can be checked directly.
"""

# ---- local side, thorfinn's production block e101r5, STRICT 3v3 ----
loc_round_off = 152388.8          # us per round, gate off
loc_round_on = 152137.2           # us per round, gate on
loc_drafts_per_round = 6.4254
loc_delta_round = loc_round_off - loc_round_on          # us per round
loc_delta_draft = loc_delta_round / loc_drafts_per_round
loc_pct = 100.0 * loc_delta_round / loc_round_off

# ---- ranked side, 9612d3ba vs 51b9bf85, candidate leg only ----
# per prompt: (ranked round us, drafts per round, measured candidate-leg pct)
RANKED = {
    "beagle":   (57502.0, 4.3818181818, 0.1837),
    "republic": (60007.0, 4.9892473118, 0.1496),
    "essays":   (60925.0, 5.0869565217, 0.1510),
    "medicine": (62009.0, 5.2555555556, 0.1771),
    "botany":   (68006.0, 6.1481481481, 0.1838),
}
RANKED_MEAN_PCT = 0.1572          # all eight prompts, sd 0.0236

print("LOCAL  (M4 Pro, thorfinn e101r5 STRICT)")
print(f"  delta per round      {loc_delta_round:8.1f} us")
print(f"  delta per draft      {loc_delta_draft:8.2f} us")
print(f"  local percent        {loc_pct:8.4f} %")
print()
print("RANKED (M5, 9612d3ba vs 51b9bf85, candidate leg, schedule identical)")
print(f"{'prompt':10s} {'round us':>10s} {'d/rnd':>7s} {'pct':>8s}"
      f" {'us/round':>9s} {'us/draft':>9s}")
tot_draft = 0.0
for name, (rnd, d, pct) in RANKED.items():
    us_round = pct / 100.0 * rnd
    us_draft = us_round / d
    tot_draft += us_draft
    print(f"{name:10s} {rnd:10.0f} {d:7.3f} {pct:8.4f} "
          f"{us_round:9.1f} {us_draft:9.2f}")
rk_draft = tot_draft / len(RANKED)
print(f"{'mean':10s} {'':10s} {'':7s} {'':8s} {'':9s} {rk_draft:9.2f}")
print()

abs_ratio = rk_draft / loc_delta_draft
pct_ratio = RANKED_MEAN_PCT / loc_pct
print("TRANSFER")
print(f"  absolute us per draft, ranked / local   {abs_ratio:6.3f}")
print(f"  percent, ranked / local                 {pct_ratio:6.3f}")
print(f"  Finding 22 LATENCY class predicted      {2.401:6.3f}")
print(f"  overprediction factor                   {2.401 / pct_ratio:6.2f} x")
print()

# What the advisor told thorfinn to expect, and what he should have expected.
print("REPRICING E101 CHAIN C")
print(f"  advisor frame, 1:1 absolute us          +0.323 %")
print(f"  corrected, {pct_ratio:.2f} x local percent          "
      f"{pct_ratio * loc_pct:+.3f} %")
print(f"  rival measured on the ranked box        +{RANKED_MEAN_PCT:.3f} %")
print()

# Reprice every LATENCY-class line of the Finding 22 table.
print("REPRICING THE FINDING 22 LATENCY TABLE")
print(f"{'family':28s} {'local %':>8s} {'old x2.40':>10s} "
      f"{'new x1.00':>10s} {'owner':>10s}")
LAT = [
    ("SDPA over FA history", 0.993, "closed"),
    ("fused residual + RMSNorm", 0.605, "edward"),
    ("GDN prework", 0.426, "E105"),
    ("q/k norm + RoPE", 0.117, "E105"),
    ("KV cache write", 0.070, "E105"),
    ("MTP top-2", 0.044, "-"),
]
e105 = 0.0
for name, loc, owner in LAT:
    new = loc * pct_ratio
    if owner == "E105":
        e105 += new
    print(f"{name:28s} {loc:8.3f} {loc * 2.401:10.3f} {new:10.3f} {owner:>10s}")
print()
print(f"  E105 target pool re-prices  1.473 %  ->  {e105:.3f} %")
print(f"  E105 stated minimum useful stacked ceiling      0.600 %")
print(f"  headroom above its own bar                      {e105 - 0.600:+.3f} %")
print()

# Chain A, thorfinn's queued follow-up, on the same corrected frame.
chain_a_census = 40.42            # us per draft, OPS=0 census
chain_c_census = 104.21           # us per draft, OPS=0 census, net
census_to_prod = loc_delta_draft / chain_c_census
a_local_prod = chain_a_census * census_to_prod
a_local_pct = 100.0 * a_local_prod * loc_drafts_per_round / loc_round_off
print("CHAIN A, thorfinn's queued follow-up")
print(f"  census us per draft                    {chain_a_census:7.2f}")
print(f"  census -> production deflator          {census_to_prod:7.3f}")
print(f"  implied local production us per draft  {a_local_prod:7.2f}")
print(f"  implied local percent                  {a_local_pct:7.3f} %")
print(f"  implied RANKED percent                 "
      f"{a_local_pct * pct_ratio:7.3f} %")
print(f"  serial-free detection floor              0.160 %")
print(f"  published detection floor                0.277 %")
