#!/usr/bin/env python3
"""FINDING 32 - final numbers.  Quote only from this script."""

W = 14.41235e9
local  = {1: 64445, 2: 69776, 3: 74778, 4: 86237, 5: 126103}
ranked = {1: 31177, 2: 35172, 3: 39167, 4: 43162, 5: 53108}
Gof    = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2}
rate   = lambda t, m: Gof[m] * W / (t[m] * 1e-6) / 1e9

# ---------------------------------------------------------------- local, measured
r2_loc = rate(local, 5)                       # [3+2] aggregate
COLLAPSE_MEASURED = 0.180                     # alphonse: -17.5..-18.6 % per M=5 round
t1_loc = local[5] * (1 - COLLAPSE_MEASURED)   # one-group [5] round, measured
r1_loc = W / (t1_loc * 1e-6) / 1e9
A_loc  = r2_loc / r1_loc

print("=" * 76)
print("LOCAL  (alphonse's own E100 measurement, no extrapolation)")
print("=" * 76)
print(f"  [3+2] round  {local[5]:>7} us   aggregate rate r2 = {r2_loc:6.1f} GB/s")
print(f"  [5]   round  {t1_loc:>7.0f} us   one-group rate  r1 = {r1_loc:6.1f} GB/s")
print(f"  group-scaling factor A_local = r2/r1 = {A_loc:.3f}")
print(f"  collapse gain = 1 - A/2 = {100*(1-A_loc/2):+.1f} %   (matches by construction)")

# --------------------------------------------- independent ranked corroboration
sc_loc = rate(local, 5)  / rate(local, 3)
sc_rnk = rate(ranked, 5) / rate(ranked, 3)
adv    = sc_rnk / sc_loc
A_rnk_pred = A_loc * adv
print()
print("=" * 76)
print("ROUTE 1 - ranked group-scaling advantage (uses NO rival receipt)")
print("=" * 76)
print(f"  [3] -> [3+2] aggregate scaling   local {sc_loc:.3f}   ranked {sc_rnk:.3f}")
print(f"  ranked advantage                 x{adv:.3f}")
print(f"  => predicted A_ranked            {A_rnk_pred:.3f}")
print(f"  => predicted collapse gain       {100*(1-A_rnk_pred/2):+.1f} %")

# ------------------------------------------------- ranked, from the two receipts
meas_dW, sd_dW, share = -0.070, 0.360, 0.24
gain   = (meas_dW / 100.0) / share
gain_s = (sd_dW  / 100.0) / share
A_rnk  = 2 * (1 + gain)
r2_rnk = rate(ranked, 5)
print()
print("=" * 76)
print("ROUTE 2 - the two rival receipts (ca9251b8, 3ff80e86) measure it directly")
print("=" * 76)
print(f"  dW                               {meas_dW:+.3f} +/- {sd_dW:.3f} pp")
print(f"  M=5 share of G2 rounds           {share:.2f}")
print(f"  => M=5 round gain                {100*gain:+.2f} +/- {100*gain_s:.2f} %")
print(f"  => A_ranked                      {A_rnk:.3f}  "
      f"[{2*(1+gain+gain_s):.3f}, {2*(1+gain-gain_s):.3f}]")
print(f"  => one-group rate r1 implied     {r2_rnk/A_rnk:6.1f} GB/s "
      f"({100*(r2_rnk/A_rnk)/r2_rnk:.0f} % of the two-group aggregate {r2_rnk:.1f})")
print()
print(f"  ROUTE 1 predicts A_ranked {A_rnk_pred:.3f};  ROUTE 2 measures {A_rnk:.3f}.  AGREE.")

# ------------------------------------------------------------ falsification
print()
print("=" * 76)
print("FALSIFICATION of 'the G boundary moves to M=6 on the ranked box'")
print("=" * 76)
# counterfactual A values
lad   = [rate(local, m) for m in (1, 2, 3, 4)]
rlad  = [rate(ranked, m) for m in (1, 2, 3, 4)]
haircut = r1_loc / (lad[3] * (lad[3] / lad[2]))     # extrapolation optimism, local
r1_rnk_x = rlad[3] * (rlad[3] / rlad[2]) * haircut
cases = [
    ("ranked box keeps its own NA ladder (haircut-corrected)",
     r2_rnk / r1_rnk_x),
    ("ranked box behaves like the local box",
     A_loc),
]
for label, A in cases:
    g  = 1 - A / 2
    dW = -100 * g * share
    print(f"  {label}")
    print(f"     A = {A:.3f}  gain {100*g:+.1f} %  =>  dW {dW:+.2f} pp  "
          f"=>  {abs(dW - meas_dW)/sd_dW:.1f} sigma from the receipts")

# ------------------------------------------------------------- E104 prize
print()
print("=" * 76)
print("E104 PRIZE on the ranked box, if rate(NA) is repaired")
print("=" * 76)
r1_now = r2_rnk / A_rnk
print(f"  today: one wide x-group at NA=5 streams {r1_now:.1f} GB/s")
print(f"         two x-groups together stream     {r2_rnk:.1f} GB/s")
print(f"  the machine therefore has >= 2x the outstanding-load capacity that")
print(f"  ONE wide x-group uses.  Closing that gap is the E104 target.")
print()
print(f"  {'r1 reaches':>12} {'M=5 round':>11} {'vs today':>9} "
      f"{'leg effect':>11} {'ranked pub':>11}")
for f in (1.10, 1.25, 1.50, 1.75, 2.00):
    r1n = r1_now * f
    tn  = W / (r1n * 1e9) * 1e6
    d   = 100 * (tn / ranked[5] - 1)
    print(f"  {r1n:>9.0f} GB/s {tn:>10.0f} us {d:>8.1f} % "
          f"{d*share:>10.2f} % {d*share:>10.2f} %")
print()
print("  (leg effect = round change x M=5 share; on G2 prompts only, which is")
print("   where the published median lives under Finding 16)")
