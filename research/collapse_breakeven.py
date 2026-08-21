"""Break-even one-group rate for a stream collapse at each verify width.

Finding 32:  collapse gain = 1 - A/2,   A = r2 / r1
             A_ranked = A_local * 1.244   (ranked group-scaling advantage)
Ranked-neutral  <=>  A_ranked = 2  <=>  A_local = 2 / 1.244 = 1.6077
             <=>  r1_local = r2_local / 1.6077
"""

W_GB = 14.4123
ADV = 1.244                      # ranked group-scaling advantage, Finding 32 route 1
A_NEUTRAL = 2.0 / ADV

# local aggregate rates, Finding 31
r2_local = {5: 228.6, 6: 209.1, 7: 191.6, 8: 175.8}   # two concurrent groups
r1_local_known = {1: 223.6, 2: 206.6, 3: 192.7, 4: 167.1, 5: 139.4}  # one group, NA

# ranked round cost, Finding 12
ranked_round = {3: 39167, 4: 43162, 5: 53105, 6: 60341, 7: 67574, 8: 74807, 9: 82040}
share = {3: .0325, 4: .142, 5: .241, 6: .334, 7: .122, 8: .0735, 9: .0575}

print(f"A_local ranked-neutral threshold = 2 / {ADV} = {A_NEUTRAL:.4f}\n")
print("break-even ONE-GROUP local rate, by verify width")
print(f"{'M':>3} {'r2 local':>9} {'r1 break-even':>14} {'r1 measured':>12} {'shortfall':>10}")
for M in sorted(r2_local):
    be = r2_local[M] / A_NEUTRAL
    got = r1_local_known.get(M)
    g = f"{got:.1f}" if got else "NOT MEASURED"
    sf = f"{(be/got - 1)*100:+.1f} %" if got else "-"
    print(f"{M:>3} {r2_local[M]:>9.1f} {be:>14.1f} {g:>12} {sf:>10}")

tw = {M: share[M] * ranked_round[M] for M in share}
tot = sum(tw.values())
print(f"\ntime-weighted round share:  M=5 {tw[5]/tot:6.2%}   M=6 {tw[6]/tot:6.2%}"
      f"   M>=6 {(tw[6]+tw[7]+tw[8]+tw[9])/tot:6.2%}")

print("\ndirect prize on the already-collapsed M=5 leg (ranked)")
r1_ranked_5 = 272.2
base5 = ranked_round[5]
for L in (1.00, 1.10, 1.25, 1.50, 1.75, 2.00):
    t = W_GB / (r1_ranked_5 * L) * 1e6
    print(f"  rate x{L:.2f}  M=5 round {t:8.0f} us  ({(t/base5-1)*100:+6.2f} % of that round)"
          f"   leg effect {(t-base5)/base5*tw[5]/tot*100:+6.2f} %")
