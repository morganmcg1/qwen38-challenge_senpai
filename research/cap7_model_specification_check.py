#!/usr/bin/env python3
"""Advisor check on PR #2's claim that occupancy_model.py's ~3.1% cap-7
residual is evidence of a missing per-width weight-stream cost term.

Claim under test (report, "Why this is evidence and not circularity"):
    "The model's cost input C(d) is a smooth, depth-proportional table with a
     linear fill; it has no term for how many weight streams a given row width
     requires. It therefore *cannot* represent a cost cliff between width 8 and
     width 9."

Two things are checked, both from committed artifacts only:

  1. Is MEASURED_C_LOCAL_MS actually smooth at the 7->8 step (= width 8 -> 9)?
  2. If the model's OWN cost table is applied to the actual cap-7 schedule
     instead of the hardcoded DEEP_CAP=8 schedule, how much of the measured
     cap effect comes back?

Dependency-free. All inputs are quoted from the PR #2 report / repo.
"""

# ---- inputs, all quoted from the PR #2 tree -------------------------------
# research/occupancy_model.py:30-33
C = {2: 79.70, 4: 126.40, 5: 146.50, 6: 168.30, 7: 189.70, 8: 217.40}
LIN_B, LIN_M = 12.2, 22.5
NON_BLOCK_MS = 4012.5          # occupancy_model.py:45, per 512-token window
TOKENS = 512


def cost(d):
    return C[d] if d in C else LIN_B + LIN_M * (d + 1)


# Run O / P512 schedule (report: depth histogram, identical across J/O/P512)
DEPTH_HIST = {3: 1, 4: 29, 5: 2, 6: 3, 7: 46}
ROUNDS = sum(DEPTH_HIST.values())
MEASURED_CAP7_S_PER_TOK = 0.03400226   # Run O
MEASURED_CAP8_S_PER_TOK = 0.03510386   # Run I control

print("=" * 74)
print("1. IS THE COST TABLE SMOOTH AT THE WIDTH-9 BOUNDARY?")
print("=" * 74)
ds = sorted(C)
steps = []
for a, b in zip(ds, ds[1:]):
    if b - a == 1:
        step = C[b] - C[a]
        steps.append((a, b, step))
        print(f"  C({b}) - C({a}) = {C[b]:7.2f} - {C[a]:7.2f} = {step:6.2f} ms"
              f"   (width {a+1} -> {b+1})")
lower = [s for (a, b, s) in steps if b < 8]
mean_lower = sum(lower) / len(lower)
top = [s for (a, b, s) in steps if b == 8][0]
print()
print(f"  mean step below the boundary (widths 5..8) = {mean_lower:6.2f} ms")
print(f"  step INTO width 9                          = {top:6.2f} ms")
print(f"  excess at the width-9 boundary             = {top - mean_lower:6.2f} ms"
      f"  ({100*(top-mean_lower)/mean_lower:+.1f}%)")
print(f"  linear-fill slope used for unmeasured d    = {LIN_M:6.2f} ms")
print()
print("  => the table is NOT smooth at the boundary. The 7->8 step is the")
print("     largest in the table and carries a ~%.1f ms width-9 excess." %
      (top - mean_lower))
print("     Cross-check: PR #1's depth curve put the width-9 boundary excess")
print("     at 7.2 ms. These are two independent estimates of the same cliff.")

print()
print("=" * 74)
print("2. APPLY THE MODEL'S OWN COST TABLE TO THE REAL CAP-7 SCHEDULE")
print("=" * 74)
block_cap7 = sum(cost(d) * n for d, n in DEPTH_HIST.items())
per_tok_overhead = NON_BLOCK_MS / TOKENS
pred_cap7 = block_cap7 / TOKENS + per_tok_overhead
print(f"  rounds {ROUNDS}, tokens {TOKENS}  (81 base + 431 accepted drafts)")
print(f"  block time from C(d)        = {block_cap7:9.1f} ms")
print(f"  + non-block overhead        = {NON_BLOCK_MS:9.1f} ms")
print(f"  predicted ms/token          = {pred_cap7:9.4f}")
print(f"  measured  ms/token (Run O)  = {1000*MEASURED_CAP7_S_PER_TOK:9.4f}")
err = 100 * (pred_cap7 / (1000 * MEASURED_CAP7_S_PER_TOK) - 1)
print(f"  error                       = {err:+9.2f} %   <- table is well "
      f"calibrated ON the cap-7 schedule")

print()
n_sat = DEPTH_HIST[7]
step78 = cost(8) - cost(7)
print("  Counterfactual: same acceptance, cap 8. The %d rounds that saturate"
      % n_sat)
print("  at depth 7 attempt depth 8 instead (+%.1f ms each); E of them gain a"
      % step78)
print("  further accepted token. Sweep E over its whole feasible range:")
print()
print("     E   extra tok   pred cap-8 ms/tok   vs cap-7   (measured +3.24%)")
n_sat = DEPTH_HIST[7]
extra_cost = n_sat * (cost(8) - cost(7))
for E in [0, 10, 20, 25, 30, 35, 40, 46]:
    block_cap8 = block_cap7 + extra_cost
    toks = TOKENS + E
    pred_cap8 = block_cap8 / toks + per_tok_overhead
    d = 100 * (pred_cap8 / pred_cap7 - 1)
    print(f"  {E:4d}   {E:8d}   {pred_cap8:15.4f}   {d:+8.2f} %")

meas_delta = 100 * (MEASURED_CAP8_S_PER_TOK / MEASURED_CAP7_S_PER_TOK - 1)
print()
print(f"  measured cap-8 vs cap-7 = {meas_delta:+.2f} %")
print()
print("  => Over the ENTIRE feasible range of E the sign is correct and the")
print("     magnitude is the right order. The model's own cost table already")
print("     prices the cap effect. What it does not have is the cap-7")
print("     SCHEDULE: DEEP_CAP is a module constant fixed at 8, so all three")
print("     cap-7 arms were scored against a cap-8 Markov chain.")
print()
print("  => The ~3.1% residual is a SPECIFICATION error, not a missing-term")
print("     error. Free falsification test: set DEEP_CAP = 7 and re-score")
print("     J / O / P512. No GPU time, no rebuild.")
