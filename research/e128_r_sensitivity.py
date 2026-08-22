"""How does the E128 depth gap depend on the UNPINNED ranked round count R?

R is not published. The identity R = 512 - A (every round emits one committed
token plus its accepted drafts) leaves R and the accept rate jointly free:
    accept_rate = (512 - R) / (R * effective_mean_draft_len)

A uniform scale on R only rescales the cost curve, which cannot move the argmin.
The gap therefore depends on R ONLY through the implied acceptance. Sweep it.
"""
import json


def round_us(M):
    if M <= 4:
        return 27215.4 + 3966.4 * M
    return 17020.7 + 7154.2 * M


CAP = 7


def eacc(p, d):
    if d <= 0:
        return 0.0
    if abs(1.0 - p) < 1e-12:
        return float(d)
    return p * (1.0 - p ** d) / (1.0 - p)


def solve_p(d, target):
    lo, hi = 1e-6, 0.999999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eacc(mid, d) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


CASES = [("beagle", 4.3818, 110), ("medicine", 5.2556, 90),
         ("essays", 5.0870, 92), ("republic", 4.9892, 93),
         ("botany", 6.1481, 81)]

print("=" * 104)
print("E128 SENSITIVITY TO THE UNPINNED RANKED ROUND COUNT R")
print("=" * 104)
for name, eff, R_assumed in CASES:
    Rmin = int(512.0 / (1.0 + eff)) + 1          # accept rate <= 1
    print()
    print(f"--- {name}   eff_len={eff:.4f}   assumed R={R_assumed}   "
          f"feasible R from {Rmin}")
    print(f"{'R':>5} {'accept rate':>12} {'per-step p':>11} {'opt d':>6} "
          f"{'gain %':>8}  direction")
    print("-" * 104)
    for R in sorted({Rmin, Rmin + 4, Rmin + 8, R_assumed - 8, R_assumed - 4,
                     R_assumed, R_assumed + 8, R_assumed + 20, R_assumed + 40,
                     R_assumed + 80}):
        if R < Rmin or R > 512:
            continue
        rate = (512.0 - R) / (R * eff)
        if not (0.0 < rate <= 1.0):
            continue
        p = solve_p(eff, rate * eff)
        base = round_us(eff + 1.0) / (1.0 + eacc(p, eff))
        bd, bc = eff, base
        for k in range(0, CAP + 1):
            c = round_us(k + 1.0) / (1.0 + eacc(p, float(k)))
            if c < bc:
                bd, bc = float(k), c
        g = 100.0 * (base - bc) / base
        arrow = "DEEPER" if bd > eff + 1e-9 else (
            "SHALLOWER" if bd < eff - 1e-9 else "no change")
        mark = "  <== assumed" if R == R_assumed else ""
        print(f"{R:5d} {rate:12.4f} {p:11.4f} {bd:6.1f} {g:8.3f}  {arrow}{mark}")

print()
print("=" * 104)
print("PLUTARCH BOUND ON THE CANDIDATE M=1 ROUND (R-robust, uses only published fields)")
print("=" * 104)
board = json.load(open("/tmp/yukon-board/full.json"))
rows = board["submissions"] if isinstance(board, dict) else board
rec = next(r for r in rows if str(r.get("id", "")).startswith("44559d02"))
pl = next(e for e in rec["officialMetrics"]["per_prompt"]
          if e["prompt_sha256"][:8] == "c1ec5866")
tot = 512.0 * pl["mtp_seconds_per_token_mean"] * 1e6
N = pl["non_drafting_round_count"]
s1 = pl["serial_seconds_per_token_mean"] * 1e6
print(f"plutarch candidate leg total   {tot:12,.0f} us")
print(f"non-drafting rounds            {N:12d}   (a hard lower bound on R)")
print(f"baseline serial M=1 round      {s1:12,.1f} us")
print(f"upper bound on candidate M=1   {tot/N:12,.1f} us   ratio {tot/N/s1:.4f}")
print(f"value at R=512 (all rounds)    {tot/512:12,.1f} us   ratio {tot/512/s1:.4f}")
print()
print("So the candidate build's own M=1 round is 9 to 22 % faster than the PINNED")
print("baseline serial round, before any speculation. That share of the ranked score")
print("comes from accumulated candidate-runtime work, not from drafting.")
