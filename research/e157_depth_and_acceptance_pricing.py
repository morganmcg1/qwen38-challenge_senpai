"""FINDING 286/287/288 — the ranked draft schedule, the marginal-row rule, and
the repriced value of one acceptance point.

Inputs are the eight per-prompt ranked quantities from receipt `0cf1637e`
(round_count, leg_seconds, effective_mean_draft_len, derived clean_round_us)
and edward's E154 local row law, W&B run `fi9mm878`:

    clean_round_us = 23,421 + 15,708 * verified_rows       harness=local
    an accepted token makes the round 262.9 +- 29.4 us faster at fixed depth

Everything below is derived; nothing is fitted to the ranked set.

Run:  python3 research/e157_depth_and_acceptance_pricing.py
"""
import statistics

# name, round_count, leg_seconds, effective_mean_draft_len, clean_round_us
RANKED = [("plutarch", 486.76, 15.49210, 0.1557, 30744.6),
          ("drama",    252.01,  9.13101, 2.2976, 34140.1),
          ("travel",   212.33,  7.99539, 2.6479, 35176.7),
          ("beagle",   110.00,  5.47533, 4.3818, 44990.8),
          ("republic",  93.00,  4.97203, 4.9892, 47808.7),
          ("essays",    92.04,  5.03450, 5.0870, 48974.9),
          ("medicine",  90.01,  4.97664, 5.2556, 49448.1),
          ("botany",    81.04,  4.94848, 6.1481, 54561.1)]

SCALE = 148775.0 / 53576.0          # M4 Pro round / ranked round at t = 6.5641
A = 23421.0 / SCALE                 # ranked fixed term, us
B = 15708.0 / SCALE                 # ranked price per verified row, us
ROLL = 262.9 / SCALE                # ranked rollback price per rejected draft
W = {"beagle": 0.5000, "essays": 0.4474, "republic": 0.0329, "medicine": 0.0197}
OURS, GAP = 3.68278758, 0.04632


def tokens(p, d):
    t, pw = 1.0, 1.0
    for _ in range(d):
        pw *= p
        t += pw
    return t


def dtokens_dp(p, d):
    return sum(i * p ** (i - 1) for i in range(1, d + 1))


def calibrate(t_obs, d):
    """Per-position acceptance p that reproduces the observed tokens/round."""
    lo, hi = 1e-6, 0.999999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tokens(mid, d) < t_obs else (lo, mid)
    return 0.5 * (lo + hi)


def seed_cost():
    """Residual leg time not explained by rounds. Prompt-independent."""
    v = [(leg * 1e6 / r - cl) * r / 1e6 for _, r, leg, _, cl in RANKED]
    return statistics.mean(v), statistics.stdev(v)


PRE, PRE_SD = seed_cost()


def leg_seconds(p, d):
    t = tokens(p, d)
    cost = A + B * (d + 1) + ROLL * (d - (t - 1))
    return PRE + (512.0 / t) * cost / 1e6


def breakeven(p, d):
    """Marginal row d+1 pays only if p**(d+1) exceeds this. Host-invariant:
    A, B and ROLL all carry the same host factor, so it cancels in the ratio."""
    return tokens(p, d) * (B + ROLL) / (A + B * (d + 1)
                                        + ROLL * (d - (tokens(p, d) - 1)))


def recover():
    out = []
    for name, r, leg, edl, cl in RANKED:
        d = round((cl - A) / B - 1.0)
        out.append((name, calibrate(512.0 / r, d), d, (cl - A) / B - 1.0, edl))
    return out


def weighted(per_prompt):
    return (sum(W[k] * per_prompt[k] for k in W)) / sum(W.values())


def main():
    print(f"ranked row law   clean = {A:,.0f} + {B:,.0f} * rows"
          f"   rollback {ROLL:.1f} us")
    print(f"seed cost per leg {PRE:.4f} s   sd {PRE_SD:.5f} s"
          f"   (prompt independent)\n")

    rec = recover()
    print("FINDING 286 - recovered schedule; d - edl must be 1 by construction")
    resid = [dr - edl for n, _, _, dr, edl in rec if n != "plutarch"]
    for name, p, d, draw, edl in rec:
        print(f"  {name:10s} p {p:.4f}  d {d}  d_raw {draw:6.3f}"
              f"  edl {edl:6.4f}  d-edl {draw - edl:+6.3f}")
    print(f"  seven all-drafting: mean {statistics.mean(resid):+.4f}"
          f"  sd {statistics.stdev(resid):.4f}\n")

    print("FINDING 287 - the marginal drafted row does not pay")
    gains = {}
    for name, p, d, _, _ in rec:
        if name == "plutarch":
            continue
        base = leg_seconds(p, d)
        best, best_d = min((leg_seconds(p, dd), dd) for dd in range(9))
        gains[name] = 100.0 * (base / best - 1.0)
        print(f"  {name:10s} offered {p ** (d + 1):.4f}"
              f"  breakeven {breakeven(p, d):.4f}"
              f"  d {d} -> d* {best_d}  raw gain {gains[name]:+7.4f} %")
    depth_only = weighted(gains)
    print(f"  Rule 148 weighted depth recalibration {depth_only:+.4f} %\n")

    print("FINDING 288 - a +3 point head makes the live schedule correct")
    both, recall_only = {}, {}
    for name, p, d, _, _ in rec:
        if name not in W:
            continue
        p2 = min(p + 0.03, 0.999)
        base = leg_seconds(p, d)
        best2, d2 = min((leg_seconds(p2, dd), dd) for dd in range(9))
        both[name] = 100.0 * (base / best2 - 1.0)
        recall_only[name] = 100.0 * (base / leg_seconds(p2, d) - 1.0)
        _, d_now = min((leg_seconds(p, dd), dd) for dd in range(9))
        print(f"  {name:10s} live d {d}   d* at p {d_now}"
              f"   d* at p+3pp {d2}")
    wb, wr = weighted(both), weighted(recall_only)
    print(f"  depth only {depth_only:+.4f} %   recall only {wr:+.4f} %"
          f"   both {wb:+.4f} %")
    print(f"  sub-additive by {wb - depth_only - wr:+.4f} pp,"
          f" almost exactly the depth fix\n")

    print("ADVISOR ERROR 193 corrected - one acceptance point")
    pts = {}
    for name, p, d, _, _ in rec:
        if name not in W:
            continue
        t = tokens(p, d)
        pts[name] = 0.01 * dtokens_dp(p, d) / t
        print(f"  {name:10s} dE/dp {dtokens_dp(p, d):7.3f}"
              f"  1/t {100 / t:5.2f} %/tok"
              f"  per point {100 * pts[name]:+.4f} %"
              f"  = {OURS * pts[name]:+.4f} absolute")
    wp = weighted(pts)
    print(f"  Rule 148 weighted {100 * wp:+.4f} % of raw"
          f" = {OURS * wp:+.4f} absolute = {OURS * wp / GAP:.2f}x the gap")
    print(f"  acceptance points needed to close the gap: {GAP / (OURS * wp):.2f}")


if __name__ == "__main__":
    main()
