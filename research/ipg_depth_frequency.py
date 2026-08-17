#!/usr/bin/env python3
"""E14 desk arithmetic: what a cheaper M=5 verify buys.

Two questions that need no new hardware once the kernel arms are measured:

1. Does a cheaper M=5 verify re-open depth 4? The shipped policy walk tests
   `reach > h[d]*(1+expected)/(1+cumH)`. At the 3 -> 4 test the best possible
   `reach` is 1, so depth 4 is reachable only if that threshold drops below 1.
   With PR #1's measured curve it is 1.0693, i.e. structurally unreachable.
   Only h[3] moves when the M=5 kernel gets cheaper, so the break-even h[3] is
   a closed form.

2. What is it worth end to end? Depth-4 rounds are the only rounds that issue
   an M=5 verify, so a per-round saving applies at exactly the depth-4
   frequency, weighted by round cost.

    python3 research/ipg_depth_frequency.py --delta-m5-us 120
"""
import argparse
import json

# PR #1 (Edward), research/results/qwen38-r1-e1-depth-cost-curve.md.
H_MEASURED = [0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909]
C_DECL_US = {0: 65009.4, 1: 70482.4, 2: 75519.2, 3: 91287.8, 4: 115690.9,
             5: 134668.0, 6: 154169.1, 7: 172827.0, 8: 198236.5}
# PR #13 (Edward) control histogram, offered depth -> rounds. Verify width is
# M = d + 1, so depth 4 is the only depth that reaches M = 5.
CONTROL_HIST = {1: 39, 2: 129, 3: 67, 4: 18}
# PR #13's best Hp arm. It never offers depth 4, hence never issues M=5.
EDWARD_HP_HIST = {1: 2, 2: 231, 3: 13}


def threshold_3to4(h, reach=1.0, expected=3.0):
    cum = h[0] + h[1] + h[2]
    return h[3] * (1.0 + expected) / (1.0 + cum)


def breakeven_h3(h):
    """Largest h[3] for which the 3 -> 4 test can pass at all (reach = 1)."""
    cum = h[0] + h[1] + h[2]
    return (1.0 + cum) / 4.0


def reachable_q(h):
    """Smallest constant per-position acceptance q that passes the 3 -> 4 test.

    reach = q^4, expected = q + q^2 + q^3."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        q = 0.5 * (lo + hi)
        thr = h[3] * (1.0 + q + q * q + q ** 3) / (1.0 + h[0] + h[1] + h[2])
        if q ** 4 > thr:
            hi = q
        else:
            lo = q
    return hi if hi < 1.0 else None


def round_share(hist, curve):
    total = sum(curve[d] * k for d, k in hist.items())
    return {d: curve[d] * k / total for d, k in hist.items()}, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-m5-us", type=float, default=None,
                    help="measured per-round microseconds removed from an M=5 "
                         "verify by the arm (positive = faster)")
    ap.add_argument("--h3-measured", type=float, default=None,
                    help="measured h[3] under the arm, in units of C(0)")
    ap.add_argument("--json-out")
    ap.add_argument("--measured-hist",
                    help="depth_histogram.py JSON; adds a third, locally measured "
                         "policy row instead of relying on the copied constants")
    args = ap.parse_args()

    rep = {}
    cum3 = H_MEASURED[0] + H_MEASURED[1] + H_MEASURED[2]
    base_thr = threshold_3to4(H_MEASURED)
    be = breakeven_h3(H_MEASURED)
    rep["baseline"] = {"h": H_MEASURED[:4], "cum_h0_h2": cum3,
                       "threshold_3to4_at_q1": base_thr,
                       "breakeven_h3": be,
                       "required_h3_drop_pct": 100.0 * (H_MEASURED[3] - be)
                       / H_MEASURED[3]}
    print("=" * 74)
    print("1. depth-4 reachability")
    print("=" * 74)
    print(f"  cumH(h0..h2)                  {cum3:.4f}")
    print(f"  shipped h3                    {H_MEASURED[3]:.4f}")
    print(f"  3->4 threshold at q=1         {base_thr:.4f}"
          f"   (reach <= 1, so {'UNREACHABLE' if base_thr >= 1 else 'reachable'})")
    print(f"  break-even h3                 {be:.4f}")
    print(f"  required h3 reduction         {rep['baseline']['required_h3_drop_pct']:.2f}%")
    print(f"  = per-round microseconds      {(H_MEASURED[3]-be)*C_DECL_US[0]:.1f} us"
          f"  of the {(H_MEASURED[3])*C_DECL_US[0]:.1f} us 4th draft step")

    h_arm = None
    if args.h3_measured is not None:
        h_arm = list(H_MEASURED)
        h_arm[3] = args.h3_measured
    elif args.delta_m5_us is not None:
        h_arm = list(H_MEASURED)
        h_arm[3] = H_MEASURED[3] - args.delta_m5_us / C_DECL_US[0]
    if h_arm:
        thr = threshold_3to4(h_arm)
        q = reachable_q(h_arm)
        rep["arm"] = {"h3": h_arm[3], "threshold_3to4_at_q1": thr,
                      "depth4_reachable": thr < 1.0,
                      "min_constant_q": q}
        print(f"\n  arm h3                        {h_arm[3]:.4f}")
        print(f"  arm 3->4 threshold at q=1     {thr:.4f}"
              f"   ({'REACHABLE' if thr < 1 else 'still unreachable'})")
        print(f"  min constant acceptance q     "
              + (f"{q:.4f}" if q else "unreachable at any q"))

    print("\n" + "=" * 74)
    print("2. scored-fixture frequency of M=5")
    print("=" * 74)
    policies = [("PR #13 control policy", CONTROL_HIST),
                ("PR #13 best Hp arm", EDWARD_HP_HIST)]
    if args.measured_hist:
        with open(args.measured_hist) as fh:
            measured = json.load(fh)
        for arm, entry in measured.items():
            policies.append((f"measured {arm}",
                             {int(d): c for d, c in entry["hist"].items()}))
    for name, hist in policies:
        n = sum(hist.values())
        d4 = hist.get(4, 0)
        share, total_us = round_share(hist, C_DECL_US)
        rep.setdefault("frequency", {})[name] = {
            "hist": hist, "rounds": n, "depth4_rounds": d4,
            "round_fraction": d4 / n, "time_fraction": share.get(4, 0.0)}
        print(f"  {name:24s} {hist}")
        print(f"    rounds {n:4d}   depth-4 rounds {d4:3d}"
              f"   round fraction {100.0*d4/n:5.2f}%"
              f"   round-cost-weighted time fraction {100.0*share.get(4,0.0):5.2f}%")
        if args.delta_m5_us is not None and d4:
            saved = d4 * args.delta_m5_us
            print(f"    at -{args.delta_m5_us:.1f} us per M=5 verify:"
                  f" end-to-end {-100.0*saved/total_us:+.3f}%")
            rep["frequency"][name]["end_to_end_pct"] = -100.0 * saved / total_us

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rep, fh, indent=2)


if __name__ == "__main__":
    main()
