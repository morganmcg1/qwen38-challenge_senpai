"""What the 623e77af receipt says about the three levers on the QMV kernel.

The one-pass arm moved three quantities at once on widths 6 and 7:

  issue count    down, because row-keyed statements cost 1/ipg per output
  occupancy      down, because a wider body needs more registers
  weight passes  down, because ceil(m / ipg) is the number of reads of the
                 whole weight matrix (Qwen35.swift:1815-1818, read this session)

Two of those should make the kernel faster and one slower. The receipt measured
the net. This solves for how much each lever can be worth, and reports the
shortfall of every model that was used to price this class of work.

harness=ranked for the measured effect, compile-only for the lever sizes.
"""

import json
import math

CENSUS = "research/e133-artifacts/f23-composed-census.json"
ARCH = "applegpu_g17s"
QMV_SHARE = 0.8735          # F139

# Route change made by the one-pass arm, width -> (shipped ipg, onepass ipg).
ROUTE = {6: (3, 6), 7: (4, 7)}

# Effect on rounds that actually run at width 6 or 7, inverted from the
# per-prompt dose-response in F24 section 3, as a fraction of round time.
OBSERVED_ROUND = -0.005

# Models that were used to price this class of work, as a fraction of round.
MODELS = {
    "F151 issue model": -0.1096,
    "row-keyed issue, ipg scaling": None,     # filled in below
    "weight-traffic model, passes halve": None,
}


def main():
    d = json.load(open(CENSUS))
    cell = {c["na"]: c for c in d["variants"]["shipped"]["cells"].values()}

    print("harness=ranked (effect) + compile-only (levers)")
    print(f"QMV share of round: {QMV_SHARE} (F139)")
    print()
    print(f"{'M':>3}{'route':>12}{'passes':>9}{'air/out':>18}"
          f"{'issue d%':>10}{'occ':>10}{'occ d%':>9}")
    iss, occ, pas = [], [], []
    for M, (a, b) in ROUTE.items():
        pa, pb = math.ceil(M / a), math.ceil(M / b)
        ia = cell[a]["air_per_output_element"]
        ib = cell[b]["air_per_output_element"]
        oa = cell[a][ARCH]["simdgroups_derived"]
        ob = cell[b][ARCH]["simdgroups_derived"]
        di = (ib / ia - 1) * 100
        do = (ob / oa - 1) * 100
        iss.append(ib / ia)
        occ.append(ob / oa)
        pas.append(pb / pa)
        print(f"{M:>3}{f'ipg{a}->ipg{b}':>12}{f'{pa}->{pb}':>9}"
              f"{f'{ia:.4f}->{ib:.4f}':>18}{di:>10.1f}{f'{oa}->{ob}':>10}"
              f"{do:>9.1f}")

    mi = sum(iss) / len(iss)
    mo = sum(occ) / len(occ)
    mp = sum(pas) / len(pas)
    print()
    print(f"mean issue ratio {mi:.4f}  ({(mi - 1) * 100:+.1f} %)")
    print(f"mean occupancy ratio {mo:.4f}  ({(mo - 1) * 100:+.1f} %)")
    print(f"mean weight-pass ratio {mp:.4f}  ({(mp - 1) * 100:+.1f} %)")
    print()

    MODELS["row-keyed issue, ipg scaling"] = (mi - 1) * QMV_SHARE
    MODELS["weight-traffic model, passes halve"] = (mp - 1) * QMV_SHARE

    print("--- what each model predicted for a round at width 6 or 7 ---")
    print(f"{'model':<38}{'predicted':>12}{'observed':>11}{'shortfall':>11}")
    for name, pred in MODELS.items():
        sf = pred / OBSERVED_ROUND if OBSERVED_ROUND else float("inf")
        print(f"{name:<38}{pred * 100:>11.2f} %{OBSERVED_ROUND * 100:>10.2f} %"
              f"{sf:>10.1f}x")
    print()

    print("--- solving the two-channel blend T ~ issue^a * occupancy^-b ---")
    obs = math.log(1 + OBSERVED_ROUND / QMV_SHARE)
    rows = []
    for M, (a, b) in ROUTE.items():
        ia = cell[a]["air_per_output_element"]
        ib = cell[b]["air_per_output_element"]
        oa = cell[a][ARCH]["simdgroups_derived"]
        ob = cell[b][ARCH]["simdgroups_derived"]
        rows.append((math.log(ib / ia), math.log(oa / ob)))
    (p, q), (r, s) = rows
    det = p * s - q * r
    ea = (obs * s - q * obs) / det
    eb = (p * obs - obs * r) / det
    print(f"  exponent on issue      a = {ea:+.4f}")
    print(f"  exponent on occupancy  b = {eb:+.4f}")
    print(f"  a + b = {ea + eb:+.4f}")
    print()
    print("  A kernel bound by issue would give a near 1. One bound by")
    print("  occupancy would give b near 1. Both land near zero, so at these")
    print("  widths the kernel time is nearly invariant to both levers.")
    print()
    launch_accounting()


# Qwen35.swift:1959-1961, submitted blob:
#   columns = grid == .tight ? (m + ipg - 1) / ipg : m
# and Qwen35.swift:1546-1550 makes every threadgroup with first_m >= M return
# before it loads any weight. So under `wide` the launched count is m and is
# INDEPENDENT of ipg, while the working count is ceil(m / ipg).
SHIPPED_PLAN = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
ONEPASS_PLAN = {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 4, 9: 3}
F83 = {6: 0.188, 7: 0.211, 8: 0.1871}


def launch_accounting():
    print("--- launched vs working threadgroup columns ---")
    print(f"{'M':>3}{'ipg_ship':>10}{'ipg_1pass':>11}{'wide':>7}"
          f"{'work_ship':>11}{'work_1pass':>12}{'empty_1pass':>13}")
    for M in sorted(ONEPASS_PLAN):
        ws = math.ceil(M / SHIPPED_PLAN[M])
        wo = math.ceil(M / ONEPASS_PLAN[M])
        print(f"{M:>3}{SHIPPED_PLAN[M]:>10}{ONEPASS_PLAN[M]:>11}{M:>7}"
              f"{ws:>11}{wo:>12}{f'{M - wo} of {M}':>13}")

    mass = sum(F83.values())
    wide = sum(F83[M] * M for M in F83) / mass
    tight = sum(F83[M] * math.ceil(M / ONEPASS_PLAN[M]) for M in F83) / mass
    print()
    print(f"F83-weighted launched columns, grid=wide   {wide:.4f}")
    print(f"F83-weighted launched columns, grid=tight  {tight:.4f}")
    print(f"tight would launch {wide / tight:.2f}x fewer threadgroups "
          f"for identical work and identical output")
    print()
    print("Under `wide` the launched count does not depend on ipg at all, so no")
    print("table change can move it. That is exactly the invariance the receipt")
    print("measured. `tight` is an existing env switch, MLX_E120_QMV_GRID, and")
    print("the ledger records no test of it.")


if __name__ == "__main__":
    main()
