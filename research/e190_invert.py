#!/usr/bin/env python3
"""E190: invert the paid cap-5 receipt into a direct price for the 6-row cell.

    usage: python3 research/e190_invert.py [--receipt /tmp/e190_receipt.json]

harness=ranked.

WHY THIS INVERSION IS CLEAN. `e190_predict.chain` walks the paid cap-4 anchor
C up to cap 5 with a single term:

    R += t[4] * (R(6) - R(5))

No other cost-model value enters. Walking down from the cap-7 anchor A instead
would drag in R(7) and R(8), which this receipt cannot separate. So the
C-anchored walk turns the published cap-5 median into a measurement of exactly
one quantity, the 6-row increment

    dR6 = R(6) - R(5),

conditional on the E177 survival model and the accepted-length slope mu, which
are both fitted on the A and C receipts alone and are unchanged here.

The published median is strictly decreasing in dR6, so a bisection is exact to
machine precision.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e177_ranked_depth_law as e177  # noqa: E402
import e190_predict as e190  # noqa: E402

MUE_MS = e190.MUE_MS
SIGMA = e190.SIGMA_PUBLISHED


def score_with_dr6(data, tvar, mu, dr6_s):
    """Published cap-5 median implied by a 6-row increment, in seconds."""
    raws = []
    for name in e177.ORDER:
        rec = data["C"]["rec"][name]
        t = tvar[name]
        abar = rec["abar"] + t[4] * mu[name]
        r_round = rec["R"] + t[4] * dr6_s
        n_rounds = e177.TOKENS / (1.0 + abar)
        total = n_rounds * r_round
        vec = data["A"]["vec"][name]
        raws.append(vec["serial"] / (total / e177.TOKENS + vec["prefill"]))
    return e177.published_median(raws)


def solve_dr6(data, tvar, mu, target, lo=-0.050, hi=0.200):
    """Bisect the strictly decreasing score(dr6) for the observed median."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if score_with_dr6(data, tvar, mu, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", default="/tmp/e190_receipt.json")
    args = parser.parse_args()

    receipt = json.loads(pathlib.Path(args.receipt).read_text())
    observed = receipt["officialScore"]

    data, surv, bounds, mu, fits, core = e190.build()
    quad = fits["quadratic"]
    stairs, shape = e190.staircase_variants(quad)
    stairs["staircase-ranked-refit"] = e190.refit_shape_on_ranked(core, shape)
    tvar = {n: list(surv[n]) for n in e177.ORDER}

    dr6 = solve_dr6(data, tvar, mu, observed)
    check = score_with_dr6(data, tvar, mu, dr6)

    # 1-sigma receipt-channel band on the observed median -> band on dR6.
    dr6_lo = solve_dr6(data, tvar, mu, observed * (1 + SIGMA))
    dr6_hi = solve_dr6(data, tvar, mu, observed * (1 - SIGMA))

    print("=" * 100)
    print("E190 INVERSION: the paid cap-5 receipt prices the 6-row cell "
          "(harness=ranked)")
    print("=" * 100)
    print("\n  observed published median   %.14f" % observed)
    print("  bisection closes at         %.14f" % check)

    print("\n  MEASURED 6-row increment, dR6 = R(6) - R(5)")
    print("    dR6 = %+7.3f ms   1-sigma band [%+7.3f, %+7.3f] ms"
          % (1000 * dr6, 1000 * dr6_lo, 1000 * dr6_hi))
    print("    that is %+6.2f MUE, band [%+6.2f, %+6.2f] MUE"
          % (1000 * dr6 / MUE_MS, 1000 * dr6_lo / MUE_MS, 1000 * dr6_hi / MUE_MS))

    print("\n  PREDICTED dR6 under each law, and the error each one makes")
    rows = [("quadratic", quad)] + list(stairs.items())
    print("    %-24s %10s %10s %10s %9s"
          % ("law", "dR6 ms", "err ms", "err MUE", "score"))
    out = {}
    for key, model in rows:
        pred = (e177.cost_at_width(model, 6) - e177.cost_at_width(model, 5))
        pscore = e190.score_of(data, tvar, mu, model, 5, "C")
        out[key] = {"dr6_ms": 1000 * pred, "err_ms": 1000 * (pred - dr6),
                    "err_mue": 1000 * (pred - dr6) / MUE_MS,
                    "c_anchored_score": pscore}
        print("    %-24s %10.3f %10.3f %10.2f %9.6f"
              % (key, 1000 * pred, 1000 * (pred - dr6),
                 1000 * (pred - dr6) / MUE_MS, pscore))

    print("\n  the same laws' R(5) and implied R(6)")
    print("    %-24s %10s %10s %10s" % ("law", "R(5) ms", "R(6) ms", "ratio"))
    for key, model in rows:
        r5 = 1000 * e177.cost_at_width(model, 5)
        r6 = 1000 * e177.cost_at_width(model, 6)
        print("    %-24s %10.3f %10.3f %10.4f" % (key, r5, r6, r6 / r5))
    r5_quad = e177.cost_at_width(quad, 5)
    print("    %-24s %10.3f %10.3f %10.4f"
          % ("MEASURED (quad R(5) base)", 1000 * r5_quad,
             1000 * (r5_quad + dr6), (r5_quad + dr6) / r5_quad))

    print("\n  VERDICT INPUTS")
    print("    paid cap-4 anchor C      %.14f" % e177.CAP4_PUBLISHED)
    print("    observed cap-5           %.14f  (%+.2f sigma vs the cap-4 anchor)"
          % (observed, (observed - e177.CAP4_PUBLISHED) / (SIGMA * observed)))
    print("    paid cap-7 receipt A     %.14f" % e177.BEST_A)
    print("    crown                    %.14f" % e177.CROWN)

    payload = {
        "harness": "ranked",
        "observed_published_median": observed,
        "bisection_closure": check,
        "dr6_seconds": dr6,
        "dr6_ms": 1000 * dr6,
        "dr6_ms_sigma_band": [1000 * dr6_lo, 1000 * dr6_hi],
        "dr6_mue": 1000 * dr6 / MUE_MS,
        "laws": out,
        "cap4_anchor": e177.CAP4_PUBLISHED,
        "cap7_anchor": e177.BEST_A,
        "crown": e177.CROWN,
        "sigma_published": SIGMA,
        "mue_ms": MUE_MS,
    }
    dest = pathlib.Path("research/out/e190")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "inversion.json").write_text(json.dumps(payload, indent=2))
    print("\n  wrote research/out/e190/inversion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
