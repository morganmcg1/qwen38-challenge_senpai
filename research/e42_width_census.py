#!/usr/bin/env python3
"""E42 side-product: the M in {5,9} share of beagle/medicine DECODE TIME.

The advisor's question is a value input for "E27 done right", whose worth is
`share(M in {5,9}) x saving`. A round-count share is not a substitute because
the widths cost different amounts. This script supplies the missing half of
that conversion -- cost weights C(M) measured on this host -- and is explicit
about which half is measured and which is only partially identified.

  cost weights C(M)    LOCAL M4 Pro measurement, E42's own base curve
  round shares rho(M)  RANKED telemetry, partially identified

Published ranked fields that constrain rho, and whether they are used:

  effective_mean_draft_len  USED   sum (M-1) rho_M = n
  non_drafting_round_count  USED   nd = 0 pins rho_1 = 0 exactly
  round_count R (107/99)    USED   only for the integrality remark; the bracket
                                   itself does not depend on R because nd = 0
                                   already pins rho_1 = 0 for any R
  mtp_seconds_per_token     not    absolute ranked leg time; using it would
                                   import local absolute cost onto a ranked host
  serial_seconds_per_token  not    depth-0 leg, carries no width information
  raw_ratio_of_means        not    a ratio of the two legs above
  accepted_pair_count       not    one accepted pair per prompt in this run
  integrality (R rho_M in Z) not   would TIGHTEN; the LP brackets below are
                                   therefore conservative in the safe direction

MAIN NEGATIVE RESULT: those constraints leave the ABSOLUTE share of M in {5,9}
vacuous at [0, 1] on both prompts. A two-point mixture on {2,6} hits the
published mean with zero mass at 5 and 9; a two-point mixture on {5,9} hits it
with all mass there. No published ranked field distinguishes them. So this
census does not report an absolute ranked share, because none is identified.

MAIN POSITIVE RESULT: the RATIO of time share to round share IS tightly
identified, because the mean-M constraint pins the mean round cost C_bar to a
few percent. That ratio is the reusable instrument: it converts any round-share
claim, including alphonse's, into a decode-time share.

The share and the mean round cost are linear / linear-fractional over a polytope
with two equalities, so their extrema sit at vertices with at most two non-zero
components; those are enumerated exactly rather than sampled. The amplification
ratio is not vertex-attained and is bounded by its two factors instead.
"""

from __future__ import annotations

import collections
import itertools
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TELEMETRY = ROOT / ".mlxfast-private/ranked-telemetry.json"
BASE_CURVE = ROOT / ".mlxfast-private/qmv-curve/e42-base/vendored.json"
BASE_LEG = ROOT / ".mlxfast-private/e42/runs/base/reports/leg-1/04-mtp-timed.json"
DRAFT_SIDE_SHAPE = "head.compact_draft_vocab"

NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
TARGET_WIDTHS = (5, 9)
PROMPTS_OF_INTEREST = ("beagle", "medicine")
# The thresholds the advisor asked this census to rule on. Quoted from the
# assignment feedback, not re-derived here.
ALPHONSE_BOUND = {"beagle": 0.0570, "medicine": 0.0374}
# Edward's derived ranked round counts, used only for the integrality remark.
RANKED_ROUNDS = {"beagle": 107, "medicine": 99}
# Non-QMV seconds per round, from E42's own ladder-slope intercept on this host.
NON_QMV_MS_PER_ROUND = 68.06


def load_qmv_cost_ms() -> dict[int, float]:
    """Q(M): predicted QMV ms per verify forward, from E42's base curve."""
    payload = json.loads(BASE_CURVE.read_text())
    out: dict[int, float] = {}
    for shape in payload["shapes"]:
        if shape["name"] == DRAFT_SIDE_SHAPE or not shape["calls_per_verify"]:
            continue
        for row in shape["rows"]:
            out[row["m"]] = out.get(row["m"], 0.0) + (
                shape["calls_per_verify"] * row["seconds_per_call"] * 1000.0
            )
    return out


def load_ranked_prompts() -> tuple[dict[str, dict], dict]:
    payload = json.loads(TELEMETRY.read_text())
    best = None
    for sub in payload["submissions"]:
        metrics = sub.get("officialMetrics") or {}
        if not metrics.get("per_prompt"):
            continue
        if "senpai" not in (sub.get("note") or "").lower():
            continue
        if best is None or (sub.get("officialScore") or 0) > (
            best.get("officialScore") or 0
        ):
            best = sub
    if best is None:
        raise SystemExit("e42_width_census: no senpai ranked row with per-prompt data")
    prompts = {
        NAMES[p["prompt_sha256"][:8]]: p for p in best["officialMetrics"]["per_prompt"]
    }
    return prompts, best


def load_local_histogram() -> tuple[dict[int, int], dict]:
    """The one FULLY measured width census available: E42's own base leg."""
    leg = json.loads(BASE_LEG.read_text())
    hist = collections.Counter(d + 1 for d in leg["effective_draft_lengths"])
    return dict(sorted(hist.items())), leg


def vertices(mean_m: float, support: tuple[int, ...]) -> list[dict[int, float]]:
    """Exact vertices of {rho >= 0, sum rho = 1, sum M rho = mean_m}."""
    out = []
    for m in support:
        if abs(m - mean_m) < 1e-12:
            out.append({m: 1.0})
    for lo, hi in itertools.combinations(support, 2):
        w_hi = (mean_m - lo) / (hi - lo)
        if -1e-12 <= w_hi <= 1 + 1e-12:
            out.append({lo: 1.0 - w_hi, hi: w_hi})
    return out


def mean_cost(rho: dict[int, float], cost: dict[int, float]) -> float:
    return sum(w * cost[m] for m, w in rho.items())


def time_share(rho: dict[int, float], cost: dict[int, float]) -> float:
    hit = sum(w * cost[m] for m, w in rho.items() if m in TARGET_WIDTHS)
    return hit / mean_cost(rho, cost)


def describe(rho: dict[int, float]) -> str:
    return " ".join(f"rho_{m}={w:.6f}" for m, w in sorted(rho.items()) if w > 1e-12)


def census(name: str, prompt: dict, cost: dict[int, float]) -> dict:
    mean_m = 1.0 + prompt["effective_mean_draft_len"]
    nd = prompt["non_drafting_round_count"]
    # nd = 0 removes M=1 from the support entirely; a non-zero nd would instead
    # pin rho_1 = nd/R and require the published round count.
    support = tuple(range(2, 10)) if nd == 0 else tuple(range(1, 10))
    verts = vertices(mean_m, support)

    first = lambda pair: pair[0]  # noqa: E731 - ties make dicts incomparable
    shares = sorted(((time_share(v, cost), v) for v in verts), key=first)
    cbars = sorted(((mean_cost(v, cost), v) for v in verts), key=first)

    # Amplification = time share / round share = (mean C over target mass) /
    # C_bar. It is a ratio of two linear-fractional forms, so unlike C_bar it is
    # NOT maximised at a polytope vertex; a vertex scan understates its range.
    # Bound the two factors independently instead: the numerator is a convex
    # combination of C(5) and C(9), and C_bar is an LP whose extrema are the
    # vertices above. That product bound is a guaranteed superset.
    target_costs = [cost[m] for m in TARGET_WIDTHS]
    amp_low = min(target_costs) / cbars[-1][0]
    amp_high = max(target_costs) / cbars[0][0]
    vertex_amps = sorted(
        time_share(v, cost) / sum(w for m, w in v.items() if m in TARGET_WIDTHS)
        for v in verts
        if sum(w for m, w in v.items() if m in TARGET_WIDTHS) > 1e-9
    )

    return {
        "mean_m": mean_m,
        "non_drafting_round_count": nd,
        "rho_1_pinned_to_zero": nd == 0,
        "support": list(support),
        "vertex_count": len(verts),
        "ranked_round_count": RANKED_ROUNDS[name],
        "rho_granularity": 1.0 / RANKED_ROUNDS[name],
        "absolute_time_share_low": shares[0][0],
        "absolute_time_share_high": shares[-1][0],
        "absolute_time_share_argmin_rho": describe(shares[0][1]),
        "absolute_time_share_argmax_rho": describe(shares[-1][1]),
        "mean_round_cost_ms_low": cbars[0][0],
        "mean_round_cost_ms_high": cbars[-1][0],
        "mean_round_cost_argmin_rho": describe(cbars[0][1]),
        "mean_round_cost_argmax_rho": describe(cbars[-1][1]),
        "amplification_low": amp_low,
        "amplification_high": amp_high,
        "amplification_bound_is_vertex_attained": False,
        "amplification_vertex_scan_low": vertex_amps[0],
        "amplification_vertex_scan_high": vertex_amps[-1],
        "alphonse_round_share": ALPHONSE_BOUND[name],
        "alphonse_converted_time_share_low": ALPHONSE_BOUND[name] * amp_low,
        "alphonse_converted_time_share_high": ALPHONSE_BOUND[name] * amp_high,
    }


def solve_normal_equations(x: list[list[float]], y: list[float]) -> list[float]:
    n = len(x[0])
    aug = [
        [sum(x[r][i] * x[r][j] for r in range(len(x))) for j in range(n)]
        + [sum(x[r][i] * y[r] for r in range(len(x)))]
        for i in range(n)
    ]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(aug[r][i]))
        aug[i], aug[piv] = aug[piv], aug[i]
        for r in range(n):
            if r != i:
                f = aug[r][i] / aug[i][i]
                for c in range(i, n + 1):
                    aug[r][c] -= f * aug[i][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


MODELS: dict[str, list] = {
    "linear": [lambda m: 1.0, lambda m: float(m)],
    "quadratic": [lambda m: 1.0, lambda m: float(m), lambda m: float(m * m)],
    "cubic": [
        lambda m: 1.0,
        lambda m: float(m),
        lambda m: float(m * m),
        lambda m: float(m**3),
    ],
    "step_ge6_plus_linear": [
        lambda m: 1.0,
        lambda m: float(m),
        lambda m: 1.0 if m >= 6 else 0.0,
    ],
    "step_ge6_plus_quadratic": [
        lambda m: 1.0,
        lambda m: float(m),
        lambda m: float(m * m),
        lambda m: 1.0 if m >= 6 else 0.0,
    ],
    # The mechanistic form: cost is (weight passes) x (per-pass linear cost),
    # where passes = ceil(M / IPG) is 1 for M<=5 and 2 for M>=6.
    "passes_x_linear": [
        lambda m: 1.0,
        lambda m: float(m),
        lambda m: 2.0 if m >= 6 else 1.0,
        lambda m: float(m) * (2.0 if m >= 6 else 1.0),
    ],
}


def discriminate(qmv: dict[int, float]) -> dict:
    """Priority B: does the local curve separate a step at M>=6 from a quadratic?

    The ranked corpus cannot: both families fit it with zero slack yet disagree
    4.4x on T(6)-T(5). Nine directly measured widths do separate them.
    """
    widths = sorted(qmv)
    y = [qmv[m] for m in widths]
    out: dict[str, dict] = {}
    for name, basis in MODELS.items():
        design = [[f(m) for f in basis] for m in widths]
        beta = solve_normal_equations(design, y)
        resid = [
            y[i] - sum(beta[j] * design[i][j] for j in range(len(beta)))
            for i in range(len(widths))
        ]
        rms = (sum(r * r for r in resid) / len(resid)) ** 0.5
        pred = {m: sum(beta[j] * basis[j](m) for j in range(len(beta))) for m in widths}
        out[name] = {
            "k_params": len(beta),
            "rms_ms": rms,
            "max_abs_residual_ms": max(abs(r) for r in resid),
            "residuals_ms": {str(m): resid[i] for i, m in enumerate(widths)},
            "predicted_step_6_minus_5_ms": pred[6] - pred[5],
        }
    first_diff = {
        str(widths[i]): y[i] - y[i - 1] for i in range(1, len(widths))
    }
    second_diff = {
        str(widths[i]): y[i] - 2 * y[i - 1] + y[i - 2] for i in range(2, len(widths))
    }
    return {
        "measured_step_6_minus_5_ms": qmv[6] - qmv[5],
        "first_differences_ms": first_diff,
        "second_differences_ms": second_diff,
        "second_difference_sign_changes": sum(
            1
            for a, b in zip(list(second_diff.values()), list(second_diff.values())[1:])
            if a * b < 0
        ),
        # A quadratic has CONSTANT second differences by construction, so any
        # sign change in the measured second differences refutes it outright,
        # independently of any fit residual.
        "quadratic_refuted_by_second_difference_sign_change": any(
            a * b < 0
            for a, b in zip(list(second_diff.values()), list(second_diff.values())[1:])
        ),
        "models": out,
    }


def local_census(hist: dict[int, int], cost: dict[int, float]) -> dict:
    total = sum(c * cost[m] for m, c in hist.items())
    hit = sum(c * cost[m] for m, c in hist.items() if m in TARGET_WIDTHS)
    rounds = sum(hist.values())
    hit_rounds = sum(c for m, c in hist.items() if m in TARGET_WIDTHS)
    return {
        "histogram": {str(m): c for m, c in hist.items()},
        "round_count": rounds,
        "mean_m": sum(m * c for m, c in hist.items()) / rounds,
        "numerator_ms": hit,
        "denominator_ms": total,
        "time_share": hit / total,
        "round_share": hit_rounds / rounds,
        "amplification": (hit / total) / (hit_rounds / rounds),
    }


def main() -> int:
    qmv = load_qmv_cost_ms()
    cost = {m: NON_QMV_MS_PER_ROUND + qmv[m] for m in qmv}
    ranked, submission = load_ranked_prompts()
    hist, leg = load_local_histogram()

    print("=== half 1 of 2: cost weights C(M), MEASURED on local M4 Pro ===")
    print("  source: E42 base curve (--shapes-only, 7 verify shapes, reps 21)")
    print("          + ladder-slope non-QMV intercept")
    print("   M     Q(M) ms    C(M) ms   weight_passes")
    for m in sorted(qmv):
        print(f"   {m}   {qmv[m]:9.3f}  {cost[m]:9.3f}       {1 if m <= 5 else 2}")
    print(f"\n  non-QMV per round = {NON_QMV_MS_PER_ROUND:.2f} ms")
    print(f"  Q(6)-Q(5) = {qmv[6] - qmv[5]:.3f} ms   <- the 1->2 weight-pass step")

    disc = discriminate(qmv)
    print("\n=== priority B: step at M>=6 vs plain quadratic, on 9 measured widths ===")
    print(
        "  measured Q(6)-Q(5) = "
        f"{disc['measured_step_6_minus_5_ms']:.3f} ms"
    )
    print(
        "  first  differences "
        + " ".join(f"{k}:{v:+7.2f}" for k, v in disc["first_differences_ms"].items())
    )
    print(
        "  second differences "
        + " ".join(f"{k}:{v:+7.2f}" for k, v in disc["second_differences_ms"].items())
    )
    print(
        f"  second-difference sign changes = {disc['second_difference_sign_changes']}"
        f" -> quadratic refuted outright: "
        f"{disc['quadratic_refuted_by_second_difference_sign_change']}"
    )
    print("  model                     k      rms ms   max|res|   predicted T(6)-T(5)")
    for name, res in disc["models"].items():
        print(
            f"  {name:24s} {res['k_params']}  {res['rms_ms']:10.4f} "
            f"{res['max_abs_residual_ms']:10.3f} "
            f"{res['predicted_step_6_minus_5_ms']:14.3f} ms"
        )

    print("\n=== the one FULLY measured decode-time census (public fixture) ===")
    loc = local_census(hist, cost)
    print(f"  fixture round widths: {loc['histogram']}")
    print(
        f"  rounds {loc['round_count']}, mean M {loc['mean_m']:.4f}, "
        f"accept rate {leg['accepted_draft_rate']:.4f}"
    )
    print(
        f"  numerator   = sum over M in {{5,9}} of count*C(M) = "
        f"{loc['numerator_ms']:.1f} ms"
    )
    print(
        f"  denominator = sum over all M of count*C(M)      = "
        f"{loc['denominator_ms']:.1f} ms"
    )
    print(
        f"  DECODE-TIME share = {loc['time_share']*100:.3f} %   "
        f"(round share {loc['round_share']*100:.3f} %, "
        f"amplification {loc['amplification']:.4f}x)"
    )
    print(
        "  CAVEAT: this fixture is a width CEILING (mean M "
        f"{loc['mean_m']:.2f} vs beagle 5.53); it is not beagle or medicine, and"
    )
    print(
        f"          {leg['verify_block_replayed_round_count']} of "
        f"{loc['round_count']} rounds replayed a verify block, so true cost at"
    )
    print("          those widths is slightly higher than one forward per round.")

    print("\n=== half 2 of 2: ranked round shares rho(M), PARTIALLY IDENTIFIED ===")
    print(
        f"  ranked row {submission['id'][:8]} score "
        f"{submission['officialScore']:.5f} commit "
        f"{submission['officialMetrics']['commit'][:12]}"
    )
    results = {}
    for name in PROMPTS_OF_INTEREST:
        rec = census(name, ranked[name], cost)
        results[name] = rec
        print(
            f"\n  {name}: mean M {rec['mean_m']:.4f}, "
            f"nd {rec['non_drafting_round_count']}, "
            f"support {rec['support'][0]}..{rec['support'][-1]}, "
            f"{rec['vertex_count']} vertices, R={rec['ranked_round_count']}"
        )
        print(
            f"    absolute decode-time share in "
            f"[{rec['absolute_time_share_low']*100:.3f} %, "
            f"{rec['absolute_time_share_high']*100:.3f} %]  <- VACUOUS, not reportable"
        )
        print(f"      argmin {rec['absolute_time_share_argmin_rho']}")
        print(f"      argmax {rec['absolute_time_share_argmax_rho']}")
        spread = (
            rec["mean_round_cost_ms_high"] / rec["mean_round_cost_ms_low"] - 1
        ) * 50
        print(
            f"    mean round cost C_bar in "
            f"[{rec['mean_round_cost_ms_low']:.2f}, "
            f"{rec['mean_round_cost_ms_high']:.2f}] ms  (+/- {spread:.2f} %)"
        )
        print(
            f"    AMPLIFICATION time_share/round_share in "
            f"[{rec['amplification_low']:.4f}x, {rec['amplification_high']:.4f}x]"
            "  <- IDENTIFIED"
        )
        print(
            f"      (vertex scan alone would say "
            f"[{rec['amplification_vertex_scan_low']:.4f}x, "
            f"{rec['amplification_vertex_scan_high']:.4f}x]; the ratio is not "
            "vertex-attained, so the wider bound above is the valid one)"
        )
        bound = rec["alphonse_round_share"]
        lo = rec["alphonse_converted_time_share_low"]
        hi = rec["alphonse_converted_time_share_high"]
        verdict = (
            "STRADDLES" if lo <= bound <= hi else ("ABOVE" if lo > bound else "BELOW")
        )
        print(
            f"    alphonse {bound*100:.2f} % read as a ROUND share converts to a "
            "decode-time share of"
        )
        print(
            f"      [{lo*100:.3f} %, {hi*100:.3f} %]  -> {verdict} the "
            f"{bound*100:.2f} % threshold"
        )

    out = {
        "submission_id": submission["id"],
        "submission_score": submission["officialScore"],
        "submission_commit": submission["officialMetrics"]["commit"],
        "cost_weights_host": "local-m4-pro",
        "cost_weights_source": "E42 base curve (--shapes-only) + ladder-slope intercept",
        "cost_weights_are_gate_qualified": False,
        "non_qmv_ms_per_round": NON_QMV_MS_PER_ROUND,
        "qmv_ms_per_round": {str(m): v for m, v in qmv.items()},
        "decode_ms_per_round": {str(m): v for m, v in cost.items()},
        "target_widths": list(TARGET_WIDTHS),
        "width_model_discrimination": disc,
        "local_fixture_census": loc,
        "ranked_census": results,
        "absolute_ranked_share_is_identified": False,
        "field_that_would_close_it": (
            "a per-round width histogram (or any second width moment) in "
            "officialMetrics.per_prompt"
        ),
    }
    path = ROOT / "research/e42-artifacts/width-census.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
