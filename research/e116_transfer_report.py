#!/usr/bin/env python3
"""E116 rungs 3 and 4: how much of a round's time reaches the leg, and what is
the composed transfer from a kernel percent to an absolute leg percent?

    usage: research/e116_transfer_report.py LEG_DIR [LEG_DIR ...]
               --dose-unit-us US --alpha A [--alpha-half-width H]
               [--wide-qmv-share S] [--json OUT]

RUNG 3. A four-point dose ladder, `k = 0, 4, 8, 12`, every round dosed. The
endpoint is ABSOLUTE candidate MTP seconds per token as the trusted parent
measures it (`parent_measured_seconds_per_token` from `mtp-timed`), not a
serial-to-MTP ratio. The local ratio cancels any change that speeds both legs;
the absolute leg time does not.

    beta = d(leg us per token) / ( d(round us) x R / 512 )

`beta = 1.00` is the algebraic prediction that all round time flows to the leg.
E94's constant residual of 7.82 to 8.00 ms per token predicts about 0.75.

ESTIMATORS, NAMED, WITH THEIR SE (campaign rule 39).

  1. `ols`: ordinary least squares of leg microseconds per token on dose units
     over all legs. SE from the residual mean square. This is the headline.
  2. `arm_means`: the k=12 arm mean minus the k=0 arm mean, divided by 12. SE
     from the two arm SEMs. Reported so a reader can see the endpoints without
     the linearity assumption.
  3. `neighbour`: the neighbour-averaged contrast. Each leg of the high arm is
     contrasted against the mean of the legs of the low arm that bracket it in
     session order, which removes a linear session drift. Reported wherever the
     ordering allows it.

A nonlinear response is itself a result, so the OLS residuals are printed.

RUNG 4. The composition the campaign uses is

    predicted leg %  =  arm % of wide-QMV kernel time
                        x wide-QMV share of round
                        x alpha
                        x beta

That form is only correct when the share divides the LEG-AMORTISED round,
`leg us/token x tokens / R`, because that is the only frame in which a percent
of the round and a percent of the leg are the same number. Dividing by the
parent `block_request` frame instead inflates the coefficient by 1/0.768.

Pass `--wide-qmv-us-per-round`, the absolute measured microseconds, and this
file forms the share against the leg-amortised frame itself. `--wide-qmv-share`
remains available for a share that is already in that frame; passing a share
measured against any other frame is the exact defect this experiment exists to
remove.

Prefer `--wide-qmv-us-per-leg`, the census total over the whole 512-token
window. The leg endpoint carries no round count, so a census taken under a
harness that segments the same window into a different number of rounds still
divides correctly. `--local-iterate` reports 78 MTP rounds for the window that
`mtp-timed` reports as 77, so the per-round forms are NOT interchangeable
between the two harnesses.

RULE 34. Two round frames appear here and they are named separately:
`e116_rung3_control_round_us` is the mean parent-measured
`block_request_seconds` over rounds 1..R-1 of the k=0 legs of THIS ladder.
Nothing in this file divides by a wall/rounds frame or by a seed model.

RULE 37. One dose unit is one M=1 `mlp.gate_up`-shaped affine 4-bit group-64
QMV. `--dose-unit-us` is an M=1 rate and is never a scored-width rate.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e109_v2_report import read_meta, t95  # noqa: E402
from e116_absorption_report import dose_of, histogram  # noqa: E402


def leg_record(leg_dir: pathlib.Path) -> dict:
    report = json.loads((leg_dir / "report.json").read_text())
    meta = read_meta(leg_dir / "meta.txt")
    us = [s * 1e6 for s in report["block_request_seconds"]]
    widths = list(report["effective_draft_lengths"])
    return {
        "leg": leg_dir.name,
        "harness": "local",
        "arm_label": meta.get("arm_label"),
        "arm_env": meta.get("arm_env", ""),
        "dose_units": dose_of(meta.get("arm_env", "")),
        "block": int(meta.get("block", -1)),
        "all_tokens_matched": report["all_tokens_matched"],
        "decode_token_count": report["decode_token_count"],
        "round_count": report["round_count"],
        "leg_us_per_token": report["parent_measured_seconds_per_token"] * 1e6,
        "decode_seconds": report.get("decode_seconds"),
        "seed_prefill_seconds": report.get("seed_prefill_seconds"),
        "effective_mean_draft_len": report.get("effective_mean_draft_len"),
        "accepted_draft_total": report.get("accepted_draft_total"),
        "rejected_draft_total": report.get("rejected_draft_total"),
        "control_round_us": statistics.fmean(us[1:]) if len(us) > 1 else
        float("nan"),
        "width_histogram": histogram([k + 1 for k in widths]),
        # Rule 40: the per-round series travels with the result.
        "round_us": [round(value, 1) for value in us],
        "round_width": [k + 1 for k in widths],
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "worker_sha256": meta.get("worker_sha256"),
        "git_head": meta.get("git_head"),
        "git_dirty_build": meta.get("git_dirty_build"),
        "leg_wall_seconds": meta.get("leg_wall_seconds"),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
    }


def ols(xs: list[float], ys: list[float]) -> dict:
    n = len(xs)
    if n < 3:
        return {"n": n, "slope": float("nan"), "slope_se": float("nan"),
                "intercept": float("nan"), "residuals": [],
                "r_squared": float("nan")}
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in residuals)
    sst = sum((y - mean_y) ** 2 for y in ys)
    mse = sse / (n - 2)
    slope_se = math.sqrt(mse / sxx)
    half = t95(n - 2) * slope_se
    return {"n": n, "slope": slope, "slope_se": slope_se,
            "slope_half_width": half,
            "slope_ci95": [slope - half, slope + half],
            "intercept": intercept, "residuals": residuals,
            "residual_sd": math.sqrt(mse),
            "r_squared": 1.0 - sse / sst if sst > 0 else float("nan")}


def arm_summary(legs: list[dict], key: str) -> dict:
    values = [leg[key] for leg in legs]
    n = len(values)
    mean = statistics.fmean(values) if n else float("nan")
    sd = statistics.stdev(values) if n > 1 else float("nan")
    return {"legs": n, "mean": mean, "sd": sd,
            "sem": sd / math.sqrt(n) if n > 1 else float("nan"),
            "values": values}


def neighbour_contrast(legs: list[dict], low: int, high: int,
                       key: str) -> dict:
    """Each high-arm leg against the mean of its bracketing low-arm legs.

    Session order is the leg's own index in the ordered list, so a linear
    drift over the session cancels in every contrast that has a low-arm leg on
    each side. A high-arm leg with a low-arm leg on only one side falls back to
    that one neighbour, and the count of one-sided contrasts is reported.
    """
    ordered = sorted(legs, key=lambda leg: leg["leg"])
    lows = [(i, leg) for i, leg in enumerate(ordered)
            if leg["dose_units"] == low]
    highs = [(i, leg) for i, leg in enumerate(ordered)
             if leg["dose_units"] == high]
    if not lows or not highs:
        return {"n": 0}
    contrasts, one_sided = [], 0
    for index, leg in highs:
        before = [l for j, l in lows if j < index]
        after = [l for j, l in lows if j > index]
        if before and after:
            reference = (before[-1][key] + after[0][key]) / 2.0
        elif before:
            reference = before[-1][key]
            one_sided += 1
        else:
            reference = after[0][key]
            one_sided += 1
        contrasts.append(leg[key] - reference)
    n = len(contrasts)
    mean = statistics.fmean(contrasts)
    sd = statistics.stdev(contrasts) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    half = t95(n - 1) * sem if n > 1 else float("nan")
    return {"n": n, "one_sided": one_sided, "mean": mean, "sd": sd,
            "sem": sem, "half_width": half,
            "ci95": [mean - half, mean + half], "contrasts": contrasts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("legs", nargs="+")
    ap.add_argument("--dose-unit-us", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="rung 2 absorption coefficient")
    ap.add_argument("--alpha-half-width", type=float, default=float("nan"))
    ap.add_argument("--wide-qmv-share", type=float, default=float("nan"),
                    help="rung 4 wide-QMV share, already in the "
                         "leg-amortised round frame")
    ap.add_argument("--wide-qmv-us-per-round", type=float,
                    default=float("nan"),
                    help="rung 4 absolute measured wide-QMV microseconds per "
                         "round; the share is formed here against the "
                         "leg-amortised frame")
    ap.add_argument("--wide-qmv-us-per-leg", type=float,
                    default=float("nan"),
                    help="rung 4 absolute measured wide-QMV microseconds "
                         "summed over the whole 512-token leg; the share is "
                         "formed against the leg endpoint and no round count "
                         "enters, so a census taken under a harness that "
                         "segments the same window into a different number "
                         "of rounds stays comparable")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--json")
    args = ap.parse_args()

    legs = [leg_record(pathlib.Path(p)) for p in args.legs]
    legs.sort(key=lambda leg: leg["leg"])
    doses = sorted({leg["dose_units"] for leg in legs})
    by_dose = {k: [leg for leg in legs if leg["dose_units"] == k]
               for k in doses}

    base_legs = by_dose.get(0, [])
    control_round_us = (statistics.fmean(l["control_round_us"]
                                         for l in base_legs)
                        if base_legs else float("nan"))
    rounds = statistics.fmean(leg["round_count"] for leg in legs)
    tokens = statistics.fmean(leg["decode_token_count"] for leg in legs)
    # RULE 34, THE TENTH FRAME. `e116_leg_amortised_round_us` is the leg's
    # whole cost divided by its round count: it carries the seed prefill and
    # every between-round cost the parent pays, and it is the only frame in
    # which a percent of the leg endpoint is also a percent of a round. The
    # parent's own `block_request_seconds` mean is the ninth frame's
    # descendant and is reported beside it, so the gap between them is the
    # per-round share of everything that is not inside a block request.
    leg_amortised_round_us = (
        statistics.fmean(leg["leg_us_per_token"] for leg in base_legs)
        * tokens / rounds if base_legs else float("nan"))

    fit = ols([float(leg["dose_units"]) for leg in legs],
              [leg["leg_us_per_token"] for leg in legs])

    arms = {k: arm_summary(v, "leg_us_per_token") for k, v in by_dose.items()}
    top = doses[-1]
    arm_delta_per_unit = float("nan")
    arm_half_per_unit = float("nan")
    if 0 in arms and arms[top]["legs"] and arms[0]["legs"]:
        delta = arms[top]["mean"] - arms[0]["mean"]
        variance, df = 0.0, 0
        for side in (arms[top], arms[0]):
            if side["legs"] > 1 and not math.isnan(side["sem"]):
                variance += side["sem"] ** 2
                df += side["legs"] - 1
        half = t95(df) * math.sqrt(variance) if df > 0 else float("nan")
        arm_delta_per_unit = delta / top
        arm_half_per_unit = half / top

    neighbour = neighbour_contrast(legs, 0, top, "leg_us_per_token")

    # beta. d(round us)/dk is alpha x the M=1 dose rate; the round happens R
    # times per `tokens` tokens, so the leg-side prediction of one dose unit is
    # alpha x dose_unit_us x R / tokens microseconds per token.
    predicted_leg_us_per_token_per_unit = (
        args.alpha * args.dose_unit_us * rounds / tokens)
    beta = fit["slope"] / predicted_leg_us_per_token_per_unit
    beta_half = (fit["slope_half_width"] / predicted_leg_us_per_token_per_unit
                 if not math.isnan(fit.get("slope_half_width", float("nan")))
                 else float("nan"))

    # alpha CANCELS in the product. beta divides by alpha and the composition
    # multiplies by it, so `alpha x beta` is
    #     slope / (dose_unit_us x R / tokens)
    # which depends only on the ladder slope and the M=1 dose rate. The rung 2
    # coefficient therefore decides how the transfer SPLITS between round
    # absorption and round-to-leg flow; it cannot move the composed number, and
    # an error in alpha cannot propagate into the headline. The composed CI is
    # taken straight from the ladder slope for that reason.
    per_unit_at_unit_transfer = args.dose_unit_us * rounds / tokens
    composed = fit["slope"] / per_unit_at_unit_transfer
    composed_half = (fit["slope_half_width"] / per_unit_at_unit_transfer
                     if not math.isnan(
                         fit.get("slope_half_width", float("nan")))
                     else float("nan"))
    leg_total_us = (statistics.fmean(leg["leg_us_per_token"]
                                     for leg in base_legs) * tokens
                    if base_legs else float("nan"))
    given = [n for n, v in (("--wide-qmv-share", args.wide_qmv_share),
                            ("--wide-qmv-us-per-round",
                             args.wide_qmv_us_per_round),
                            ("--wide-qmv-us-per-leg",
                             args.wide_qmv_us_per_leg))
             if not math.isnan(v)]
    if len(given) > 1:
        raise SystemExit("pass exactly one of " + ", ".join(given))
    share = args.wide_qmv_share
    if not math.isnan(args.wide_qmv_us_per_round):
        share = args.wide_qmv_us_per_round / leg_amortised_round_us
    elif not math.isnan(args.wide_qmv_us_per_leg):
        share = args.wide_qmv_us_per_leg / leg_total_us
    composed_with_share = composed * share

    out = {
        "harness": "local",
        "experiment":
            "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "rung": 3,
        "endpoint": "absolute candidate MTP microseconds per token, trusted "
                    "parent measured (parent_measured_seconds_per_token)",
        "round_frame": "e116 rung 3 = mean parent block_request_seconds over "
                       "rounds 1..R-1 of the k=0 legs of this ladder",
        "e116_rung3_control_round_us": control_round_us,
        "e116_leg_amortised_round_us": leg_amortised_round_us,
        "block_request_share_of_leg_amortised_round":
            control_round_us / leg_amortised_round_us
            if leg_amortised_round_us else float("nan"),
        "mean_round_count": rounds,
        "mean_decode_token_count": tokens,
        "dose_unit_us_m1_census": args.dose_unit_us,
        "alpha": args.alpha,
        "alpha_half_width": args.alpha_half_width,
        "doses": doses,
        "estimator_ols": fit,
        "estimator_arm_means": {
            "per_arm": {str(k): {kk: vv for kk, vv in a.items()}
                        for k, a in arms.items()},
            "top_dose": top,
            "slope_per_unit": arm_delta_per_unit,
            "slope_half_width_per_unit": arm_half_per_unit,
        },
        "estimator_neighbour": neighbour,
        "predicted_leg_us_per_token_per_dose_unit": (
            predicted_leg_us_per_token_per_unit),
        "beta": beta,
        "beta_half_width": beta_half,
        "beta_ci95": [beta - beta_half, beta + beta_half],
        "round_to_leg_alpha_times_beta": composed,
        "round_to_leg_alpha_times_beta_half_width": composed_half,
        "round_to_leg_alpha_times_beta_ci95": [composed - composed_half,
                                               composed + composed_half],
        "alpha_cancels_in_the_product": True,
        "wide_qmv_share_of_leg_amortised_round_measured": share,
        "wide_qmv_us_per_round_measured": args.wide_qmv_us_per_round,
        "wide_qmv_us_per_leg_measured": args.wide_qmv_us_per_leg,
        "e116_leg_total_us": leg_total_us,
        "share_frame": "wide-QMV microseconds divided by the e116 tenth "
                       "frame: the leg-amortised round, or equivalently the "
                       "whole leg when the census total is supplied",
        "composed_kernel_percent_to_leg_percent": composed_with_share,
        "composed_kernel_percent_to_leg_percent_ci95": [
            (composed - composed_half) * share,
            (composed + composed_half) * share],
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "legs": legs,
    }

    print("E116 rung 3 -- dose ladder to absolute leg seconds per token"
          "   harness=local")
    print("  ungated: cool_gate_passed_real_gate=false,"
          " gate_qualified_for_timing=false, official_or_ranked_score=false")
    print(f"  endpoint: {out['endpoint']}")
    print(f"  round frame: {out['round_frame']}"
          f"  = {control_round_us:,.0f} us")
    print(f"  tenth frame: e116 leg-amortised round"
          f" = {leg_amortised_round_us:,.0f} us"
          f"  (block requests are"
          f" {100.0 * out['block_request_share_of_leg_amortised_round']:.1f} %"
          " of it)")
    print(f"  mean rounds per leg {rounds:.2f} over {tokens:.0f} tokens;"
          f" dose unit {args.dose_unit_us:.2f} us at M=1; alpha {args.alpha:.3f}")
    print()
    print(f"{'leg':<14} {'arm':<6} {'k':>3} {'matched':>8} {'rounds':>7}"
          f" {'us/token':>11} {'round us':>10} {'draft':>6} {'entry C':>8}")
    for leg in legs:
        print(f"{leg['leg']:<14} {str(leg['arm_label']):<6}"
              f" {leg['dose_units']:>3} {str(leg['all_tokens_matched']):>8}"
              f" {leg['round_count']:>7} {leg['leg_us_per_token']:>11,.1f}"
              f" {leg['control_round_us']:>10,.0f}"
              f" {leg['effective_mean_draft_len'] or 0:>6.2f}"
              f" {str(leg['gpu_temp_entry_c']):>8}")
    print()
    for k in doses:
        a = arms[k]
        print(f"  k={k:<3} n={a['legs']} mean {a['mean']:,.1f} us/token"
              f"  sd {a['sd']:.1f}  sem {a['sem']:.1f}"
              f"  ({100.0 * (a['mean'] - arms[0]['mean']) / arms[0]['mean']:+.3f} %"
              f" vs k=0)")
    print()
    print(f"  ols        slope {fit['slope']:+.2f}"
          f" +-{fit.get('slope_half_width', float('nan')):.2f} us/token"
          f" per dose unit   R^2 {fit['r_squared']:.4f}"
          f"   residual sd {fit['residual_sd']:.2f}")
    print(f"             residuals "
          + " ".join(f"{r:+.1f}" for r in fit["residuals"]))
    print(f"  arm_means  slope {arm_delta_per_unit:+.2f}"
          f" +-{arm_half_per_unit:.2f} us/token per dose unit"
          f"  (k={top} minus k=0)")
    if neighbour.get("n"):
        print(f"  neighbour  k={top} minus bracketing k=0 mean:"
              f" {neighbour['mean']:+.1f} +-{neighbour['half_width']:.1f}"
              f" us/token over n={neighbour['n']}"
              f" ({neighbour['one_sided']} one sided)"
              f"  -> {neighbour['mean'] / top:+.2f} per dose unit")
    print()
    print(f"  predicted per dose unit at beta=1:"
          f" {predicted_leg_us_per_token_per_unit:+.2f} us/token")
    print(f"  beta = {beta:.3f}  95% CI [{out['beta_ci95'][0]:.3f},"
          f" {out['beta_ci95'][1]:.3f}]")
    print(f"  alpha x beta = {composed:.3f}"
          f"  95% CI [{composed - composed_half:.3f},"
          f" {composed + composed_half:.3f}]"
          f"   (alpha cancels; this is slope / (dose_unit_us x R / tokens))")
    if not math.isnan(share):
        print(f"  measured wide-QMV share of the leg-amortised round"
              f" = {share:.4f}")
        print(f"  composed kernel% -> leg% transfer ="
              f" {composed_with_share:.4f}"
              f"  95% CI [{(composed - composed_half) * share:.4f},"
              f" {(composed + composed_half) * share:.4f}]")

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
