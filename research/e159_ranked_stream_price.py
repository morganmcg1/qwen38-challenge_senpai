"""E159: price the ranked weight-stream term with the tree's own structured law.

E95 states the verify cost model as `verify_us = a + b*G + c*M`, where
`G = ceil(M / IPG(M))` is the number of quantized weight streams the wide QMV
launches and `M` is the verify width. My first pass fitted seven free per-width
marginals to the board, which the board cannot identify: its `width` column is a
mean and it is collinear with schedule identity within a prompt.

The structured law has three parameters for eight ranked prompts, so it is
well posed on the anchor receipt alone. It also returns the one coefficient the
campaign already has an independent board estimate for.
"""

import json
import math
import pathlib

import numpy as np

ART = pathlib.Path(__file__).resolve().parent / "e159-artifacts"

# Ranked anchor receipt 5a9f130a, FINDING 330. `q` is the mean proposed draft
# depth, so the mean verify width is q + 1. `round_ms` is the paired per-prompt
# candidate-leg round time.
ANCHOR = [
    # name, rounds, q, accepted, round_ms, serial_ms, raw
    ("plutarch", 488, 0.1557, 0.049, 30.522, 37.9740, 1.2607),
    ("drama", 252, 2.2976, 1.032, 34.241, 37.9454, 2.1222),
    ("travel", 213, 2.6479, 1.404, 35.109, 37.9646, 2.4277),
    ("beagle", 110, 4.3818, 3.655, 44.983, 37.9208, 3.5453),
    ("republic", 93, 4.9892, 4.505, 47.723, 38.0130, 3.9199),
    ("essays", 92, 5.0870, 4.565, 48.932, 38.0201, 3.8704),
    ("medicine", 90, 5.2556, 4.689, 49.350, 37.9068, 3.9065),
    ("botany", 81, 6.1481, 5.321, 54.474, 37.9495, 3.9332),
]

INPUTS_PER_GROUP = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}

# Independent board estimate of the same coefficient, from the fixed-M
# cross-tree contrast <T,8,4> vs <T,8,3> (streams 2 -> 3).
LEDGER_BOARD_STREAM_MS = 20.291
# Local M4 Pro estimate from the E159 dense fixed-depth sweep.
LOCAL_STREAM_MS = 30.212
LOCAL_STREAM_SE_MS = 3.330


def streams(m: int) -> int:
    if m <= 1:
        return 1
    ipg = INPUTS_PER_GROUP[m]
    return (m + ipg - 1) // ipg


def two_point(mean_width: float):
    """Split a mean width over its two bracketing integers."""
    lo = int(math.floor(mean_width))
    hi = lo + 1
    p_hi = mean_width - lo
    return [(lo, 1.0 - p_hi), (hi, p_hi)]


def expected_streams(mean_width: float) -> float:
    return sum(p * streams(m) for m, p in two_point(mean_width))


def ols(x, y, names):
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - float(resid @ resid) / float((y - y.mean()) @ (y - y.mean()))
    return {
        "coefficients": {
            n: {"value": round(float(b), 4), "se": round(float(s), 4)}
            for n, b, s in zip(names, beta, se)
        },
        "residual_sd_ms": round(math.sqrt(sigma2), 4),
        "r_squared": round(r2, 5),
        "dof": dof,
    }, beta


def local_levels_fit():
    """The same law on the local dense sweep, where widths are integers.

    The ranked prompts only supply mean widths, and over their range the stream
    count is nearly a clipped linear function of width. The local sweep has one
    leg per integer width from 2 to 9, so the two regressors separate.
    """
    r_dec = [69.484, 71.561, 78.323, 91.634, 127.084, 138.065, 145.655, 185.427]
    w = np.arange(2.0, 10.0)
    g = np.array([float(streams(int(m))) for m in w])
    y = np.array(r_dec)
    ones = np.ones_like(y)

    with_stream, _ = ols(np.column_stack([ones, g, w]), y, ["a", "b_stream", "c_row"])
    linear_only, _ = ols(np.column_stack([ones, w]), y, ["a", "c_row"])
    return {
        "width_stream_correlation": round(float(np.corrcoef(w, g)[0, 1]), 4),
        "fit_with_stream": with_stream,
        "fit_linear_only": linear_only,
    }


def main() -> None:
    rows = []
    for name, rounds, q, acc, round_ms, serial_ms, raw in ANCHOR:
        width = q + 1.0
        rows.append(
            {
                "prompt": name,
                "rounds": rounds,
                "mean_verify_width": round(width, 4),
                "expected_streams": round(expected_streams(width), 4),
                "round_ms": round_ms,
                "accepted": acc,
                "raw": raw,
            }
        )

    width = np.array([r["mean_verify_width"] for r in rows])
    g = np.array([r["expected_streams"] for r in rows])
    y = np.array([r["round_ms"] for r in rows])
    ones = np.ones_like(y)

    full, beta_full = ols(np.column_stack([ones, g, width]), y, ["a", "b_stream", "c_row"])
    no_stream, _ = ols(np.column_stack([ones, width]), y, ["a", "c_row"])

    # Drop plutarch: 449 of its 488 rounds do not draft at all, so it is nearly
    # a serial control and it anchors the intercept on its own.
    keep = np.array([r["prompt"] != "plutarch" for r in rows])
    drop_plutarch, _ = ols(
        np.column_stack([ones[keep], g[keep], width[keep]]), y[keep],
        ["a", "b_stream", "c_row"],
    )

    # Is the ranked fit able to separate a stream step from a linear width
    # term at all? Over the ranked width range E[streams] is close to a clipped
    # linear function of width, so check the collinearity before trusting it.
    ranked_corr = float(np.corrcoef(width[keep], g[keep])[0, 1])
    local = local_levels_fit()

    # Can the ranked anchor hide a large stream penalty behind the linear width
    # term? Pin b_stream and refit the remaining two coefficients. A per-row
    # cost that must go negative to absorb the pinned value is a falsification.
    pinned = []
    for b_fixed in (0.0, 2.0, 5.0, 10.0, 20.291, 30.212):
        adj = y[keep] - b_fixed * g[keep]
        fit, beta = ols(
            np.column_stack([ones[keep], width[keep]]), adj, ["a", "c_row"]
        )
        pinned.append(
            {
                "b_stream_pinned_ms": b_fixed,
                "c_row_ms": fit["coefficients"]["c_row"]["value"],
                "residual_sd_ms": fit["residual_sd_ms"],
            }
        )

    # Leave one prompt out of the eight-prompt fit.
    loo = {}
    for i, r in enumerate(rows):
        m = np.ones(len(rows), dtype=bool)
        m[i] = False
        fit, _ = ols(
            np.column_stack([ones[m], g[m], width[m]]), y[m], ["a", "b_stream", "c_row"]
        )
        loo[r["prompt"]] = {
            "b_stream_ms": fit["coefficients"]["b_stream"]["value"],
            "c_row_ms": fit["coefficients"]["c_row"]["value"],
            "residual_sd_ms": fit["residual_sd_ms"],
        }

    # The advisor's median-setting pair, priced directly and without any fit.
    beagle = next(r for r in rows if r["prompt"] == "beagle")
    essays = next(r for r in rows if r["prompt"] == "essays")
    d_round = essays["round_ms"] - beagle["round_ms"]
    d_width = essays["mean_verify_width"] - beagle["mean_verify_width"]
    d_streams = essays["expected_streams"] - beagle["expected_streams"]
    median_pair = {
        "delta_round_ms": round(d_round, 4),
        "delta_mean_width": round(d_width, 4),
        "delta_expected_streams": round(d_streams, 4),
        "implied_ms_per_row_if_no_stream_term": round(d_round / d_width, 4),
        "local_stream_cost_would_add_ms": round(d_streams * LOCAL_STREAM_MS, 4),
        "residual_row_cost_if_local_stream_cost_held": round(
            (d_round - d_streams * LOCAL_STREAM_MS) / d_width, 4
        ),
    }

    b = float(beta_full[1])
    b_se = float(full["coefficients"]["b_stream"]["se"])

    def z(other, other_se):
        return (b - other) / math.sqrt(b_se**2 + other_se**2)

    def price_stream_removal(b_stream: float):
        """Value of driving every verify width back to one weight stream.

        Reported on the paired per-prompt candidate leg per RULE 177, and then
        pushed through the published median, which is the mean of the two
        middle raw values over eight prompts.
        """
        per_prompt = {}
        raws = []
        for (name, rounds, q, acc, round_ms, serial_ms, raw), gi in zip(ANCHOR, g):
            saved_per_round = max(gi - 1.0, 0.0) * b_stream
            mtp_now = round_ms / (acc + 1.0)
            saved_per_token = saved_per_round / (acc + 1.0)
            # Scale into the reported candidate mtp, which also carries the
            # prompt's prefill share, so the round saving is not the whole leg.
            reported_mtp = round_ms / (acc + 1.0)
            frac = saved_per_token / reported_mtp if reported_mtp else 0.0
            new_raw = raw / (1.0 - frac) if frac < 1.0 else float("inf")
            per_prompt[name] = {
                "expected_streams": round(float(gi), 4),
                "saved_ms_per_round": round(saved_per_round, 4),
                "round_pct": round(100.0 * saved_per_round / round_ms, 4),
                "candidate_leg_pct": round(100.0 * frac, 4),
                "raw_now": raw,
                "raw_after": round(new_raw, 5),
            }
            raws.append((name, raw, new_raw))
            del mtp_now
        before = sorted(r for _, r, _ in raws)
        after = sorted(r for _, _, r in raws)
        med_before = 0.5 * (before[3] + before[4])
        med_after = 0.5 * (after[3] + after[4])
        return {
            "b_stream_ms": round(b_stream, 4),
            "per_prompt": per_prompt,
            "published_median_before": round(med_before, 6),
            "published_median_after": round(med_after, 6),
            "published_median_pct": round(
                100.0 * (med_after - med_before) / med_before, 4
            ),
            "same_sign_prompts": sum(
                1 for v in per_prompt.values() if v["candidate_leg_pct"] > 0
            ),
            "candidate_leg_mean_pct": round(
                sum(v["candidate_leg_pct"] for v in per_prompt.values()) / 8.0, 4
            ),
        }

    arm = {
        "ranked_point_estimate": price_stream_removal(b),
        "ranked_lower_1se": price_stream_removal(max(b - b_se, 0.0)),
        "ranked_upper_1se": price_stream_removal(b + b_se),
        "if_local_price_held": price_stream_removal(LOCAL_STREAM_MS),
    }

    result = {
        "experiment": "e159-ranked-stream-price",
        "harness": "ranked (anchor receipt 5a9f130a), structured law refit",
        "official_or_ranked_score": False,
        "law": "round_ms = a + b_stream * E[streams] + c_row * E[verify_width]",
        "law_source": "E95QmvWidthProbeTests.swift: verify_us = a + b*G + c*M",
        "per_prompt": rows,
        "fit_full": full,
        "fit_without_stream_term": no_stream,
        "fit_without_plutarch": drop_plutarch,
        "comparison": {
            "ranked_fit_b_ms": round(b, 4),
            "ranked_fit_b_se_ms": round(b_se, 4),
            "ledger_board_fixed_m_contrast_ms": LEDGER_BOARD_STREAM_MS,
            "z_vs_ledger_board": round(z(LEDGER_BOARD_STREAM_MS, 0.0), 3),
            "local_m4pro_dense_sweep_ms": LOCAL_STREAM_MS,
            "z_vs_local": round(z(LOCAL_STREAM_MS, LOCAL_STREAM_SE_MS), 3),
        },
        "ranked_width_stream_correlation_drafting_prompts": round(ranked_corr, 4),
        "ranked_pinned_stream_scan": pinned,
        "ranked_loo": loo,
        "median_setting_pair_direct": median_pair,
        "arm_value_remove_second_stream": arm,
        "local_dense_sweep": local,
        "identification_warning": (
            "E[streams] saturates at 2.0 for essays, medicine and botany, so "
            "b_stream is identified almost entirely by the beagle-to-republic "
            "transition. Treat the standard error as optimistic."
        ),
    }

    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "e159_ranked_stream_price.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    print("prompt      rounds  meanM   E[G]   round_ms")
    for r in rows:
        print(
            "  %-9s %5d  %6.4f  %5.3f  %8.3f"
            % (r["prompt"], r["rounds"], r["mean_verify_width"],
               r["expected_streams"], r["round_ms"])
        )
    print()
    for label, fit in (
        ("a + b*G + c*M          ", full),
        ("a + c*M (no stream)    ", no_stream),
        ("a + b*G + c*M, no plut ", drop_plutarch),
    ):
        coef = "  ".join(
            "%s=%.3f+-%.3f" % (k, v["value"], v["se"]) for k, v in fit["coefficients"].items()
        )
        print("%s  %s   resid_sd=%.3f  R2=%.5f"
              % (label, coef, fit["residual_sd_ms"], fit["r_squared"]))
    print()
    print("ranked b_stream = %.3f +- %.3f ms" % (b, b_se))
    print("  vs ledger board fixed-M contrast %.3f ms   z = %+.2f"
          % (LEDGER_BOARD_STREAM_MS, z(LEDGER_BOARD_STREAM_MS, 0.0)))
    print("  vs local M4 Pro dense sweep %.3f +- %.3f ms   z = %+.2f"
          % (LOCAL_STREAM_MS, LOCAL_STREAM_SE_MS, z(LOCAL_STREAM_MS, LOCAL_STREAM_SE_MS)))
    print()
    print("identifiability: corr(width, E[streams])")
    print("  ranked drafting prompts  %.4f" % ranked_corr)
    print("  local integer sweep      %.4f" % local["width_stream_correlation"])
    for label, fit in (
        ("local a + b*G + c*M   ", local["fit_with_stream"]),
        ("local a + c*M         ", local["fit_linear_only"]),
    ):
        coef = "  ".join(
            "%s=%.3f+-%.3f" % (k, v["value"], v["se"]) for k, v in fit["coefficients"].items()
        )
        print("%s  %s   resid_sd=%.3f  R2=%.5f"
              % (label, coef, fit["residual_sd_ms"], fit["r_squared"]))
    print()
    print("ranked anchor, 7 drafting prompts, b_stream pinned:")
    print("  b_pin_ms   c_row_ms   resid_sd_ms")
    for p in pinned:
        print("  %8.3f  %9.3f  %11.3f" % (p["b_stream_pinned_ms"], p["c_row_ms"], p["residual_sd_ms"]))
    print()
    print("LOO b_stream (all 8, drop one):")
    for k, v in loo.items():
        print("  drop %-9s b=%+8.3f  c_row=%+7.3f  resid_sd=%.3f" % (k, v["b_stream_ms"], v["c_row_ms"], v["residual_sd_ms"]))
    print()
    print("median-setting pair, beagle -> essays, no fit:")
    for k, v in median_pair.items():
        print("  %-46s %+9.4f" % (k, v))
    print()
    for lbl, k in (("point b=%.3f"%b,"ranked_point_estimate"),("b-1se","ranked_lower_1se"),("b+1se","ranked_upper_1se"),("local price","if_local_price_held")):
        a2=arm[k]
        print("arm %-16s median %.5f -> %.5f  (%+.3f%%)  cand-leg mean %+.3f%%  same-sign %d/8"
              % (lbl,a2["published_median_before"],a2["published_median_after"],a2["published_median_pct"],a2["candidate_leg_mean_pct"],a2["same_sign_prompts"]))
    print()
    print("per-prompt at the ranked point estimate:")
    for k2,v2 in arm["ranked_point_estimate"]["per_prompt"].items():
        print("  %-9s E[G]=%.3f  save %6.3f ms/round (%5.2f%% of round)  cand-leg %+6.3f%%  raw %.5f -> %.5f"
              % (k2,v2["expected_streams"],v2["saved_ms_per_round"],v2["round_pct"],v2["candidate_leg_pct"],v2["raw_now"],v2["raw_after"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
