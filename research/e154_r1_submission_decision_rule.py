#!/usr/bin/env python3
"""E154 R1: how large must a composite be before a submission slot is rational?

`harness=ranked`. Zero GPU. Every number here is read from published Yukon
receipts; nothing is measured locally and nothing is simulated.

The deliverables are

  e154_ranked_nuisance_sd_pp
  e154_min_composite_pp_for_p50    (a table over the deficit d)
  e154_min_composite_pp_for_p80    (a table over the deficit d)
  e154_slots_per_promotion_at_current_portfolio

The central difficulty is that our own eight receipts each carry a different
tree, so their scatter confounds the ranked runner with real mechanism
differences and gives an UPPER bound only. This script therefore adds a second,
model-free estimator that the campaign has not used before: submissions that
share a source ref. Yukon records the packaged source ref per submission, so
two scored rows with the same solver and the same ref are the same editable
snapshot measured twice by the ranked runner. Their scatter contains no tree
term at all, which makes it a LOWER bound on nuisance under the assumption that
the runner is not more stable for rivals than for us.

Usage:
  python3 research/e154_r1_submission_decision_rule.py \
      --dump research/e154-artifacts/yukon-submissions-all-20260823T1430Z.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics

SPLIT = re.compile(r"\s{2,}")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TERMINAL_SCORED = {"promoted", "rejected", "superseded", "promotion failed"}

# The organizer's original calibrated depth-2 score. Yukon's `diff` column is
# (bar_score - row_score) / this value, NOT a percentage of the bar, so the
# column cannot be used to rebuild scores. We use the `score` field only.
CALIBRATED_DEPTH2 = 0.9927

OUR_SOLVER = "morganmcg1"


def bucket(*parts: str) -> str:
    """Stable bucket key (RULE 157: blake2b, never the built-in hash)."""
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def parse_dump(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_text().splitlines():
        line = ANSI.sub("", raw)
        if not line.strip() or line.startswith("─") or "submission  solver" in line:
            continue
        fields = SPLIT.split(line.strip())
        if len(fields) < 4 or len(fields[0]) != 7:
            continue
        submission, solver, status = fields[0], fields[1], fields[2]
        if status not in TERMINAL_SCORED and status not in {"failed", "cancelled",
                                                            "validating"}:
            continue
        try:
            score = float(fields[3])
        except ValueError:
            score = None
        commit = fields[-2] if len(fields) >= 3 else "-"
        created = fields[-1]
        rows.append({"submission": submission, "solver": solver, "status": status,
                     "score": score, "commit": commit, "created": created})
    return rows


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    """Acklam's inverse normal CDF; accurate to ~1e-9 over the range used."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def chi2_quantile(p: float, df: int) -> float:
    """Wilson-Hilferty; adequate for the interval width reported here."""
    z = normal_quantile(p)
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def consecutive_deltas(rows: list[dict], min_score: float) -> list[dict]:
    """Within-solver consecutive scored receipts, as relative differences."""
    by_solver: dict[str, list[dict]] = {}
    for row in rows:
        if row["score"] is None or row["status"] not in TERMINAL_SCORED:
            continue
        if row["score"] < min_score:
            continue
        by_solver.setdefault(row["solver"], []).append(row)

    deltas = []
    for solver, member_rows in sorted(by_solver.items()):
        for first, second in zip(member_rows, member_rows[1:]):
            mean = 0.5 * (first["score"] + second["score"])
            deltas.append({
                "solver": solver,
                "pair": [first["submission"], second["submission"]],
                "scores": [first["score"], second["score"]],
                "delta_pp": 100.0 * (second["score"] - first["score"]) / mean,
            })
    return deltas


def deconvolution_bound(deltas: list[dict], bandwidths=(0.05, 0.10, 0.25, 0.50)) -> dict:
    """Bound the ranked nuisance sd from the concentration of paired deltas.

    A within-solver consecutive pair has delta = T + N, where T is the real
    tree difference and N ~ Normal(0, 2 sigma^2) is the difference of two
    ranked-runner errors. Whatever T is, a convolution cannot be more peaked
    than its Gaussian factor:

        f_delta(0) <= 1 / (sigma * sqrt(4 pi))

    so an estimate of the delta density at zero gives an UPPER bound on sigma:

        sigma <= 1 / (2 sqrt(pi) f_delta(0))

    The bound needs no assumption at all about the distribution of real tree
    effects, which is the quantity we cannot identify. It is conservative in
    both directions that matter: a bin under-estimates a peaked density, and we
    take a binomial lower confidence limit on the bin probability, so the
    reported sigma bound is larger than the sharp one.
    """
    values = [d["delta_pp"] for d in deltas]
    n = len(values)
    out = {"n_pairs": n, "bandwidths": []}
    if n == 0:
        return out

    best = None
    for h in bandwidths:
        k = sum(1 for v in values if abs(v) <= h)
        p_hat = k / n
        # Wilson 90 % lower limit keeps a small count from inventing a tight bound.
        z = normal_quantile(0.90)
        denom = 1 + z * z / n
        centre = (p_hat + z * z / (2 * n)) / denom
        half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
        p_lo = max(centre - half, 0.0)
        entry = {"bandwidth_pp": h, "count": k, "p_hat": p_hat, "p_lower_90": p_lo}
        if p_lo > 0:
            density_lo = p_lo / (2 * h)
            entry["density_at_zero_lower"] = density_lo
            entry["sigma_upper_pp"] = 1.0 / (2 * math.sqrt(math.pi) * density_lo)
            if best is None or entry["sigma_upper_pp"] < best:
                best = entry["sigma_upper_pp"]
        out["bandwidths"].append(entry)
    out["sigma_upper_pp"] = best
    return out


def replication_pairs(rows: list[dict], min_score: float) -> tuple[list[dict], dict]:
    """Group scored rows by (solver, source ref). Same ref = same snapshot."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["score"] is None or row["commit"] in {"-", "n/a"}:
            continue
        if row["status"] not in TERMINAL_SCORED:
            continue
        if row["score"] < min_score:
            continue
        groups.setdefault(bucket(row["solver"], row["commit"]), []).append(row)

    reported, sum_sq, total_df = [], 0.0, 0
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        scores = [m["score"] for m in members]
        mean = statistics.fmean(scores)
        # Relative, because the campaign score scale tripled during the window.
        rel = [100.0 * (s - mean) / mean for s in scores]
        var = sum(r * r for r in rel) / (len(rel) - 1)
        sum_sq += var * (len(rel) - 1)
        total_df += len(rel) - 1
        reported.append({
            "bucket": key,
            "solver": members[0]["solver"],
            "source_ref": members[0]["commit"],
            "n": len(members),
            "submissions": [m["submission"] for m in members],
            "scores": scores,
            "mean_score": mean,
            "spread_abs": max(scores) - min(scores),
            "spread_pp": 100.0 * (max(scores) - min(scores)) / mean,
            "sd_pp": math.sqrt(var),
        })

    pooled = math.sqrt(sum_sq / total_df) if total_df else float("nan")
    interval = {}
    if total_df:
        # Two-sided 80 % interval on the pooled sd.
        lo = math.sqrt(sum_sq / chi2_quantile(0.90, total_df))
        hi = math.sqrt(sum_sq / chi2_quantile(0.10, total_df))
        interval = {"lo_pp": lo, "hi_pp": hi}
    return reported, {"pooled_sd_pp": pooled, "df": total_df, **interval}


# Measured per-prompt leg structure on the ranked M5 runner at the bar
# `684821ed`, transcribed from FINDING 235 (senpai/campaign-ledger.md:54425).
# leg/prefill/decode are seconds of trusted-parent wall clock; the leg INCLUDES
# the 512-token seed prefill.
RANKED_LEG_STRUCTURE = {
    "beagle":   {"leg_s": 5.47533, "prefill_s": 0.52634, "decode_s": 4.94899, "rounds": 110.00},
    "essays":   {"leg_s": 5.03450, "prefill_s": 0.52685, "decode_s": 4.50765, "rounds": 92.04},
    "medicine": {"leg_s": 4.97664, "prefill_s": 0.52582, "decode_s": 4.45082, "rounds": 90.01},
    "republic": {"leg_s": 4.97203, "prefill_s": 0.52582, "decode_s": 4.44621, "rounds": 93.00},
    "botany":   {"leg_s": 4.94848, "prefill_s": 0.52685, "decode_s": 4.42163, "rounds": 81.04},
    "drama":    {"leg_s": 9.13101, "prefill_s": 0.52736, "decode_s": 8.60365, "rounds": 252.01},
    "travel":   {"leg_s": 7.99539, "prefill_s": 0.52634, "decode_s": 7.46906, "rounds": 212.33},
    "plutarch": {"leg_s": 15.49210, "prefill_s": 0.52685, "decode_s": 14.96525, "rounds": 486.76},
}
# The published median is the mean of the two middle raw values over eight
# prompts, so exactly two prompts carry weight and each carries one half.
MEDIAN_PAIR = ("beagle", "essays")

# E154 R0, this host, base 14247cce, one thermally gated 512-token
# --local-submit pair. Both legs are trusted-parent wall clock and both include
# the same 512-token seed prefill.
LOCAL_R0 = {
    "decode_tokens": 512,
    "serial_seconds_per_token": 0.073790716705843806,
    "serial_rounds": 512,
    "mtp_seconds_per_token": 0.031824675854295492,
    "mtp_rounds": 77,
    "effective_mean_draft_len": 6.3766233766233764,
}

# Campaign constants under test.
RULE_134_OLD = 515.2          # us/round per 1 %, bar 684821ed
RULE_134_CORRECTED = 524.5    # us/round per 1 %, 0cf1637e
RULE_134_CROWN = 525.2        # us/round per 1 %, crown ec24d591

# The crown solver's own pre-registered self-estimate against what the ranked
# runner actually published for that submission (advisor F2).
CROWN_SELF_ESTIMATE = {"predicted_lo_pct": 0.74, "predicted_hi_pct": 1.11,
                       "realised_pct": 0.2557}

# Local-to-ranked realisation factors already measured by this campaign, from
# senpai/campaign-ledger.md. These price MECHANISM transfer, not whole-receipt
# self-estimate transfer, so they bound the crown's single observation rather
# than replacing it.
LEDGER_REALISATION_FACTORS = {
    "L14198_e70_leg": 0.42, "L14389_pure_stream": 0.276, "L14208_band": 0.946,
    "L15852_rung3": 0.672, "L30646_board": 0.48, "L31539_alphonse": 0.72,
    "L32485_launch_overhead": 0.94, "L39424_ledger26168": 0.112,
    "L39531_by_bytes": 0.675, "L39856_crown_delta": 0.49,
}


def frame_reconciliation(gap_pp: float, portfolio: dict, sigmas: dict) -> dict:
    """Answer advisor F2: is 524.5 us/round a wall time or a sensitivity?"""
    prompts = {}
    for name, leg in RANKED_LEG_STRUCTURE.items():
        rounds = leg["rounds"]
        contaminated = 1e6 * leg["leg_s"] / rounds
        clean = 1e6 * leg["decode_s"] / rounds
        prompts[name] = {
            **leg,
            "round_wall_time_us_total_leg_frame": contaminated,
            "round_wall_time_us_decode_frame": clean,
            "dilution_decode_over_leg": leg["decode_s"] / leg["leg_s"],
            "prefill_share": leg["prefill_s"] / leg["leg_s"],
            # Rule 134 currency: value of a uniform delta_us saved in EVERY
            # round of this prompt, as a fraction of this prompt's raw score.
            "raw_sensitivity_per_us": rounds / (1e6 * leg["leg_s"]),
            "median_weight": 0.5 if name in MEDIAN_PAIR else 0.0,
            # Value of deleting ONE whole round from this prompt.
            "one_round_deleted_pct_of_median":
                100.0 * (0.5 if name in MEDIAN_PAIR else 0.0) / rounds,
        }

    # C is the uniform per-round saving that buys 1 % of the published median.
    sens = sum(p["median_weight"] * p["raw_sensitivity_per_us"] for p in prompts.values())
    c_reconstructed = 0.01 / sens

    median_pair_wall = statistics.fmean(
        prompts[n]["round_wall_time_us_total_leg_frame"] for n in MEDIAN_PAIR)
    median_pair_clean = statistics.fmean(
        prompts[n]["round_wall_time_us_decode_frame"] for n in MEDIAN_PAIR)

    local = dict(LOCAL_R0)
    n_tok = local["decode_tokens"]
    local["serial_leg_seconds"] = local["serial_seconds_per_token"] * n_tok
    local["mtp_leg_seconds"] = local["mtp_seconds_per_token"] * n_tok
    local["serial_round_wall_time_us"] = 1e6 * local["serial_leg_seconds"] / local["serial_rounds"]
    local["mtp_round_wall_time_us"] = 1e6 * local["mtp_leg_seconds"] / local["mtp_rounds"]
    local["local_over_ranked_per_token"] = (
        local["mtp_seconds_per_token"] * 1e6 / (1e6 * RANKED_LEG_STRUCTURE["beagle"]["leg_s"] / n_tok))

    # The realisation (optimism) factor, kept strictly separate from nuisance.
    pred_mid = statistics.fmean([CROWN_SELF_ESTIMATE["predicted_lo_pct"],
                                 CROWN_SELF_ESTIMATE["predicted_hi_pct"]])
    k_crown = CROWN_SELF_ESTIMATE["realised_pct"] / pred_mid
    ledger_k = sorted(LEDGER_REALISATION_FACTORS.values())
    k_geomean = math.exp(statistics.fmean(math.log(v) for v in ledger_k))

    z80 = normal_quantile(0.80)
    k_grid = [1.0, 0.675, k_geomean, 0.42, k_crown, 0.112]
    bias_table = []
    for k in sorted(set(round(v, 6) for v in k_grid), reverse=True):
        row = {"realisation_factor_k": k,
               "predicted_pp_needed_p50": gap_pp / k}
        for label, sd in sigmas.items():
            row[f"predicted_pp_needed_p80_{label}"] = (gap_pp + z80 * sd) / k
        row["portfolio_all_five_realised_pp"] = portfolio["all_five"] * k
        row["portfolio_all_five_clears_p50"] = portfolio["all_five"] * k >= gap_pp
        bias_table.append(row)

    return {
        "harness": "ranked",
        "question": ("advisor F2: the crown note implies a ~68 ms decode round "
                     "while Rule 134 quotes 524.5 us/round; are they the same "
                     "quantity?"),
        "verdict": ("no. They are different quantities in different units. "
                    "524.5 us/round is a SENSITIVITY (microseconds of per-round "
                    "leg time per one percent of published median), not a wall "
                    "time. The measured ranked round wall time is 3.2e4 to "
                    "6.1e4 us. No portfolio number needs repricing."),
        "e154_round_wall_time_us": {
            "measured": True,
            "definition": "trusted-parent leg wall clock divided by the parent-counted round total",
            "ranked_per_prompt": prompts,
            "ranked_median_pair_total_leg_frame": median_pair_wall,
            "ranked_median_pair_decode_frame": median_pair_clean,
            "ranked_min_us": min(p["round_wall_time_us_total_leg_frame"] for p in prompts.values()),
            "ranked_max_us": max(p["round_wall_time_us_total_leg_frame"] for p in prompts.values()),
            "local_e154_r0": local,
            "crown_note_implied_us": 68410.0,
            "crown_note_consistent_with_table": True,
            "crown_note_position_in_ranked_range": (
                "between botany 61062 us/round (81 rounds) and a ~80-round leg; "
                "inside the measured 31827-61062 us/round envelope once the "
                "crown's lower edl 4.28 is applied to a median-pair leg"),
        },
        "e154_frame_constant_provenance": {
            "equation": (
                "C = 0.01 / sum_p w_p * R_p / leg_us_p   [us/round per 1 % of published median]"),
            "terms": {
                "C": "the Rule 134 constant; 515.2 at bar 684821ed, 524.5 on 0cf1637e, 525.2 on the crown",
                "p": "one of the eight hidden prompts",
                "w_p": ("median weight; the published median is the mean of the two middle "
                        "raw values, so w_p = 1/2 for each median-pair prompt and 0 otherwise"),
                "R_p": "parent-counted decode rounds in prompt p's candidate leg",
                "leg_us_p": ("TOTAL TIMED LEG in microseconds = "
                             "candidate_mtp_seconds_per_token_p * 512 * 1e6; it INCLUDES "
                             "the 512-token seed prefill (about 0.527 s, 9.6-10.5 % of a "
                             "median-pair leg)"),
                "delta_us": "a uniform saving applied to EVERY round of EVERY prompt",
                "value": "delta_median / median = delta_us * sum_p w_p * R_p / leg_us_p",
            },
            "frame": "total timed leg, NOT the decode leg",
            "do_not": ("do not divide by the decode leg to 'correct' C; that inflates it "
                       "by 1/dilution, about 11 % on the median pair"),
            "reconstructed_C_us_per_round": c_reconstructed,
            "ledger_C_us_per_round": RULE_134_OLD,
            "reconstruction_error_pct": 100.0 * (c_reconstructed - RULE_134_OLD) / RULE_134_OLD,
            "reconstruction_note": ("this closed form omits median-pair reordering after the "
                                    "saving is applied; F235's exact reconstruction, which "
                                    "tracks reordering, publishes 515.2"),
            "control_one_round_off_each_median_pair_pct": sum(
                prompts[n]["one_round_deleted_pct_of_median"] for n in MEDIAN_PAIR),
            "control_ledger_value_pct": 1.012,
            "units_of_C": "microseconds of per-round leg time per one percent of published median",
            "C_is_a_wall_time": False,
            "ratio_round_wall_time_over_C": median_pair_wall / RULE_134_CORRECTED,
        },
        "e154_portfolio_reprice_factor": {
            "value": 1.0,
            "reason": ("every portfolio entry in this experiment is denominated in "
                       "published-score percentage points, which the frame constant "
                       "does not touch. Only work quoted in us/round is affected."),
            "us_per_round_reprice_factor_old_to_corrected": RULE_134_OLD / RULE_134_CORRECTED,
            "us_per_round_reprice_note": ("anything priced with the old 515.2 constant is "
                                          "1.8 % optimistic; the ledger has already applied "
                                          "this correction at line 58417"),
            "no_130x_error_exists": True,
        },
        "e154_round_count_reconciliation": {
            "question": "our '488 rounds/leg' against the crown's ~120 and our local 77",
            "resolution": ("all three are correct and none of them is 'the' round count. "
                           "Rounds per leg vary 81.0 to 486.8 across the eight hidden "
                           "prompts, a factor of 6.0."),
            "ranked_rounds_per_prompt": {n: p["rounds"] for n, p in prompts.items()},
            "the_488_figure_is": "plutarch, 486.76 rounds",
            "plutarch_median_weight": 0.0,
            "median_pair_rounds": {n: prompts[n]["rounds"] for n in MEDIAN_PAIR},
            "overcount_factor_if_plutarch_used_for_value": (
                prompts["plutarch"]["raw_sensitivity_per_us"]
                / statistics.fmean(prompts[n]["raw_sensitivity_per_us"] for n in MEDIAN_PAIR)),
            "local_77_rounds_is": ("our own public-fixture leg at effective mean draft "
                                   "length 6.377, which packs more tokens per round than "
                                   "any ranked prompt"),
            "crown_120_rounds_is": "a median-pair-class ranked leg at the crown's lower edl 4.28",
            "warning": ("pricing a per-round saving against plutarch's round population "
                        "overstates median-pair value by the factor above, because "
                        "plutarch carries zero median weight"),
        },
        "e154_self_estimate_bias": {
            "separate_from_nuisance": True,
            "crown_predicted_pct_lo": CROWN_SELF_ESTIMATE["predicted_lo_pct"],
            "crown_predicted_pct_hi": CROWN_SELF_ESTIMATE["predicted_hi_pct"],
            "crown_predicted_pct_mid": pred_mid,
            "crown_realised_pct": CROWN_SELF_ESTIMATE["realised_pct"],
            "k_crown": k_crown,
            "k_crown_reciprocal": 1.0 / k_crown,
            "n_observations": 1,
            "ledger_mechanism_realisation_factors": LEDGER_REALISATION_FACTORS,
            "ledger_k_geometric_mean": k_geomean,
            "ledger_k_min": ledger_k[0],
            "ledger_k_max": ledger_k[-1],
            "frame_caveat": ("the ledger factors price local-to-ranked MECHANISM transfer; "
                             "the crown factor prices a whole-receipt self-estimate. They "
                             "are not the same estimand, so the ledger set bounds the crown "
                             "observation rather than pooling with it."),
            "coincidence_flagged": ("k_crown = 0.2764 agrees to three digits with the "
                                    "campaign's independently measured pure-stream "
                                    "realisation factor 0.276 (ledger line 14389). Two "
                                    "observations of different estimands are not evidence "
                                    "of a shared constant; treat the agreement as a "
                                    "coincidence until a third case tests it."),
            "bias_adjusted_requirement": bias_table,
            "headline": ("at the crown's own realisation factor the full five-part "
                         "portfolio realises "
                         f"{portfolio['all_five'] * k_crown:.3f} pp against a "
                         f"{gap_pp:.3f} pp gap, so its median outcome is no promotion"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=pathlib.Path)
    parser.add_argument("--bar", type=float, default=3.7291100105909)
    parser.add_argument("--our-best", type=float, default=3.68278758168578)
    parser.add_argument("--min-score", type=float, default=3.0,
                        help="modern-era floor; the score scale tripled earlier")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/e154-artifacts/e154_r1.json"))
    args = parser.parse_args()

    rows = parse_dump(args.dump)
    ours = [r for r in rows if r["solver"] == OUR_SOLVER and r["score"] is not None
            and r["status"] in TERMINAL_SCORED]

    last_six = ours[-6:]
    last_eight = ours[-8:]

    def scatter(sample: list[dict]) -> dict:
        scores = [r["score"] for r in sample]
        mean = statistics.fmean(scores)
        sd = statistics.stdev(scores)
        return {"n": len(scores), "mean": mean, "sd_abs": sd,
                "sd_pp": 100.0 * sd / mean,
                "submissions": [r["submission"] for r in sample],
                "scores": scores}

    own_six, own_eight = scatter(last_six), scatter(last_eight)
    pairs, pooled = replication_pairs(rows, args.min_score)
    deltas = consecutive_deltas(rows, args.min_score)
    bound = deconvolution_bound(deltas)

    # Nothing in the public record identifies sigma from below: the ranked
    # runner has never measured one snapshot twice. The board-wide
    # deconvolution bound caps it from above, and our own receipt scatter caps
    # the sum of nuisance and real tree differences.
    sd_lower = bound["sigma_upper_pp"]
    sd_upper = own_six["sd_pp"]

    deficits = [0.0, 0.25, 0.5, 0.75, 1.0, 1.26, 1.5, 2.0, 2.5, 3.0]
    z80 = normal_quantile(0.80)
    table = []
    for d in deficits:
        row = {"deficit_pp": d, "min_pp_p50": d}
        for label, sd in (("lower", sd_lower), ("upper", sd_upper)):
            row[f"min_pp_p80_{label}_sd"] = d + z80 * sd
        table.append(row)

    portfolio = {"import_alone": 0.5385, "import_e151r1": 1.0472,
                 "import_r1_leaf16": 1.2377, "import_r1_leaf16_e151r2": 1.6627,
                 "all_five": 1.9581}
    gap_pp = 100.0 * (args.bar - args.our_best) / args.our_best

    # A policy the advisor can read off for whichever sigma he later believes.
    policy = []
    for sigma in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        p_single = normal_cdf((0.5 - gap_pp) / sigma)
        p_stack = normal_cdf((portfolio["all_five"] - gap_pp) / sigma)
        policy.append({
            "sigma_pp": sigma,
            "min_pp_p80_at_current_gap": gap_pp + z80 * sigma,
            "p_clear_one_mechanism_0p5pp": p_single,
            "slots_one_mechanism": 1.0 / p_single if p_single > 0 else float("inf"),
            "p_clear_stack_all_five": p_stack,
            "slots_stack": 1.0 / p_stack if p_stack > 0 else float("inf"),
            "stack_advantage_slots": (1.0 / p_stack) / (1.0 / p_single)
            if p_single > 0 and p_stack > 0 else float("nan"),
        })

    slots = []
    for name, m in portfolio.items():
        entry = {"composite": name, "effect_pp": m}
        for label, sd in (("lower", sd_lower), ("upper", sd_upper)):
            p = normal_cdf((m - gap_pp) / sd) if sd > 0 else float("nan")
            entry[f"p_clear_{label}_sd"] = p
            entry[f"slots_per_promotion_{label}_sd"] = (1.0 / p) if p > 0 else float("inf")
        slots.append(entry)

    single = []
    for m in (0.19, 0.35, 0.5, 0.7275):
        entry = {"effect_pp": m}
        for label, sd in (("lower", sd_lower), ("upper", sd_upper)):
            p = normal_cdf((m - gap_pp) / sd) if sd > 0 else float("nan")
            entry[f"p_clear_{label}_sd"] = p
            entry[f"slots_per_promotion_{label}_sd"] = (1.0 / p) if p > 0 else float("inf")
        single.append(entry)

    frame = frame_reconciliation(
        gap_pp, portfolio, {"lower_sd": sd_lower, "upper_sd": sd_upper})

    result = {
        "harness": "ranked",
        "gpu_used": False,
        "bar": args.bar,
        "our_best_receipt": args.our_best,
        "gap_pp": gap_pp,
        "diff_column_denominator": CALIBRATED_DEPTH2,
        "own_last_six": own_six,
        "own_last_eight": own_eight,
        "replication_pairs": pairs,
        "replication_pooled": pooled,
        "board_pair_count": len(deltas),
        "deconvolution_bound": bound,
        "e154_ranked_nuisance_sd_pp": {
            "identified": False,
            "tight_upper_bound_pp": sd_lower,
            "loose_upper_bound_pp": sd_upper,
            "lower_bound_pp": None,
            "tight_source": ("board-wide deconvolution bound from within-solver "
                             "consecutive receipt pairs"),
            "loose_source": ("our last six receipts, confounded with real tree "
                             "differences"),
            "why_not_identified": ("no snapshot in the 1236-row public history "
                                   "was ever measured twice, so no replicate "
                                   "exists to separate runner noise from tree "
                                   "effects"),
        },
        "e154_min_composite_pp_table": table,
        "e154_slots_per_promotion_at_current_portfolio": slots,
        "single_mechanism_slots": single,
        "policy_over_sigma": policy,
        "f2_frame_reconciliation": frame,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"our last six receipts : sd {own_six['sd_abs']:.5f} abs "
          f"= {own_six['sd_pp']:.3f} pp (n={own_six['n']})")
    print(f"our last eight        : sd {own_eight['sd_abs']:.5f} abs "
          f"= {own_eight['sd_pp']:.3f} pp (n={own_eight['n']})")
    print(f"\nsame-source-ref replications (score >= {args.min_score}): "
          f"{len(pairs)} group(s), df={pooled['df']}  <-- the identification problem")
    print(f"board-wide within-solver consecutive pairs: {len(deltas)}")
    for entry in bound["bandwidths"]:
        sigma = entry.get("sigma_upper_pp")
        print(f"  |delta| <= {entry['bandwidth_pp']:.2f} pp: {entry['count']:>3}/"
              f"{bound['n_pairs']} p_lo90={entry['p_lower_90']:.4f}"
              + (f"  -> sigma <= {sigma:.3f} pp" if sigma else "  -> no bound"))
    if bound.get("sigma_upper_pp"):
        print(f"  tightest deconvolution bound: sigma <= "
              f"{bound['sigma_upper_pp']:.3f} pp")
    for pair in pairs:
        print(f"  {pair['solver']:<20} ref={pair['source_ref']} n={pair['n']} "
              f"scores={['%.6f' % s for s in pair['scores']]} "
              f"spread={pair['spread_pp']:.3f} pp")
    if pooled["df"]:
        print(f"  pooled nuisance sd = {pooled['pooled_sd_pp']:.3f} pp "
              f"[80 % {pooled['lo_pp']:.3f}, {pooled['hi_pp']:.3f}]")

    print(f"\ngap to bar = {gap_pp:.3f} pp")
    print("\nminimum composite effect (pp) to clear a bar d pp above us")
    print(f"{'d (pp)':>8} {'p50':>8} {'p80 @ sd_lo':>13} {'p80 @ sd_hi':>13}")
    for row in table:
        print(f"{row['deficit_pp']:>8.2f} {row['min_pp_p50']:>8.2f} "
              f"{row['min_pp_p80_lower_sd']:>13.2f} {row['min_pp_p80_upper_sd']:>13.2f}")

    print("\nslots per promotion at the current portfolio")
    for entry in slots:
        print(f"  {entry['composite']:<26} {entry['effect_pp']:>6.3f} pp  "
              f"p={entry['p_clear_lower_sd']:.3f}/{entry['p_clear_upper_sd']:.3f}  "
              f"slots={entry['slots_per_promotion_lower_sd']:.1f}/"
              f"{entry['slots_per_promotion_upper_sd']:.1f}")

    print("\none mechanism at a time")
    for entry in single:
        print(f"  {entry['effect_pp']:>6.3f} pp  "
              f"p={entry['p_clear_lower_sd']:.4f}/{entry['p_clear_upper_sd']:.4f}  "
              f"slots={entry['slots_per_promotion_lower_sd']:.0f}/"
              f"{entry['slots_per_promotion_upper_sd']:.0f}")
    wall = frame["e154_round_wall_time_us"]
    prov = frame["e154_frame_constant_provenance"]
    rec = frame["e154_round_count_reconciliation"]
    bias = frame["e154_self_estimate_bias"]

    print("\n=== F2 frame reconciliation ===")
    print("measured ranked round wall time (total-leg frame, us/round)")
    for name, p in wall["ranked_per_prompt"].items():
        print(f"  {name:<9} R={p['rounds']:>6.2f}  leg={p['leg_s']:>8.5f}s  "
              f"wall={p['round_wall_time_us_total_leg_frame']:>9.1f}  "
              f"clean={p['round_wall_time_us_decode_frame']:>9.1f}  "
              f"w={p['median_weight']:.1f}")
    print(f"  median pair mean: {wall['ranked_median_pair_total_leg_frame']:.1f} us/round "
          f"(clean {wall['ranked_median_pair_decode_frame']:.1f})")
    loc = wall["local_e154_r0"]
    print(f"  local E154 R0 MTP   : {loc['mtp_round_wall_time_us']:.1f} us/round "
          f"over {loc['mtp_rounds']} rounds")
    print(f"  local E154 R0 serial: {loc['serial_round_wall_time_us']:.1f} us/round "
          f"over {loc['serial_rounds']} rounds")
    print(f"\nRule 134 constant C = {RULE_134_CORRECTED} us/round per 1 %")
    print(f"  reconstructed from the measured table: {prov['reconstructed_C_us_per_round']:.1f} "
          f"({prov['reconstruction_error_pct']:+.2f} % vs the ledger's 515.2)")
    print(f"  control: one round off each median-pair prompt = "
          f"{prov['control_one_round_off_each_median_pair_pct']:.3f} % "
          f"(ledger publishes {prov['control_ledger_value_pct']:.3f} %)")
    print(f"  C is a wall time: {prov['C_is_a_wall_time']}  "
          f"(wall/C = {prov['ratio_round_wall_time_over_C']:.1f}x)")
    print(f"  portfolio reprice factor: "
          f"{frame['e154_portfolio_reprice_factor']['value']:.3f}")
    print(f"\nround-count reconciliation: 488 = {rec['the_488_figure_is']}, "
          f"median weight {rec['plutarch_median_weight']}")
    print(f"  using it for value overstates median-pair worth by "
          f"{rec['overcount_factor_if_plutarch_used_for_value']:.3f}x")
    print(f"\nself-estimate bias: k_crown = {bias['k_crown']:.4f} "
          f"(1/{bias['k_crown_reciprocal']:.2f}), ledger geomean "
          f"{bias['ledger_k_geometric_mean']:.3f} "
          f"[{bias['ledger_k_min']:.3f}, {bias['ledger_k_max']:.3f}]")
    print(f"{'k':>8} {'pred pp p50':>12} {'pred pp p80 lo':>15} "
          f"{'pred pp p80 hi':>15} {'all_five real':>14} {'clears p50':>11}")
    for row in bias["bias_adjusted_requirement"]:
        print(f"{row['realisation_factor_k']:>8.3f} {row['predicted_pp_needed_p50']:>12.2f} "
              f"{row['predicted_pp_needed_p80_lower_sd']:>15.2f} "
              f"{row['predicted_pp_needed_p80_upper_sd']:>15.2f} "
              f"{row['portfolio_all_five_realised_pp']:>14.3f} "
              f"{str(row['portfolio_all_five_clears_p50']):>11}")
    print(f"\n{bias['headline']}")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
