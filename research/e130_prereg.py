#!/usr/bin/env python3
"""E130 rung 7 pre-registration.

Fixes the readout statistic, the noise model, the decision thresholds and the
promotion arithmetic BEFORE the ranked receipt for submission 0c6191b7 lands,
so no statistic can be chosen after seeing the answer.

harness = ranked for every quantity here.

Usage:
    python3 research/e130_prereg.py \
        --board /tmp/yukon-board/full.json \
        --out research/e130-artifacts/rung7-preregistration.json
"""

from __future__ import annotations

import argparse
import json
import math

# --- fixed model constants, all recorded before the receipt exists ----------

# Census weighted-residency gain of the prune_na5_pair arm on the g17s cell.
# research/e130-artifacts/rung3-shipped-arm.json ->
#   e130_ranked_g17s_f47_weighted_residency_pct = +12.82
G_RESIDENCY = 0.1282

# Proposal-head share of candidate decode time. FINDING 143 (ledger 280)
# brackets this at 7 %-9 % from four independent lines.
S_HEAD_GRID = (0.07, 0.08, 0.09)
S_HEAD_CENTRAL = 0.08

# Residency -> time coupling coefficient. F8 working bracket.
C_GRID_LO, C_GRID_HI = 0.199, 0.301
C_CENTRAL = 0.25

PARENT_RECEIPT = "cf79f7df-305e-4ffd-b78f-4f65f5c3b0dd"
CANDIDATE_SUBMISSION = "0c6191b7-215b-4dfc-873f-2449fbce5416"


def saving_pct(s_head: float, c: float) -> float:
    """Predicted reduction in candidate mtp seconds per token, percent."""
    return s_head * G_RESIDENCY * c * 100.0


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation without a SciPy dependency."""
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else float("nan")


def rank_diagnostics(family: dict) -> dict:
    """Does the published median rank receipts by real candidate speed?

    Across the whole schedule family, speed differences are real and large, so
    the median must track them. Inside one speed tier, the real differences sit
    below the instrument sd, so any surviving rank information is noise.
    """
    members = family["members"]
    spt = [m["candidate_mean_spt"] for m in members]
    score = [m["score"] for m in members]
    out = {
        "positive_control_across_family": {
            "n": len(members),
            "spearman_spt_vs_score": spearman(spt, score),
            "expected": "strongly negative: lower decode time scores higher",
        },
        "within_tier": [],
    }
    for tier in family["speed_tiers"]:
        ids = set(tier["members"])
        sel = [m for m in members if m["id"] in ids]
        if len(sel) < 3:
            continue
        s = [m["candidate_mean_spt"] for m in sel]
        v = [m["score"] for m in sel]
        rng_pct = 100.0 * (max(s) - min(s)) / (sum(s) / len(s))
        score_rng_pct = 100.0 * (max(v) - min(v)) / (sum(v) / len(v))
        out["within_tier"].append(
            {
                "rank": tier["rank"],
                "n": len(sel),
                "candidate_spt_range_pct": rng_pct,
                "score_range_pct": score_rng_pct,
                "spearman_spt_vs_score": spearman(s, v),
            }
        )
    return out


def load_tier0(board_path: str, family: dict) -> dict:
    tier0 = next(t for t in family["speed_tiers"] if t["rank"] == 0)
    with open(board_path) as fh:
        rows = json.load(fh)
    by_id = {r["id"]: r for r in rows}
    members = []
    for rid in tier0["members"]:
        row = by_id[rid]
        m = row["officialMetrics"]
        members.append(
            {
                "receipt": rid,
                "short": rid[:8],
                "solver": row["solverUsername"],
                "promoted_source_ref": row.get("promotedSourceRef"),
                "promotion_status": row.get("promotionStatus"),
                "official_score": row["officialScore"],
                "candidate_mtp_seconds_per_token_mean": m[
                    "candidate_mtp_seconds_per_token_mean"
                ],
                "baseline_serial_seconds_per_token_mean": m[
                    "baseline_serial_seconds_per_token_mean"
                ],
            }
        )
    members.sort(key=lambda x: x["candidate_mtp_seconds_per_token_mean"])
    return {"tier": tier0, "members": members}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="/tmp/yukon-board/full.json")
    ap.add_argument("--family", default="research/e130-artifacts/rung6-schedule-family.json")
    ap.add_argument("--sd", default="research/e130-artifacts/rung6-instrument-sd.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.sd) as fh:
        sd_art = json.load(fh)

    sd_cand_pooled = sd_art["schedule_tiers"]["candidate_rel_sd_pct"]
    sd_score_pooled = sd_art["schedule_tiers"]["score_rel_sd_pct"]

    with open(args.family) as fh:
        family = json.load(fh)

    t0 = load_tier0(args.board, family)
    tier0 = t0["tier"]
    members = t0["members"]
    sd_cand_tier0 = tier0["candidate_rel_sd_pct"]
    sd_score_tier0 = tier0["score_rel_sd_pct"]

    parent = next(m for m in members if m["receipt"] == PARENT_RECEIPT)
    tier_best_score = max(members, key=lambda m: m["official_score"])

    # --- prediction grid ---------------------------------------------------
    grid = [
        {
            "s_head": s,
            "c": c,
            "predicted_candidate_saving_pct": saving_pct(s, c),
        }
        for s in S_HEAD_GRID
        for c in (C_GRID_LO, C_CENTRAL, C_GRID_HI)
    ]
    pred_lo = min(g["predicted_candidate_saving_pct"] for g in grid)
    pred_hi = max(g["predicted_candidate_saving_pct"] for g in grid)
    pred_central = saving_pct(S_HEAD_CENTRAL, C_CENTRAL)

    # --- noise model for a single receipt versus a single reference --------
    root2 = math.sqrt(2.0)
    n_tier = len(members)
    # one new receipt against the mean of the n existing tier-0 receipts
    mix = math.sqrt(1.0 + 1.0 / n_tier)

    noise = {
        "candidate_spt_paired_vs_parent": {
            "conservative_pooled_sd_pct": root2 * sd_cand_pooled,
            "optimistic_tier0_sd_pct": root2 * sd_cand_tier0,
        },
        "candidate_spt_vs_tier0_mean": {
            "conservative_pooled_sd_pct": mix * sd_cand_pooled,
            "optimistic_tier0_sd_pct": mix * sd_cand_tier0,
        },
        "published_median_paired_vs_parent": {
            "conservative_pooled_sd_pct": root2 * sd_score_pooled,
            "optimistic_tier0_sd_pct": root2 * sd_score_tier0,
        },
    }

    sig_cand = noise["candidate_spt_paired_vs_parent"]["conservative_pooled_sd_pct"]
    sig_median = noise["published_median_paired_vs_parent"][
        "conservative_pooled_sd_pct"
    ]

    # --- power of each statistic at the predicted effect -------------------
    def power_z(effect_pct: float, sigma_pct: float) -> float:
        return effect_pct / sigma_pct

    power = {
        "candidate_spt": {
            "z_at_prediction_floor": power_z(pred_lo, sig_cand),
            "z_at_prediction_central": power_z(pred_central, sig_cand),
            "z_at_prediction_ceiling": power_z(pred_hi, sig_cand),
            "resolves_prediction_at_2sigma": power_z(pred_lo, sig_cand) >= 2.0,
        },
        "published_median": {
            "z_at_prediction_floor": power_z(pred_lo, sig_median),
            "z_at_prediction_central": power_z(pred_central, sig_median),
            "z_at_prediction_ceiling": power_z(pred_hi, sig_median),
            "resolves_prediction_at_2sigma": power_z(pred_lo, sig_median) >= 2.0,
        },
    }

    # --- pre-registered decision rule --------------------------------------
    two_sigma = 2.0 * sig_cand
    decision_rule = {
        "headline_statistic": "candidate_mtp_seconds_per_token_mean",
        "headline_reference": PARENT_RECEIPT,
        "headline_delta_definition": (
            "delta_pct = 100 * (parent_candidate_spt - candidate_spt) "
            "/ parent_candidate_spt; positive means the arm is faster"
        ),
        "secondary_statistic": "officialScore (published median of raw_p)",
        "secondary_is_underpowered": not power["published_median"][
            "resolves_prediction_at_2sigma"
        ],
        "sigma_pct": sig_cand,
        "outcomes": [
            {
                "label": "A_confirmed_at_predicted_magnitude",
                "condition_pct": f"delta_pct >= {pred_lo:.4f}",
                "meaning": "coupling c lies at or above the F8 bracket floor",
            },
            {
                "label": "B_real_but_sub_predicted",
                "condition_pct": f"{two_sigma:.4f} <= delta_pct < {pred_lo:.4f}",
                "meaning": "effect is real at 2 sigma but c is below the F8 bracket",
            },
            {
                "label": "C_null",
                "condition_pct": f"abs(delta_pct) < {two_sigma:.4f}",
                "meaning": (
                    "no resolvable effect; report the 2 sigma upper bound on c "
                    "rather than a point estimate"
                ),
            },
            {
                "label": "D_harmful",
                "condition_pct": f"delta_pct <= -{two_sigma:.4f}",
                "meaning": "the arm costs time; the residency model has the wrong sign",
            },
        ],
        "c_estimator": (
            "c_hat = delta_pct / (s_head * G_RESIDENCY * 100); report the band "
            "over s_head in {0.07, 0.09} and delta_pct +/- 2 sigma"
        ),
        "c_upper_bound_under_null": (two_sigma) / (0.07 * G_RESIDENCY * 100.0),
    }

    # --- promotion arithmetic ---------------------------------------------
    # raw_p = serial / candidate, so a candidate saving of x percent lifts every
    # raw_p, and therefore the median, by 1/(1-x) - 1.
    def lifted(score: float, saving_frac: float) -> float:
        return score / (1.0 - saving_frac)

    gap_abs = tier_best_score["official_score"] - parent["official_score"]
    gap_pct = 100.0 * gap_abs / parent["official_score"]
    median_abs_sd = parent["official_score"] * sd_score_pooled / 100.0

    promotion = {
        "parent_receipt": parent["short"],
        "parent_official_score": parent["official_score"],
        "tier0_best_receipt": tier_best_score["short"],
        "tier0_best_solver": tier_best_score["solver"],
        "tier0_best_official_score": tier_best_score["official_score"],
        "frontier_gap_abs": gap_abs,
        "frontier_gap_pct": gap_pct,
        "tier0_candidate_rel_sd_pct": sd_cand_tier0,
        "tier0_score_rel_sd_pct": sd_score_tier0,
        "gap_is_within_median_noise": gap_pct < 2.0 * sd_score_tier0,
        "predicted_score_floor": lifted(parent["official_score"], pred_lo / 100.0),
        "predicted_score_central": lifted(
            parent["official_score"], pred_central / 100.0
        ),
        "predicted_score_ceiling": lifted(parent["official_score"], pred_hi / 100.0),
        "published_median_abs_sd": median_abs_sd,
        "z_of_predicted_score_over_tier0_best": (
            lifted(parent["official_score"], pred_central / 100.0)
            - tier_best_score["official_score"]
        )
        / (median_abs_sd * math.sqrt(2.0)),
        "note": (
            "The tier-0 score spread is median noise, not speed: within-tier "
            "candidate decode time varies by "
            f"{sd_cand_tier0:.4f} % while the published median varies by "
            f"{sd_score_tier0:.4f} %. The top promoted row is therefore not "
            "demonstrably faster than our parent. A promotion decision on one "
            "receipt is close to a coin flip even if the arm delivers the "
            "central prediction, so promotion is not the readout."
        ),
    }

    art = {
        "harness": "ranked",
        "experiment": "E130 rung 7",
        "submission": CANDIDATE_SUBMISSION,
        "arm": "prune_na5_pair",
        "registered_before_receipt": True,
        "coupling_model": {
            "equation": "saving_pct = s_head * g * c * 100",
            "g_residency": G_RESIDENCY,
            "g_source": "rung3 census e130_ranked_g17s_f47_weighted_residency_pct",
            "s_head_grid": list(S_HEAD_GRID),
            "s_head_source": "FINDING 143, ledger 280, four independent lines",
            "c_bracket": [C_GRID_LO, C_GRID_HI],
            "c_source": "F8 working bracket",
            "grid": grid,
            "predicted_candidate_saving_pct": {
                "floor": pred_lo,
                "central": pred_central,
                "ceiling": pred_hi,
            },
        },
        "instrument": {
            "candidate_rel_sd_pct_pooled": sd_cand_pooled,
            "candidate_rel_sd_pct_tier0": sd_cand_tier0,
            "score_rel_sd_pct_pooled": sd_score_pooled,
            "score_rel_sd_pct_tier0": sd_score_tier0,
            "source": "research/e130-artifacts/rung6-instrument-sd.json",
        },
        "tier0": {
            "n": len(members),
            "candidate_mean_spt": tier0["candidate_mean_spt"],
            "members": members,
        },
        "noise_model": noise,
        "rank_diagnostics": rank_diagnostics(family),
        "power": power,
        "decision_rule": decision_rule,
        "promotion_arithmetic": promotion,
        "caveats": [
            "The arm's residency gain is a static resident_simdgroups_derived "
            "floor-law quantity. Rung 5 retracted the occupancy probe, so no "
            "validated local measurement of c exists. This receipt is the first "
            "measurement of c on the ranked harness.",
            "F121 shows case 5 is never entered on the scored target path, so "
            "the arm deletes zero executed instructions. Any effect must come "
            "from residency alone.",
            "Single-receipt readout. The pooled sd is the conservative "
            "instrument, the tier-0 sd is optimistic with only 3 dof.",
        ],
    }

    with open(args.out, "w") as fh:
        json.dump(art, fh, indent=2)
        fh.write("\n")

    print(f"wrote {args.out}")
    print(f"predicted candidate saving: {pred_lo:.4f} % .. {pred_hi:.4f} % "
          f"(central {pred_central:.4f} %)")
    print(f"candidate-spt sigma (conservative): {sig_cand:.4f} %  -> "
          f"z floor {power['candidate_spt']['z_at_prediction_floor']:.2f}, "
          f"z central {power['candidate_spt']['z_at_prediction_central']:.2f}")
    print(f"published-median sigma:            {sig_median:.4f} %  -> "
          f"z floor {power['published_median']['z_at_prediction_floor']:.2f}, "
          f"z central {power['published_median']['z_at_prediction_central']:.2f}")
    print(f"2 sigma null threshold: {two_sigma:.4f} %")
    print(f"frontier gap parent -> tier0 best: {promotion['frontier_gap_pct']:.4f} % "
          f"(within median noise: {promotion['gap_is_within_median_noise']})")
    print(f"predicted score band: {promotion['predicted_score_floor']:.6f} .. "
          f"{promotion['predicted_score_ceiling']:.6f} vs tier0 best "
          f"{promotion['tier0_best_official_score']:.6f}")
    rd = art["rank_diagnostics"]
    pc = rd["positive_control_across_family"]
    print(f"rank control across family (n={pc['n']}): rho = "
          f"{pc['spearman_spt_vs_score']:+.4f}")
    for t in rd["within_tier"]:
        print(f"  tier {t['rank']} n={t['n']}: spt range {t['candidate_spt_range_pct']:.4f} % "
              f"score range {t['score_range_pct']:.4f} % rho = "
              f"{t['spearman_spt_vs_score']:+.4f}")


if __name__ == "__main__":
    main()
