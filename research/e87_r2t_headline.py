#!/usr/bin/env python3
"""E87 r2 rung 2: the score-relevant headline for one balanced timing session.

usage: research/e87_r2t_headline.py [--paired research/e87-paired-r2t.json]
                                    [--report research/e87-headline-r2t.json]

`research/e87_paired.py` prices the DECODE ROUNDS. The ranked score prices the
WHOLE charged leg, seed prefill included, so a round-level percentage overstates
the score effect by the prefill's share of the window. This script converts one
into the other and states both.

Ranked boundary, from senpai/verify-ranked-score-boundary.sh: the ranked serial
numerator comes from the runner's own prebuilt baseline workspace, so no
candidate edit can move it. A candidate-time reduction of `x` therefore raises
every affected ranked raw_p by 1/(1-x) - 1. Nothing local is subtracted.

The published score is (raw_beagle + raw_essays) / 2. Those two prompts run
different mean draft counts, and this mechanism is priced per draft, so the
report restates the local leg-total gain at each of those draft counts.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from e87_depth_sensitivity import fit as depth_fit
from e87_depth_sensitivity import round_gain_pct
from e87_paired import OUT, rounds, score_metric

# Per-draft head read of each arm, from the byte census in the assignment.
HEAD_BYTES = {
    "declared": 427_738_112,
    "derived25": 329_402_112,
    "derived15": 313_670_912,
}
BYTES_TO_SCORE_PCT = 0.0815  # 1 % of declared-head bytes -> % of candidate s/token

# Mean drafts per round on the two prompts the published score actually reads,
# and on one unscored prompt kept as a spread control.
SCORED_DRAFT_COUNTS = {"beagle": 4.382, "essays": 5.087, "republic": 4.989}
SCORED_PROMPTS = ("beagle", "essays")


def leg_accounting(tag: str) -> dict:
    """Split one leg into the decode rounds and everything else it is charged."""
    rs = rounds(tag)
    spt = score_metric(tag, "mtp_seconds_per_token")
    tokens = score_metric(tag, "decode_tokens")
    leg_s = spt * tokens
    round_s = sum(r["round_us"] for r in rs) / 1e6
    return {
        "clean": [r["clean"] for r in rs],
        "leg_seconds": leg_s,
        "round_seconds": round_s,
        "nonround_seconds": leg_s - round_s,
        "decode_share": round_s / leg_s,
    }


def scored_prompt_price(
    prefix: str,
    base_arm: str,
    arm: str,
    accepted_rate: float,
    local_drafts: float,
    nonround_us_per_token: float,
    measured_leg_gain_pct: float,
    leg_gain_stderr_pct: float | None,
) -> dict:
    """Restate the measured leg-total gain at the two scored prompts' depths.

    The saving is paid per draft, so it scales with drafts per round. Two things
    change together when the draft count changes: the per-round gain, and the
    decode share of the charged window, because a shallower round emits fewer
    tokens and therefore spends more decode time per token against a fixed seed
    prefill. The depth fit supplies the first and a token-rate model supplies
    the second. Both are then divided out at this session's own draft count, so
    the local fixture reproduces its measured leg-total gain exactly and only
    the RATIO between depths is taken from the model.
    """
    model = depth_fit(prefix, base_arm, arm)
    b = model["base_round_us_fit"]

    def decode_share(drafts: float) -> float:
        round_us = b["intercept_us"] + b["slope_us_per_draft"] * drafts
        per_token = round_us / (drafts * accepted_rate + 1.0)
        return per_token / (per_token + nonround_us_per_token)

    reference = round_gain_pct(model, local_drafts) * decode_share(local_drafts)
    calibration = measured_leg_gain_pct / reference

    prompts = {}
    for name, drafts in SCORED_DRAFT_COUNTS.items():
        leg_gain = round_gain_pct(model, drafts) * decode_share(drafts) * calibration
        prompts[name] = {
            "drafts_per_round": drafts,
            "modelled_round_gain_pct": round_gain_pct(model, drafts),
            "modelled_decode_share": decode_share(drafts),
            "leg_total_gain_pct": leg_gain,
            "ranked_raw_p_gain_pct": (1.0 / (1.0 - leg_gain / 100.0) - 1.0) * 100.0,
            "in_published_score": name in SCORED_PROMPTS,
            "extrapolated_below_observed_depth": drafts < model["drafts_observed_min"],
        }

    scored = [prompts[n]["ranked_raw_p_gain_pct"] for n in SCORED_PROMPTS]
    return {
        "model": "depth fit x token-rate decode share, calibrated to this session",
        "local_mean_drafts_per_round": local_drafts,
        "accepted_draft_rate": accepted_rate,
        "nonround_us_per_token": nonround_us_per_token,
        "calibration_factor": calibration,
        "depth_fit": model,
        "prompts": prompts,
        # score = (raw_beagle + raw_essays) / 2, so an equal-weight mean of the
        # two raw gains is exact when the two raw ratios are close.
        "published_score_gain_pct": st.mean(scored),
        "scored_prompt_spread_pp": max(scored) - min(scored),
        "leg_total_gain_stderr_pp": leg_gain_stderr_pct,
        "spread_within_one_stderr":
            (max(scored) - min(scored)) < leg_gain_stderr_pct
            if leg_gain_stderr_pct else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="research/e87-paired-r2t.json")
    ap.add_argument("--base", default="declared")
    ap.add_argument("--report", default="research/e87-headline-r2t.json")
    args = ap.parse_args()

    doc = json.loads(Path(args.paired).read_text())
    stratum = doc["per_leg_host_stratum"]
    by_arm: dict[str, list] = {}
    for s in stratum:
        by_arm.setdefault(s["arm"], []).append(s)

    base = [s["mtp_seconds_per_token"] for s in by_arm[args.base]]
    base_mean, base_med = st.mean(base), st.median(base)
    account = {s["tag"]: leg_accounting(s["tag"]) for s in stratum}
    flags = {tag: a["clean"] for tag, a in account.items()}
    n_rounds = min(len(v) for v in flags.values())
    tokens = score_metric(stratum[0]["tag"], "decode_tokens")

    arms = {}
    for arm, group in by_arm.items():
        v = [s["mtp_seconds_per_token"] for s in group]
        entry = [float(s["gpu_temp_entry_c"]) for s in group if s["gpu_temp_entry_c"]]
        arms[arm] = {
            "legs": sorted(v),
            "mean_seconds_per_token": st.mean(v),
            "median_seconds_per_token": st.median(v),
            "stdev_us": st.stdev(v) * 1e6 if len(v) > 1 else 0.0,
            "mean_delta_pct": (st.mean(v) / base_mean - 1.0) * 100.0,
            "median_delta_pct": (st.median(v) / base_med - 1.0) * 100.0,
            "positions": doc["abba_position"][arm]["positions"],
            "position_sum": doc["abba_position"][arm]["position_sum"],
            "entry_temp_spread_c": max(entry) - min(entry) if entry else None,
            "head_bytes_per_draft": HEAD_BYTES.get(arm),
            "mean_leg_seconds": st.mean(account[s["tag"]]["leg_seconds"] for s in group),
            "mean_round_seconds": st.mean(account[s["tag"]]["round_seconds"] for s in group),
            "mean_nonround_seconds":
                st.mean(account[s["tag"]]["nonround_seconds"] for s in group),
            "decode_share_of_window":
                st.mean(account[s["tag"]]["decode_share"] for s in group),
        }
        # 1 s.e. of this arm's mean, as a percentage of the base arm's mean.
        arms[arm]["mean_stderr_pct"] = (
            100.0 * st.stdev(v) / len(v) ** 0.5 / base_mean if len(v) > 1 else None)

    census, score = {}, {}
    for arm, group in by_arm.items():
        if arm == args.base:
            continue
        a = [flags[s["tag"]] for s in group]
        b = [flags[s["tag"]] for s in by_arm[args.base]]
        census[arm] = {
            "rounds_compared": n_rounds,
            # What the median-of-legs paired estimator can actually use.
            "usable_at_least_one_clean_leg_per_arm": sum(
                1 for i in range(n_rounds)
                if any(l[i] for l in b) and any(l[i] for l in a)),
            # The strict reading, for comparison with the 5/78 composed-tree case.
            "clean_in_every_leg_of_both_arms": sum(
                1 for i in range(n_rounds)
                if all(l[i] for l in b) and all(l[i] for l in a)),
        }
        removed = HEAD_BYTES[args.base] - HEAD_BYTES[arm]
        removed_pct = 100.0 * removed / HEAD_BYTES[args.base]
        # Sign convention: `gain` is positive when the candidate got faster.
        gain_leg = -arms[arm]["mean_delta_pct"]
        gain_round = -doc["paired"][arm]["round_us"]["median_pct"]
        score[arm] = {
            "head_bytes_removed": removed,
            "head_bytes_removed_pct": removed_pct,
            "price_list_predicted_pct": removed_pct * BYTES_TO_SCORE_PCT,
            "measured_leg_total_gain_pct": gain_leg,
            "measured_round_only_gain_pct": gain_round,
            # The prefill is charged but the head cannot touch it, so it dilutes
            # the round-level effect by exactly its share of the window.
            "implied_decode_share_of_window": gain_leg / gain_round if gain_round else None,
            "ranked_raw_p_gain_pct": (1.0 / (1.0 - gain_leg / 100.0) - 1.0) * 100.0,
            "submit2_per_draft_delta_us":
                doc["paired"][arm]["submit2_per_draft_us"]["median_delta_us"],
            "submit2_per_draft_sign_test":
                f"{doc['paired'][arm]['submit2_per_draft_us']['sign_test_arm_faster']}"
                f"/{doc['paired'][arm]['submit2_per_draft_us']['sign_test_n']}",
            "round_seconds_delta":
                arms[arm]["mean_round_seconds"] - arms[args.base]["mean_round_seconds"],
            "nonround_seconds_delta":
                arms[arm]["mean_nonround_seconds"] - arms[args.base]["mean_nonround_seconds"],
            "leg_seconds_delta":
                arms[arm]["mean_leg_seconds"] - arms[args.base]["mean_leg_seconds"],
            "leg_total_gain_stderr_pct": (
                (arms[arm]["mean_stderr_pct"] ** 2
                 + arms[args.base]["mean_stderr_pct"] ** 2) ** 0.5
                if arms[arm]["mean_stderr_pct"] and arms[args.base]["mean_stderr_pct"]
                else None),
        }
        score[arm]["scored_prompt_price"] = scored_prompt_price(
            prefix=doc["prefix"],
            base_arm=args.base,
            arm=arm,
            accepted_rate=by_arm[args.base][0]["accepted_draft_rate"],
            local_drafts=by_arm[args.base][0]["effective_mean_draft_len"],
            nonround_us_per_token=1e6 * arms[args.base]["mean_nonround_seconds"] / tokens,
            measured_leg_gain_pct=gain_leg,
            leg_gain_stderr_pct=score[arm]["leg_total_gain_stderr_pct"],
        )

    report = {
        "experiment": "e87-coarse-draft-shortlist-traffic",
        "prefix": doc["prefix"],
        "harness": "local",
        "sandbox": doc["sandbox"],
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "base_arm": args.base,
        "session_null": doc["session_null"],
        "depth_sequence_identical_across_arms":
            doc["depth_sequence_identical_across_arms"],
        "arms": arms,
        "clean_round_census": census,
        "score_model": score,
    }
    print(json.dumps(report, indent=2))
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
