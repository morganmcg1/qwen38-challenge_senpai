#!/usr/bin/env python3
"""Retract the 30.402 ms refutation and settle what the two constants measure.

RETRACTION. An earlier revision of this script concluded that the 30.402 ms
ranked depth-0 round of campaign-ledger.md:10453 was refuted, because it
implies a 16.09 s leg that no pinned serial sample on the Yukon board reaches.
The arithmetic is right and the population is wrong. 30.402 ms is not a pinned
serial round at all. It is the CANDIDATE build's depth-0 round, calibrated in
`research/prompt_round_reconstruction.py` from `row["mtp_spt"]`, the candidate
seconds per token. Comparing it against the pinned serial leg population
compares two different builds. The implied 16.09 s leg is real and correct: it
is the leg our own candidate would run on plutarch if every round were
depth-0, and the measured plutarch candidate leg is 15.517 s, 3.6 % below it,
the gap being plutarch's 38 drafting rounds.

The measured source exists and reproduces:

    python3 research/prompt_round_reconstruction.py \
      --facts research/e53-board-facts.json --submission ca9251b8 \
      --prefill-ms 526.6

`calibrate_depth0_ms()` at :111-156 reads `row["mtp_spt"]` at :134, anchors on
the prompt with the largest non-drafting share, and resolves one unknown by
fixed point. The same script emits an exact score decomposition that closes to
about 1e-11 on all eight prompts.

RESOLUTION. Both constants are correct and they measure different builds:

    pinned serial depth-0 round   36.958 ms   (three routes, 3768 board legs)
    candidate depth-0 round       30.402 ms   (reconstruction, ca9251b8)
    ratio                          1.2156

A model-free ceiling settles it without any host transfer model. plutarch runs
487 rounds, 449 of them depth-0, at a mean round of 30.781 ms. A drafting
round cannot be cheaper than a depth-0 round, so c1 <= 30.781 ms. 30.402 sits
1.2 % under that ceiling; the pinned serial 36.958 sits 20.1 % over it.
R(depth-0) = 65.009 / 30.402 = 2.1383 therefore STANDS.

SURVIVING DEFECT. 188(A) still defines R on legs ("Let L be leg time") and
then computes it from two ROUND times. That is a real definitional defect and
it is independent of which round constant is used.

SUPERSEDED BY. R is not one number. It is width dependent. This script derives
R(M) live from the same measured source and reports it as the pricing table.

harness=ranked for every number produced here. No GPU runs.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from prompt_round_reconstruction import (  # noqa: E402
    E1_DEPTH0_ROUND_MS_M4,
    WINDOW_TOKENS,
    cumulative_ms_m4,
    load_rows,
    reconstruct,
)

BOARD = pathlib.Path("research/ranked_stream_ab_board.json")
FACTS = "research/e53-board-facts.json"
SUBMISSION = "ca9251b8"
OUT = pathlib.Path("research/e70-transfer-constant.json")

# --- the constants under audit -------------------------------------------
LEDGER_CANDIDATE_ROUND_MS = 30.402       # campaign-ledger.md:10453
LEDGER_R = 2.1383                        # campaign-ledger.md:11045
RECEIPT_BEAGLE_SERIAL_SPT = 0.0379848885  # ca9251b8 public receipt
RECEIPT_BEAGLE_PREFILL_SPT = 0.001027553  # ca9251b8 public receipt

RANKED_PREFILL_MS = 526.6        # ledger 186(B) :10399, mid of 525.7-528.2
LEDGER_K_MS_RANGE = (525.7, 528.2)
LOCAL_PREFILL_S = 3.9938         # ledger 186(C) :10450
RANKED_PREFILL_S = 0.5269        # ledger 186(C) :10450


def board_serial_spt(board: dict) -> tuple[list[float], int, list[str]]:
    values: list[float] = []
    prompts: list[str] = []
    trees = board["trees"]
    for tree in trees.values():
        for prompt, spt in tree.get("ser", {}).items():
            values.append(spt)
            if prompt not in prompts:
                prompts.append(prompt)
    return values, len(trees), sorted(prompts)


def local_round_ms(width: float) -> float:
    """E1's local M4 Pro round cost at verify width M, interpolated on depth."""
    return E1_DEPTH0_ROUND_MS_M4 + cumulative_ms_m4(width - 1.0)


def main() -> None:
    board = json.loads(BOARD.read_text())
    serial_spt, tree_count, prompt_keys = board_serial_spt(board)
    n = len(serial_spt)
    mean_spt = statistics.fmean(serial_spt)
    sd_spt = statistics.stdev(serial_spt)
    lo_spt, hi_spt = min(serial_spt), max(serial_spt)
    ranked_serial_leg_s = WINDOW_TOKENS * mean_spt

    # Three independent routes to the PINNED SERIAL depth-0 round.
    k_lo, k_hi = LEDGER_K_MS_RANGE
    serial_round_board_hi = (ranked_serial_leg_s * 1000.0 - k_lo) / WINDOW_TOKENS
    serial_round_board_lo = (ranked_serial_leg_s * 1000.0 - k_hi) / WINDOW_TOKENS
    serial_round_receipt = (
        RECEIPT_BEAGLE_SERIAL_SPT - RECEIPT_BEAGLE_PREFILL_SPT) * 1000.0
    serial_routes = [
        serial_round_board_lo, serial_round_board_hi, serial_round_receipt]
    serial_round_ms = statistics.fmean(serial_routes)
    serial_route_spread_pct = 100.0 * (
        max(serial_routes) - min(serial_routes)) / serial_round_ms

    # The candidate-build side, from the measured reconstruction.
    rows = load_rows(FACTS, SUBMISSION)
    rec = reconstruct(rows, prefill_ms=RANKED_PREFILL_MS)
    c1 = rec["calibration"]["depth0_round_ms"]
    anchor = rec["calibration"]["anchor_prompt"]
    anchor_row = rec["prompts"][anchor]

    # Model-free ceiling: no round can be cheaper than a depth-0 round, so the
    # anchor's MEAN round is a hard upper bound on the depth-0 round.
    ceiling_ms = anchor_row["observed_round_ms"]

    # The retracted test, reproduced so the error is inspectable.
    implied_spt = (LEDGER_CANDIDATE_ROUND_MS + k_lo / WINDOW_TOKENS) / 1000.0
    implied_leg_s = WINDOW_TOKENS * implied_spt
    measured_anchor_leg_s = WINDOW_TOKENS * rows[anchor]["mtp_spt"]

    # R(M): the width-dependent local-to-ranked round ratio.
    width_table = []
    for name, p in rec["prompts"].items():
        width = 1.0 + p["proposed_drafts"] / p["rounds"]
        loc = local_round_ms(width)
        ranked = p["observed_round_ms"]
        width_table.append({
            "prompt": name,
            "verify_width_M": width,
            "rounds_at_M": p["rounds"],
            "accepted_drafts": p["accepted_drafts"],
            "tokens_per_round": 1.0 + p["accepted_drafts"] / p["rounds"],
            "rounds_plus_accepted": p["rounds"] + p["accepted_drafts"],
            "local_round_ms": loc,
            "ranked_round_ms": ranked,
            "R_of_M": loc / ranked,
            "ranked_candidate_leg_ms": WINDOW_TOKENS * rows[name]["mtp_spt"] * 1000.0,
        })
    width_table.sort(key=lambda r: r["verify_width_M"])
    depth0_entry = {
        "prompt": "(depth-0 control)",
        "verify_width_M": 1.0,
        "rounds_at_M": WINDOW_TOKENS,
        "accepted_drafts": 0,
        "tokens_per_round": 1.0,
        "rounds_plus_accepted": WINDOW_TOKENS,
        "local_round_ms": E1_DEPTH0_ROUND_MS_M4,
        "ranked_round_ms": c1,
        "R_of_M": E1_DEPTH0_ROUND_MS_M4 / c1,
        "ranked_candidate_leg_ms": None,
    }
    low_group = [r for r in width_table if r["verify_width_M"] < 4.0]
    high_group = [r for r in width_table if r["verify_width_M"] >= 4.0]
    low_mean = statistics.fmean(r["R_of_M"] for r in low_group)
    high_mean = statistics.fmean(r["R_of_M"] for r in high_group)
    low_spread = max(r["R_of_M"] for r in low_group) - min(
        r["R_of_M"] for r in low_group)
    high_spread = max(r["R_of_M"] for r in high_group) - min(
        r["R_of_M"] for r in high_group)

    report = {
        "harness": "ranked",
        "what": (
            "retraction of the 30.402 ms refutation, and the resolution that "
            "the two contested constants measure two different builds"),
        "retraction": {
            "earlier_claim": (
                "30.402 ms is refuted because it implies a 16.09 s pinned "
                "serial leg that no board sample reaches"),
            "why_it_fails": (
                "right arithmetic, wrong population. 30.402 ms is the "
                "CANDIDATE build's depth-0 round, calibrated from "
                "row['mtp_spt'] in prompt_round_reconstruction.py:134. It was "
                "compared against the PINNED SERIAL leg population."),
            "implied_leg_s": implied_leg_s,
            "measured_anchor_candidate_leg_s": measured_anchor_leg_s,
            "implied_vs_measured_pct": 100.0 * (
                measured_anchor_leg_s - implied_leg_s) / implied_leg_s,
            "implied_leg_is_real": (
                "the implied leg is the leg our candidate would run on "
                f"{anchor} if every round were depth-0. The measured "
                f"{anchor} candidate leg is {measured_anchor_leg_s:.3f} s, "
                "below it by exactly the cost its "
                f"{anchor_row['drafting_rounds']} drafting rounds add."),
            "status": "RETRACTED",
        },
        "measured_source": {
            "script": "research/prompt_round_reconstruction.py",
            "function": "calibrate_depth0_ms",
            "lines": "111-156",
            "reads": "row['mtp_spt'] at :134",
            "reproduce": (
                "python3 research/prompt_round_reconstruction.py --facts "
                f"{FACTS} --submission {SUBMISSION} --prefill-ms "
                f"{RANKED_PREFILL_MS}"),
            "anchor_prompt": anchor,
            "anchor_rounds": rec["calibration"]["anchor_rounds"],
            "anchor_non_drafting_share": rec["calibration"][
                "anchor_non_drafting_share"],
            "host_scale_vs_m4pro": rec["calibration"]["host_scale_vs_m4pro"],
            "recomputed_depth0_round_ms": c1,
            "ledger_10453_value": LEDGER_CANDIDATE_ROUND_MS,
            "recomputed_vs_ledger_pct": 100.0 * (
                c1 - LEDGER_CANDIDATE_ROUND_MS) / LEDGER_CANDIDATE_ROUND_MS,
        },
        "model_free_ceiling": {
            "argument": (
                "a drafting round cannot be cheaper than a depth-0 round, so "
                "the anchor's MEAN round is a hard upper bound on the "
                "candidate depth-0 round. No host transfer model is used."),
            "anchor_prompt": anchor,
            "anchor_rounds": anchor_row["rounds"],
            "anchor_non_drafting_rounds": anchor_row["non_drafting_rounds"],
            "anchor_mean_round_ms": ceiling_ms,
            "ceiling_ms": ceiling_ms,
            "candidate_c1_ms": c1,
            "candidate_margin_under_ceiling_pct": 100.0 * (
                ceiling_ms - c1) / ceiling_ms,
            "pinned_serial_round_ms": serial_round_ms,
            "pinned_serial_excess_over_ceiling_pct": 100.0 * (
                serial_round_ms - ceiling_ms) / ceiling_ms,
            "verdict": (
                "the candidate constant fits under the ceiling and the pinned "
                "serial constant cannot. R(depth-0) = 65.009 / 30.402 = "
                "2.1383 stands."),
        },
        "two_builds": {
            "pinned_serial_depth0_round_ms": serial_round_ms,
            "pinned_serial_routes": {
                "board_mean_minus_K_high": serial_round_board_lo,
                "board_mean_minus_K_low": serial_round_board_hi,
                "beagle_receipt_spt_minus_prefill_spt": serial_round_receipt,
                "route_spread_pct": serial_route_spread_pct,
            },
            "candidate_depth0_round_ms": c1,
            "serial_over_candidate": serial_round_ms / c1,
            "reading": (
                "both constants are correct. The pinned serial build runs a "
                f"{serial_round_ms:.3f} ms depth-0 round; our candidate build "
                f"runs a {c1:.3f} ms depth-0 round. The build factor "
                f"{serial_round_ms / c1:.4f} is the uniform speedup the "
                "reconstruction's exact score decomposition also reports."),
        },
        "board_evidence": {
            "source": BOARD.name,
            "submissions": tree_count,
            "prompt_keys": prompt_keys,
            "serial_spt_samples": n,
            "serial_spt_mean_s": mean_spt,
            "serial_spt_sd_s": sd_spt,
            "serial_spt_sd_pct": 100.0 * sd_spt / mean_spt,
            "serial_spt_min_s": lo_spt,
            "serial_spt_max_s": hi_spt,
            "ranked_serial_leg_s": ranked_serial_leg_s,
        },
        "R_of_M": {
            "definition": (
                "R(M) = local M4 Pro round cost at verify width M / ranked "
                "round cost at the same M. Local cost is E1's depth-0 round "
                "plus its marginal ladder, interpolated at fractional depth. "
                "Ranked cost is the reconstruction's observed round."),
            "width_definition": "M = 1 + proposed_drafts / rounds",
            "rounds_definition": "tokens_per_round = 1 + accepted / rounds",
            "table": [depth0_entry] + width_table,
            "low_width_group_mean": low_mean,
            "low_width_group_spread": low_spread,
            "high_width_group_mean": high_mean,
            "high_width_group_spread": high_spread,
            "step_pct_group_means": 100.0 * (high_mean - low_mean) / low_mean,
            "step_pct_shelf_edges": 100.0 * (
                min(r["R_of_M"] for r in high_group)
                / max(r["R_of_M"] for r in low_group) - 1.0),
            "within_group_scatter_pct": {
                "low": 100.0 * low_spread / low_mean,
                "high": 100.0 * high_spread / high_mean,
            },
            "shelf_boundary": (
                "flat at 2.11-2.17 for M <= 3.66, flat at 2.36-2.47 for "
                "M >= 5.53. The step sits where the local E1 ladder jumps "
                "from E1(4) = 91.29 ms to E1(5) = 115.69 ms, +26.7 %."),
            "caveats": [
                "ranked rounds are means over a width MIXTURE while local E1 "
                "is a fixed width, which biases R(M) downward for prompts "
                "with high width spread",
                "the public receipt gives no ranked round census, so the "
                "mixture cannot be deconvolved",
            ],
            "caveat_direction": (
                "the mixture caveat cuts against the step, so the step is "
                "more likely real than not"),
        },
        "pricing_rule": {
            "delta_ranked_ms": "delta_local_ms / R(M)",
            "delta_score_pct": (
                "100 * delta_ranked_ms * rounds_at_M / "
                "ranked_candidate_leg_ms"),
            "reporting_requirement": (
                "report R(M) and M beside every converted number. An "
                "unlabelled conversion is invalid."),
            "tau_prefill": LOCAL_PREFILL_S / RANKED_PREFILL_S,
            "tau_prefill_note": (
                "prefill transfers at its own constant and is unaffected by "
                "this retraction"),
        },
        "surviving_defect": {
            "item": "188(A)",
            "defect": (
                "R is defined on legs ('Let L be leg time') and then computed "
                "from two ROUND times. The defect is independent of which "
                "round constant is used."),
            "status": "confirmed, advisor is correcting the ledger text",
        },
        "leg_ratio_is_not_R": {
            "argument": (
                "a candidate-side saving moves only the score denominator, so "
                "delta_score_pct = 100 * delta_ranked_ms * rounds_at_M / "
                "ranked_CANDIDATE_leg. The normalizer is the candidate leg."),
            "local_candidate_spt_runs": {
                "score-runL-gate1-cap8-512": 0.034562689485028386,
                "score-runK-gate2-cap8-512": 0.03502187505364418,
                "score-runN-gate1-cap8-512-confirm": 0.03458425961434841,
            },
            "indicative_candidate_leg_ratio_beagle": (
                WINDOW_TOKENS * 0.034562689485028386 / 6.233),
            "why_not_adopted": (
                "a leg ratio compares two different width MIXTURES, so it is "
                "not comparable with R(M) and must not be promoted to a "
                "pricing constant"),
            "recommendation": (
                "retire the single R. Use R(M) for the conversion and the "
                "direct form for the score, which needs no leg ratio."),
        },
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
