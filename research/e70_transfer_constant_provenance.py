#!/usr/bin/env python3
"""Settle the provenance of the ranked depth-0 decode round constant.

The campaign carries two values for the ranked depth-0 serial decode round:

  30.402 ms   senpai/campaign-ledger.md:10453, no derivation stated,
              and :11045 turns it into R = 65.009 / 30.402 = 2.1383
  36.957 ms   derived by the advisor from the public receipt of submission
              ca9251b8, prompt beagle:
              serial_seconds_per_token - prefill_seconds_per_token

`R = 2.1383` is load-bearing: :11059, :11060, :11184, :11637, :11664, :11731
and the /3.55 divisor of 186(D) all build on it.

This script decides between them from ranked evidence that neither value was
fitted to: `research/ranked_stream_ab_board.json`, the distilled Yukon board
export, which holds the pinned serial seconds per token for every submission
and prompt on the board.

harness=ranked for every number produced here. No GPU runs.
"""

from __future__ import annotations

import json
import pathlib
import statistics

BOARD = pathlib.Path("research/ranked_stream_ab_board.json")
OUT = pathlib.Path("research/e70-transfer-constant.json")

DECODE_TOKENS = 512

# --- the two contested ranked constants ---------------------------------
LEDGER_RANKED_ROUND_MS = 30.402          # campaign-ledger.md:10453
LEDGER_R = 2.1383                        # campaign-ledger.md:11045
RECEIPT_BEAGLE_SERIAL_SPT = 0.0379848885  # ca9251b8 public receipt
RECEIPT_BEAGLE_PREFILL_SPT = 0.001027553  # ca9251b8 public receipt

# --- the uncontested measurements the constants are compared against -----
# Measured local pinned serial legs at the same 512-token window, so the leg
# ratio does not have to be reconstructed from a round time and a prefill time.
LOCAL_SERIAL_SPT_RUNS = {
    "score-runL-gate1-cap8-512": 0.07357754115946591,
    "score-runK-gate2-cap8-512": 0.07361460360698402,
    "score-runN-gate1-cap8-512-confirm": 0.07370585342869163,
    "analysis-runP-512-confirm": 0.07359492778778076,
}
LOCAL_DEPTH0_ROUND_MS = 65.0094  # E1, ledger :14313, N=1530, sd 0.16 %
LOCAL_PREFILL_S = 3.9938         # ledger 186(C) :10450
RANKED_PREFILL_S = 0.5269        # ledger 186(C) :10450
# 186(B) :10399: "K = 512 * prefill_seconds_per_token ... is a near-constant
# 525.7-528.2 ms (spread 0.46 %)".
LEDGER_K_MS_RANGE = (525.7, 528.2)
# 186(B) :10432: "K / serial_leg is 2.70-2.72 % on every prompt, as it must be
# when the serial legs are all about 19.4 s."
LEDGER_K_OVER_LEG_RANGE = (2.70, 2.72)


def collect_serial_spt(board: dict) -> tuple[list[float], int, list[str]]:
    values: list[float] = []
    prompts: list[str] = []
    trees = board["trees"]
    for tree in trees.values():
        for prompt, spt in tree.get("ser", {}).items():
            values.append(spt)
            if prompt not in prompts:
                prompts.append(prompt)
    return values, len(trees), sorted(prompts)


def main() -> None:
    board = json.loads(BOARD.read_text())
    serial_spt, tree_count, prompts = collect_serial_spt(board)
    n = len(serial_spt)
    mean_spt = statistics.fmean(serial_spt)
    sd_spt = statistics.stdev(serial_spt)
    lo_spt, hi_spt = min(serial_spt), max(serial_spt)

    # The trusted driver charges the seed prefill to the decode measurement
    # (186(A), QwenRuntimeMTP.swift:347-349), so
    #   serial_leg = 512 * serial_seconds_per_token
    #   serial_decode_round = serial_seconds_per_token - prefill_seconds_per_token
    ranked_leg_s = DECODE_TOKENS * mean_spt
    ranked_leg_lo_s = DECODE_TOKENS * lo_spt
    ranked_leg_hi_s = DECODE_TOKENS * hi_spt

    k_lo, k_hi = LEDGER_K_MS_RANGE
    round_from_board_hi = (ranked_leg_s * 1000.0 - k_lo) / DECODE_TOKENS
    round_from_board_lo = (ranked_leg_s * 1000.0 - k_hi) / DECODE_TOKENS
    round_from_receipt = (
        RECEIPT_BEAGLE_SERIAL_SPT - RECEIPT_BEAGLE_PREFILL_SPT) * 1000.0

    # Falsification of 30.402: what would the pinned serial baseline have to
    # measure for that round to be real?
    implied_spt = (
        LEDGER_RANKED_ROUND_MS + LEDGER_K_MS_RANGE[0] / DECODE_TOKENS) / 1000.0
    implied_leg_s = DECODE_TOKENS * implied_spt
    implied_k_over_leg_pct = 100.0 * (
        LEDGER_K_MS_RANGE[0] / 1000.0) / implied_leg_s
    board_values_below_implied = sum(1 for v in serial_spt if v <= implied_spt)
    shortfall_pct = 100.0 * (mean_spt - implied_spt) / mean_spt

    # The corrected transfer constants.
    tau_prefill = LOCAL_PREFILL_S / RANKED_PREFILL_S
    tau_round = LOCAL_DEPTH0_ROUND_MS / round_from_receipt
    local_leg_s = LOCAL_PREFILL_S + DECODE_TOKENS * LOCAL_DEPTH0_ROUND_MS / 1000.0
    r_leg = local_leg_s / ranked_leg_s
    measured_local_leg_s = statistics.fmean(LOCAL_SERIAL_SPT_RUNS.values()) * DECODE_TOKENS
    r_leg_measured = measured_local_leg_s / ranked_leg_s

    report = {
        "harness": "ranked",
        "what": (
            "provenance decision between the 30.402 ms ledger constant and "
            "the 36.957 ms receipt derivation for the ranked depth-0 serial "
            "decode round"),
        "board_evidence": {
            "source": BOARD.name,
            "submissions": tree_count,
            "prompt_keys": prompts,
            "serial_spt_samples": n,
            "serial_spt_mean_s": mean_spt,
            "serial_spt_sd_s": sd_spt,
            "serial_spt_sd_pct": 100.0 * sd_spt / mean_spt,
            "serial_spt_min_s": lo_spt,
            "serial_spt_max_s": hi_spt,
            "ranked_serial_leg_s": ranked_leg_s,
            "ranked_serial_leg_min_s": ranked_leg_lo_s,
            "ranked_serial_leg_max_s": ranked_leg_hi_s,
        },
        "ranked_depth0_serial_round_ms": {
            "from_board_and_ledger_K_low": round_from_board_lo,
            "from_board_and_ledger_K_high": round_from_board_hi,
            "from_beagle_receipt": round_from_receipt,
            "ledger_10453_claim": LEDGER_RANKED_ROUND_MS,
        },
        "falsification_of_30_402": {
            "implied_serial_spt_s": implied_spt,
            "implied_serial_leg_s": implied_leg_s,
            "implied_K_over_leg_pct": implied_k_over_leg_pct,
            "ledger_observed_K_over_leg_pct": list(LEDGER_K_OVER_LEG_RANGE),
            "board_serial_spt_samples_at_or_below_implied":
                board_values_below_implied,
            "board_serial_spt_samples_total": n,
            "shortfall_vs_board_mean_pct": shortfall_pct,
            "verdict": (
                "refuted: 30.402 ms requires a pinned serial leg of "
                f"{implied_leg_s:.3f} s, which no board sample reaches, and "
                f"it puts K/leg at {implied_k_over_leg_pct:.2f} % against the "
                "2.70-2.72 % the ledger itself observes in the same item"),
        },
        "corrected_transfer_constants": {
            "tau_prefill": tau_prefill,
            "tau_depth0_round": tau_round,
            "R_leg_local_over_ranked": r_leg,
            "local_serial_leg_s": local_leg_s,
            "R_leg_from_measured_local_legs": r_leg_measured,
            "measured_local_serial_leg_s": measured_local_leg_s,
            "measured_local_serial_leg_runs": LOCAL_SERIAL_SPT_RUNS,
            "ledger_R": LEDGER_R,
            "R_change_pct": 100.0 * (r_leg - LEDGER_R) / LEDGER_R,
            "R_change_pct_measured": 100.0 * (r_leg_measured - LEDGER_R) / LEDGER_R,
            "prefill_over_round_transfer_contrast": tau_prefill / tau_round,
            "ledger_transfer_contrast": 3.55,
            "arithmetic_bound_multiplier_R_over_tau": r_leg / tau_prefill,
            "arithmetic_bound_divisor": tau_prefill / r_leg,
            "ledger_arithmetic_bound_divisor": 3.55,
            "latency_bound_multiplier": r_leg,
            "ledger_latency_bound_multiplier": LEDGER_R,
        },
        "note": (
            "188(A) defines R as a LEG ratio ('Let L be leg time') but then "
            "computes it from two ROUND times. Even with a correct ranked "
            "round the two are not equal, because prefill transfers at 7.58x "
            "and decode at 1.76x. The leg ratio is computed here from the "
            "board mean and the local leg, and it is a third number again."),
        "which_leg_score_arithmetic_needs": {
            "argument": (
                "score = ranked serial spt / ranked candidate spt, and a "
                "candidate-side saving moves only the denominator, so "
                "delta_score_pct = 100 * (delta_local / tau) / "
                "ranked_CANDIDATE_leg. The normalizer is the candidate leg, "
                "not the pinned serial leg. R as used in 188(A) is therefore "
                "neither the serial-leg ratio computed above nor the depth-0 "
                "round ratio."),
            "ranked_candidate_legs_ms_186B": {
                "plutarch": 15517, "drama": 10126, "travel": 8903,
                "beagle": 6233, "medicine": 5821, "republic": 5726,
                "essays": 5764, "botany": 5673,
            },
            "local_candidate_spt_runs": {
                "score-runL-gate1-cap8-512": 0.034562689485028386,
                "score-runK-gate2-cap8-512": 0.03502187505364418,
                "score-runN-gate1-cap8-512-confirm": 0.03458425961434841,
            },
            "indicative_candidate_leg_ratio_beagle": (
                512 * 0.034562689485028386 / 6.233),
            "caveat": (
                "indicative only: the local runs above and the ranked legs of "
                "ca9251b8 are not proven to be the same candidate schedule, "
                "so this ratio is not a constant to adopt"),
            "recommendation": (
                "retire R. delta_score_pct = 100 * (delta_local / tau) / "
                "ranked_candidate_leg needs no leg ratio at all and cannot "
                "pick up the wrong leg."),
        },
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
