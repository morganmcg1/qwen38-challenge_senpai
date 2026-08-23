#!/usr/bin/env python3
"""E143 F2 re-pricing: the live crown anchor, and C-d as an interval.

F2 asks for four things R1 did not supply.

1. Every median figure re-anchored on the live promoted crown `1760479a`
   instead of F1's `572b2cc4`, through the order-statistic model.
2. C-d as a STATED UNCERTAINTY, not a point value, because the closure
   decision sits at the 80 % line and a point estimate cannot support it.
3. `e143_unresolved_fraction` reported beside C-d every time, because rows
   inside the 0.107-logit offline-versus-device gap belong to no channel and
   the C-d estimate inherits their ambiguity.
4. The per-prompt gain SHAPE of the mechanism, not only its beagle magnitude,
   because the upper median slot is the minimum of essays, republic, medicine
   and botany, and a gain that lifts the wrong prompt can lose score.

C-d is computed here as the measured residual identity

    C-d = miss - C-a - C-b - C-c

rather than from arm F's simulated classification. Once C-b is measured at
zero and C-c is zero by proof, the identity is exact against the measured miss
count, while the surrogate only reproduces the miss rate to its calibration
residual. R1 reported the surrogate count; this file reports the identity and
the interval around it.

Usage: research/e143_f2.py [--out research/e143-f2.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import e143_value

MISS_TO_SCORE_PCT = 203.0
R0 = Path("research/e143-r0.json")
R1 = Path("research/e143-r1.json")
# Rule of three at 95 % for 0 events in 2,634 trials, from R1.
CB_RULE_OF_THREE_95 = 3.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="research/e143-f2.json")
    args = parser.parse_args()

    r0 = json.loads(R0.read_text())
    r1 = json.loads(R1.read_text())
    stage_b = r0["stage_b"]
    carriers = r1["per_carrier"]

    # ---- C-d by the measured residual identity, with an interval ----------
    channel_d = {}
    for name, entry in carriers.items():
        trials = entry["trials"]
        misses = round(entry["measured_miss_rate"] * trials)
        ca = entry["ca"]["events"]
        divergence_rows = stage_b[name]["n"]
        unresolved = round(stage_b[name]["unresolved_fraction"] * divergence_rows)
        in_vocab = misses - ca
        # Point: every resolved in-vocabulary miss is C-d, because C-b measured
        # zero and C-c is zero by proof.
        resolved_in_vocab = in_vocab - unresolved
        # Low: give C-b every unresolved row AND its 95 % rule-of-three bound
        # scaled to this carrier's share of the pooled trials.
        cb_bound = CB_RULE_OF_THREE_95 * trials / sum(
            c["trials"] for c in carriers.values())
        low_events = max(0.0, resolved_in_vocab - cb_bound)
        # High: every unresolved row is C-d.
        high_events = in_vocab
        channel_d[name] = {
            "trials": trials,
            "measured_misses": misses,
            "channel_a_events": ca,
            "unresolved_divergence_rows": unresolved,
            "unresolved_fraction": stage_b[name]["unresolved_fraction"],
            "events_point": resolved_in_vocab,
            "events_low": low_events,
            "events_high": high_events,
            "rate_point": resolved_in_vocab / trials,
            "rate_low": low_events / trials,
            "rate_high": high_events / trials,
            "share_of_divergences_point": resolved_in_vocab / misses,
            "share_of_divergences_low": low_events / misses,
            "share_of_divergences_high": high_events / misses,
            "share_of_in_vocabulary_point":
                resolved_in_vocab / in_vocab if in_vocab else float("nan"),
            "share_of_in_vocabulary_low":
                low_events / in_vocab if in_vocab else float("nan"),
            "raw_ratio_pct_point": MISS_TO_SCORE_PCT * resolved_in_vocab / trials,
            "raw_ratio_pct_low": MISS_TO_SCORE_PCT * low_events / trials,
            "raw_ratio_pct_high": MISS_TO_SCORE_PCT * high_events / trials,
        }

    pooled_trials = sum(c["trials"] for c in carriers.values())
    pooled_misses = sum(channel_d[n]["measured_misses"] for n in channel_d)
    pooled_point = sum(channel_d[n]["events_point"] for n in channel_d)
    pooled_low = sum(channel_d[n]["events_low"] for n in channel_d)
    pooled_high = sum(channel_d[n]["events_high"] for n in channel_d)
    kill_line = 0.80
    pooled = {
        "trials": pooled_trials,
        "misses": pooled_misses,
        "share_of_divergences_point": pooled_point / pooled_misses,
        "share_of_divergences_low": pooled_low / pooled_misses,
        "share_of_divergences_high": pooled_high / pooled_misses,
        "kill_line": kill_line,
        "closes_axis_at_point": pooled_point / pooled_misses > kill_line,
        "closes_axis_at_worst_case": pooled_low / pooled_misses > kill_line,
        "note": ("pooled only for the closure decision, which is a property of "
                 "the mechanism and not of a regime; every price below is "
                 "per carrier"),
    }

    # ---- the primary, re-anchored on the live crown -----------------------
    beagle_pct = r1["primary"]["e143_reachable_acceptance_pct_beagle"]
    beagle_ceiling_x, beagle_ceiling_value = e143_value.ceiling("beagle")
    essays_ceiling_x, essays_ceiling_value = e143_value.ceiling("essays")

    def median_for(gain: dict[str, float]) -> float:
        return e143_value.median_pct_gain(gain)

    # C-a is the only reachable channel and it is beagle-only in this capture:
    # 5 events on beagle, 0 on essays, 2 on prompts that convert at exactly
    # zero. That is the best possible shape on this benchmark.
    shape = {
        "beagle": carriers["beagle"]["ca"]["raw_ratio_pct"] / 100.0,
        "essays": carriers["essays"]["ca"]["raw_ratio_pct"] / 100.0,
    }
    primary = {
        "anchor": "1760479a",
        "anchor_median": e143_value.median_of(e143_value.ANCHOR_1760479a),
        "e143_reachable_acceptance_pct_beagle": beagle_pct,
        "e143_reachable_acceptance_pct_beagle_ci68": [
            carriers["beagle"]["ca"]["raw_ratio_pct_ci68"][0],
            carriers["beagle"]["ca"]["raw_ratio_pct_ci68"][1]],
        "median_pct_at_1760479a": median_for(shape),
        "median_pct_at_1760479a_ci68": [
            median_for({"beagle":
                        carriers["beagle"]["ca"]["raw_ratio_pct_ci68"][0] / 100.0}),
            median_for({"beagle":
                        carriers["beagle"]["ca"]["raw_ratio_pct_ci68"][1] / 100.0})],
        "exceeds_beagle_saturation": beagle_pct > beagle_ceiling_x,
        "beagle_ceiling_x_pct": beagle_ceiling_x,
        "beagle_ceiling_value_pct": beagle_ceiling_value,
        "essays_ceiling_x_pct": essays_ceiling_x,
        "essays_ceiling_value_pct": essays_ceiling_value,
        "units": "percent of the beagle RAW RATIO, as F1 section 3 asked",
    }

    # ---- gain shape, as F2 section 5 asks ---------------------------------
    gain_shape = {
        "mechanism": "close C-a, the structurally unproposable target token",
        "beagle_pct": carriers["beagle"]["ca"]["raw_ratio_pct"],
        "essays_pct": carriers["essays"]["ca"]["raw_ratio_pct"],
        "republic_pct": 0.0,
        "medicine_pct": 0.0,
        "botany_pct": 0.0,
        "evidence": ("5 unproposable draft-trial events on 884 beagle trials, "
                     "0 on 429 essays trials, 2 on 1,321 trials of prompts "
                     "whose marginal value is exactly zero"),
        "essays_upper_68_pct": carriers["essays"]["ca"]["raw_ratio_pct_ci68"][1],
        "shape_label": "beagle-only",
        "reading": ("beagle-only is the best possible shape on this benchmark: "
                    "it converts at 0.4793 per percent with 9.67 % of runway, "
                    "and it cannot push the upper median slot into another "
                    "prompt because it does not move the upper slot at all"),
        "upper_slot_buffers_pct": e143_value.upper_slot_buffers(),
    }

    # C-d's ceiling under the order statistic, so the closure decision is
    # priced on what it forfeits rather than on a linear extrapolation.
    cd_shape = {k: channel_d[k]["rate_point"] * MISS_TO_SCORE_PCT / 100.0
                for k in ("beagle", "essays")}
    cd_shape_low = {k: channel_d[k]["rate_low"] * MISS_TO_SCORE_PCT / 100.0
                    for k in ("beagle", "essays")}
    cd_shape_high = {k: channel_d[k]["rate_high"] * MISS_TO_SCORE_PCT / 100.0
                     for k in ("beagle", "essays")}
    cd_value = {
        "beagle_raw_pct_point": channel_d["beagle"]["raw_ratio_pct_point"],
        "essays_raw_pct_point": channel_d["essays"]["raw_ratio_pct_point"],
        "median_pct_point": median_for(cd_shape),
        "median_pct_low": median_for(cd_shape_low),
        "median_pct_high": median_for(cd_shape_high),
        "naive_rule_116_median_pct": (
            0.4793 * channel_d["beagle"]["raw_ratio_pct_point"]
            + 0.5207 * channel_d["essays"]["raw_ratio_pct_point"]),
        "reading": ("the unreachable ceiling. Rule 121 saturates it; the "
                    "linear Rule 116 conversion does not and overstates it"),
    }

    state = {
        "harness": "local",
        "anchor": "1760479a",
        "anchor_note": ("F2 replaces F1's 572b2cc4 anchor with the live "
                        "promoted crown; research/e143_value.py asserts every "
                        "figure of both"),
        "primary": primary,
        "gain_shape": gain_shape,
        "channel_d": channel_d,
        "channel_d_pooled": pooled,
        "channel_d_value": cd_value,
        "e143_unresolved_fraction": stage_b["unresolved_fraction_all"],
        "e143_unresolved_fraction_by_carrier": {
            n: stage_b[n]["unresolved_fraction"] for n in channel_d},
        "channel_b_rule_of_three_95_events": CB_RULE_OF_THREE_95,
    }
    Path(args.out).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
