"""E204 stage 3 decision: does the round-anchored gain reach the whole leg?

RULE 397 makes the whole-leg absolute candidate `mtp_seconds_per_token` the
promotion-grade statistic for a mechanism that can move work across the round
anchors. This reducer reads the four gated `--local-submit` legs of the ABBA
session, applies the advisor's predeclared bands, and asserts that the four
legs share one trajectory.

Bands, on mean ON minus mean OFF as a fraction of mean OFF:
  ON faster by >= 0.2% and both pair deltas agree in sign -> leg-level winner
  |delta| < 0.1%, or the pair deltas disagree in sign     -> relocation, no
                                                             end-to-end value
  otherwise                                                -> unclear
"""

import json
import sys

WINNER_BAND = 0.002
NULL_BAND = 0.001
FIDELITY_KEYS = ["effective_mean_draft_len", "accepted_draft_rate",
                 "all_tokens_matched", "residual_divergence_count",
                 "decode_tokens", "mtp_depth"]


def load(tag):
    metrics = json.load(open(f"research/out/{tag}/score.json"))["metrics"]
    arm = tag.rsplit("-", 1)[1]
    return {"tag": tag, "arm": arm,
            "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
            "serial_seconds_per_token": metrics["serial_seconds_per_token"],
            "ratio": metrics["mtp_decode_speedup"],
            "fidelity": {k: metrics[k] for k in FIDELITY_KEYS}}


def mean(xs):
    return sum(xs) / len(xs)


def main(tags, out_path):
    legs = [load(t) for t in tags]
    off = [l for l in legs if l["arm"] == "off"]
    on = [l for l in legs if l["arm"] == "on"]

    off_mtp, on_mtp = mean([l["mtp_seconds_per_token"] for l in off]), \
        mean([l["mtp_seconds_per_token"] for l in on])
    delta = on_mtp - off_mtp
    rel = delta / off_mtp

    # ABBA pairs itself: leg 1 with leg 2, leg 4 with leg 3, so each pair spans
    # the same part of the session's thermal ramp in opposite arm order.
    pairs = [(legs[0], legs[1]), (legs[3], legs[2])]
    pair_deltas = [b["mtp_seconds_per_token"] - a["mtp_seconds_per_token"]
                   for a, b in pairs]
    same_sign = all(d > 0 for d in pair_deltas) or all(d < 0
                                                       for d in pair_deltas)

    off_serial = mean([l["serial_seconds_per_token"] for l in off])
    on_serial = mean([l["serial_seconds_per_token"] for l in on])

    on_faster_by = -rel
    if on_faster_by >= WINNER_BAND and same_sign:
        band = "leg-level winner"
    elif abs(rel) < NULL_BAND or not same_sign:
        band = "relocation, no end-to-end value"
    else:
        band = "unclear"

    fidelity = [l["fidelity"] for l in legs]
    record = {
        "artifact": "stage3-leg-level-abba",
        "experiment": "e204-round-end-seam-overlap",
        "harness": "local",
        "statistic": "whole-leg absolute mtp_seconds_per_token (RULE 397)",
        "order": [l["arm"] for l in legs],
        "legs": legs,
        "off_mean_mtp_seconds_per_token": off_mtp,
        "on_mean_mtp_seconds_per_token": on_mtp,
        "delta_on_minus_off": delta,
        "delta_fraction_of_off": rel,
        "on_faster_by_fraction": on_faster_by,
        "pair_deltas_on_minus_off": pair_deltas,
        "pair_deltas_agree_in_sign": same_sign,
        "off_pair_spread_fraction": abs(
            off[0]["mtp_seconds_per_token"]
            - off[1]["mtp_seconds_per_token"]) / off_mtp,
        "on_pair_spread_fraction": abs(
            on[0]["mtp_seconds_per_token"]
            - on[1]["mtp_seconds_per_token"]) / on_mtp,
        "serial_control": {
            "off_mean": off_serial, "on_mean": on_serial,
            "delta_fraction": (on_serial - off_serial) / off_serial,
        },
        "fidelity_identical_across_legs": all(f == fidelity[0]
                                              for f in fidelity),
        "fidelity": fidelity[0],
        "band": band,
        "promote": band == "leg-level winner",
    }
    json.dump(record, open(out_path, "w"), indent=1)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
