#!/usr/bin/env python3
"""Pre-registered competing predictions for Run M (segmentedStreakGate = 0).

Two models disagree about the endpoint of the assigned 3 -> 2 -> 1 -> 0 sweep.

occupancy_model.py keys the acceptance profile off the depth a round actually
runs at, so when the gate opens every round it also grants every round the
*deep* profile. That is the fixed-selection-gap caveat taken to its limit: at
gate 0 there is no selection left to justify the gap.

The two-state model below keys the profile off the streak state instead, which
is what the code actually branches on, so a post-reject round keeps its
measured poor acceptance when it is forced deep.
"""
import json

C_LOCAL_MS = {4: 127.073, 8: 217.805}
GAP_S = 4.016569256782532
SERIAL_DECODE_S = 37.671701073646545
RUN_L_BLOCKS_S = 13.679527759552002
RUN_L_ROUNDS = 74
RUN_L_SHALLOW_ROUNDS = 18
Q_SHALLOW = 0.864406779661017
Q_DEEP = 0.9772727272727273


def tokens(q, depth):
    return sum(q**i for i in range(depth + 1))


def steady(depth_post_reject, depth_other):
    p_rej_post = 1.0 - Q_SHALLOW**depth_post_reject
    p_rej_other = 1.0 - Q_DEEP**depth_other
    f = p_rej_other / (1.0 - p_rej_post + p_rej_other)
    tok = f * tokens(Q_SHALLOW, depth_post_reject) + (1 - f) * tokens(Q_DEEP, depth_other)
    cost = f * C_LOCAL_MS[depth_post_reject] + (1 - f) * C_LOCAL_MS[depth_other]
    return f, tok, cost, cost / tok


g1 = steady(4, 8)
g0 = steady(8, 8)
calib = RUN_L_BLOCKS_S * 1000.0 / 512.0 / g1[3]
blocks_g0 = g0[3] * calib * 512.0 / 1000.0
ratio_g0 = SERIAL_DECODE_S / (blocks_g0 + GAP_S)

out = {
    "pre_registered_for": "runM-gate0-cap8-512",
    "two_state_model": {
        "keys_profile_on": "streak state (what the code branches on)",
        "gate1": {
            "post_reject_round_share": round(g1[0], 4),
            "tokens_per_round": round(g1[1], 3),
            "ms_per_round": round(g1[2], 2),
            "ms_per_token": round(g1[3], 3),
        },
        "gate0": {
            "post_reject_round_share": round(g0[0], 4),
            "tokens_per_round": round(g0[1], 3),
            "ms_per_round": round(g0[2], 2),
            "ms_per_token": round(g0[3], 3),
        },
        "validation_gate1_share_measured": round(RUN_L_SHALLOW_ROUNDS / RUN_L_ROUNDS, 4),
        "validation_gate1_ms_per_token_measured": round(
            RUN_L_BLOCKS_S * 1000.0 / 512.0, 3
        ),
        "block_ms_per_token_change_pct": round(100.0 * (g0[3] / g1[3] - 1.0), 2),
        "predicted_raw": round(ratio_g0, 4),
        "predicted_raw_change_vs_runL_pct": round(
            100.0 * (ratio_g0 / 2.128814113 - 1.0), 2
        ),
    },
    "occupancy_model_L_calibrated": {
        "keys_profile_on": "depth actually run (grants deep profile to every round)",
        "predicted_raw": 2.17076,
        "predicted_raw_change_vs_runL_pct": round(
            100.0 * (2.17076 / 2.128814113 - 1.0), 2
        ),
    },
    "runL_measured_raw": 2.128814113,
    "decision_rule": (
        "Run M settles which conditioning is right. If raw >= 2.17 the occupancy "
        "model is vindicated and the streak gate should be deleted. If raw <= 2.05 "
        "the two-state model is right, the gate earns its keep purely by keeping "
        "post-reject rounds shallow, and gate 1 is the recommended setting."
    ),
}
print(json.dumps(out, indent=1))
with open("research/gate0-preregistration.json", "w") as handle:
    json.dump(out, handle, indent=1)
