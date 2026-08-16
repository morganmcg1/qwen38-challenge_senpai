#!/usr/bin/env python3
"""Check the fb2 occupancy model against the measured arms.

The model's q axis is pooled *conditional* per-position acceptance, not the
trace's accepted/drafted ratio, so the calibration point has to be read off the
same estimator the model uses."""
import json
import sys

sys.path.insert(0, "research")
from occupancy_model import (  # noqa: E402
    SERIAL_S_PER_TOKEN,
    deep_cap_from_trace,
    evaluate,
    position_acceptance,
    shifted_pair,
)

# label, trace, gate, measured raw. The deep cap is read from the trace.
ARMS = [
    ("I", "research/trace-runI-base-cap8-512.log", 3, 2.0947033499),
    ("J", "research/trace-runJ-cap7-512.log", 3, 2.1636873696),
    ("J2", "research/trace-runJ-gate2-512.log", 2, 2.1243836568),
    ("K", "research/trace-runK-gate2-cap8-512.log", 2, 2.1019606601),
    ("L", "research/trace-runL-gate1-cap8-512.log", 1, 2.1288141130),
    ("M", "research/trace-runM-gate0-cap8-512.log", 0, 2.0600336024),
    ("N", "research/trace-runN-gate1-cap8-512-confirm.log", 1, 2.1311965111),
    ("O", "research/trace-runO-cap7-gate3-512.log", 3, 2.1615420490),
    ("P", "research/trace-runP-cap7-gate3-512-confirm.log", 3, 2.1623244386),
]


def clean(stats):
    return (
        [
            min(max(x, 0.05), 0.995) if x is not None else 0.9
            for x in stats["acceptance"]
        ],
        [max(w, 1) for w in stats["observations"]],
    )


out = []
for label, trace, gate, measured_raw in ARMS:
    try:
        pooled = position_acceptance(trace)
        sh = position_acceptance(trace, only_deep=False)
        dp = position_acceptance(trace, only_deep=True)
        deep_cap = deep_cap_from_trace(trace)
    except FileNotFoundError:
        continue
    q = pooled["pooled_acceptance"]
    ps0, ws0 = clean(sh)
    pd0, wd0 = clean(dp)
    ps, pd = shifted_pair(ps0, ws0, pd0, wd0, q)
    model = evaluate(gate, ps, pd, False, deep_cap)
    model_cap8 = evaluate(gate, ps, pd, False, 8)
    out.append(
        {
            "arm": label,
            "gate": gate,
            "deep_cap": deep_cap,
            "q_pooled_conditional": round(q, 6),
            "pooled_shallow_cap": sh["pooled_acceptance"],
            "pooled_deep_cap": dp["pooled_acceptance"],
            "model_raw": round(model["raw"], 6),
            "measured_raw": measured_raw,
            "model_over_measured": round(model["raw"] / measured_raw, 6),
            "model_raw_forced_cap8": round(model_cap8["raw"], 6),
            "model_over_measured_forced_cap8": round(
                model_cap8["raw"] / measured_raw, 6
            ),
            "model_depth_by_state": model["depth_by_state"],
            "model_mean_effective_depth": round(model["mean_effective_depth"], 4),
            "model_mean_tokens_per_round": round(model["accepted_tokens_per_round"], 4),
            "model_ms_per_token": round(model["ms_per_token"], 4),
            "forward_raw_by_gate": {
                str(g): round(evaluate(g, ps, pd, False, deep_cap)["raw"], 6)
                for g in (-1, 3, 2, 1, 0)
            },
            "forward_raw_by_cap_at_own_gate": {
                str(c): round(evaluate(gate, ps, pd, False, c)["raw"], 6)
                for c in (5, 6, 7, 8)
            },
        }
    )

for row in out:
    print(json.dumps(row))
with open("research/occupancy-validation.json", "w") as fh:
    json.dump({"serial_s_per_token": SERIAL_S_PER_TOKEN, "arms": out}, fh, indent=1)
print("wrote research/occupancy-validation.json")
