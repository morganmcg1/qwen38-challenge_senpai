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
    evaluate,
    position_acceptance,
    shifted_pair,
)

ARMS = [
    ("I", "research/trace-runI-base-cap8-512.log", 3, 2.0947033499),
    ("J", "research/trace-runJ-gate2-512.log", 2, 2.1243836568),
    ("K", "research/trace-runK-gate2-cap8-512.log", 2, 2.1019606601),
    ("L", "research/trace-runL-gate1-cap8-512.log", 1, 2.1288141130),
    ("M", "research/trace-runM-gate0-cap8-512.log", 0, 2.0600336024),
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
    except FileNotFoundError:
        continue
    q = pooled["pooled_acceptance"]
    ps0, ws0 = clean(sh)
    pd0, wd0 = clean(dp)
    ps, pd = shifted_pair(ps0, ws0, pd0, wd0, q)
    model = evaluate(gate, ps, pd, False)
    out.append(
        {
            "arm": label,
            "gate": gate,
            "q_pooled_conditional": round(q, 6),
            "pooled_shallow_cap": sh["pooled_acceptance"],
            "pooled_deep_cap": dp["pooled_acceptance"],
            "model_raw": round(model["raw"], 6),
            "measured_raw": measured_raw,
            "model_over_measured": round(model["raw"] / measured_raw, 6),
            "model_mean_tokens_per_round": round(model["accepted_tokens_per_round"], 4),
            "model_ms_per_token": round(model["ms_per_token"], 4),
            "forward_raw_by_gate": {
                str(g): round(evaluate(g, ps, pd, False)["raw"], 6)
                for g in (-1, 3, 2, 1, 0)
            },
        }
    )

for row in out:
    print(json.dumps(row))
with open("research/occupancy-validation.json", "w") as fh:
    json.dump({"serial_s_per_token": SERIAL_S_PER_TOKEN, "arms": out}, fh, indent=1)
print("wrote research/occupancy-validation.json")
