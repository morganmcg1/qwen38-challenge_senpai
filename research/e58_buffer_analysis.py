#!/usr/bin/env python3
"""Turn the counterbalanced command-buffer session into an end-to-end answer.

The session varies only MLX_MAX_OPS_PER_BUFFER at a fixed 512 MiB referenced-byte
budget, which is the ranked geometry. Command-buffer packing is measured
separately by the census, so the slope here is time against buffers per round,
not against dispatches per round.

usage:
  research/e58_buffer_analysis.py [SESSION_TAG] [--json OUT]
"""

from __future__ import annotations

import argparse
import json

RANKED = {
    "beagle": {"leg_ms": 6233.1, "rounds": 107, "ms_per_round": 53.33, "dilution": 0.91552},
    "medicine": {"leg_ms": 5820.7, "rounds": 99, "ms_per_round": 53.48, "dilution": 0.90953},
}
# RETRACTED by ledger 193(E): this is 2 sd of the SERIAL leg's jitter applied to the
# score, and the median over eight prompts does not average the candidate-leg common
# mode away. The measured single-pair ranked MDE is 2.10 %, 7.4x larger. The value
# below is kept so this module's published arithmetic stays reproducible; import
# research/ranked_noise.py for any NEW ranked pricing.
RANKED_MDE_PERCENT = 0.283
# RETRACTED by ledger 198(G), confirmed by 202. 0.0629 is ONE adjacent-leg
# same-arm spread, not a null floor. The local null is not monotone in leg
# separation and it is host- and session-specific; measured same-arm spreads run
# to 0.2835 %. Kept so this module's published arithmetic stays reproducible.
# For a NEW decision, take the largest same-arm spread inside your own session.
LOCAL_NULL_FLOOR_PERCENT = 0.0629

DECODE_TOKENS = 512
CANDIDATE_ROUNDS = 76
CANDIDATE_DISPATCHES_PER_ROUND = 1048.62
SERIAL_DISPATCHES_PER_ROUND = 1705.41

# Census-measured packing at each arm's setting, 64-token probe runs. Counts per
# round are limit-independent, so buffers per round follow directly.
PACKING = {
    "50": {"candidate_per_buffer": 27.22, "serial_per_buffer": 31.48},
    "256": {"candidate_per_buffer": 48.39, "serial_per_buffer": 121.89},
}

ARMS = {"a": "50", "b": "256"}


def load(session: str, name: str) -> dict:
    with open(f"research/out/{session}-{name}/score.json", encoding="utf-8") as f:
        return json.load(f)["metrics"]


def meta(session: str, name: str) -> dict:
    out: dict[str, str] = {}
    with open(f"research/out/{session}-{name}/meta.txt", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", default="e58-bufops")
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    names = ["a1", "a2", "b1", "b2"]
    metrics = {n: load(args.session, n) for n in names}
    metas = {n: meta(args.session, n) for n in names}

    result: dict = {
        "session": args.session,
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "arms": {},
    }

    for letter, ops in ARMS.items():
        runs = [f"{letter}1", f"{letter}2"]
        mtp = [metrics[r]["mtp_seconds_per_token"] for r in runs]
        ser = [metrics[r]["serial_seconds_per_token"] for r in runs]
        buffers = CANDIDATE_DISPATCHES_PER_ROUND / PACKING[ops]["candidate_per_buffer"]
        result["arms"][ops] = {
            "runs": runs,
            "ops_per_buffer_limit": int(ops),
            "mtp_seconds_per_token": sum(mtp) / 2,
            "mtp_within_pair_spread_percent": 100 * abs(mtp[0] - mtp[1]) / (sum(mtp) / 2),
            "serial_seconds_per_token": sum(ser) / 2,
            "serial_within_pair_spread_percent": 100 * abs(ser[0] - ser[1]) / (sum(ser) / 2),
            "candidate_dispatches_per_buffer": PACKING[ops]["candidate_per_buffer"],
            "candidate_buffers_per_round": buffers,
            "all_tokens_matched": [bool(metrics[r]["all_tokens_matched"]) for r in runs],
            "effective_mean_draft_len": [metrics[r]["effective_mean_draft_len"] for r in runs],
            "residual_divergence_count": [metrics[r].get("residual_divergence_count") for r in runs],
            "gpu_temp_entry_c": [float(metas[r]["gpu_temp_entry_c"]) for r in runs],
            "gpu_temp_exit_c": [float(metas[r]["gpu_temp_exit_c"]) for r in runs],
        }

    base, wide = result["arms"]["50"], result["arms"]["256"]
    tokens_per_round = DECODE_TOKENS / CANDIDATE_ROUNDS
    delta_s_per_token = wide["mtp_seconds_per_token"] - base["mtp_seconds_per_token"]
    delta_ms_per_round = delta_s_per_token * tokens_per_round * 1000
    removed_buffers = base["candidate_buffers_per_round"] - wide["candidate_buffers_per_round"]

    entry_temps = [t for a in result["arms"].values() for t in a["gpu_temp_entry_c"]]
    result["effect"] = {
        "candidate_delta_percent": 100 * delta_s_per_token / base["mtp_seconds_per_token"],
        "candidate_delta_ms_per_round": delta_ms_per_round,
        "serial_delta_percent": 100
        * (wide["serial_seconds_per_token"] - base["serial_seconds_per_token"])
        / base["serial_seconds_per_token"],
        "buffers_removed_per_round": removed_buffers,
        "ns_per_removed_buffer": -delta_ms_per_round * 1e6 / removed_buffers if removed_buffers else None,
        "entry_temp_spread_c": max(entry_temps) - min(entry_temps),
        "local_null_floor_percent": LOCAL_NULL_FLOOR_PERCENT,
    }

    for prompt, r in RANKED.items():
        saving_ms = -delta_ms_per_round
        pct_round = 100 * saving_ms / r["ms_per_round"]
        result["effect"][f"{prompt}_percent_of_ranked_round"] = pct_round
        result["effect"][f"{prompt}_percent_of_score"] = pct_round * r["dilution"]

    result["effect"]["ranked_mde_percent"] = RANKED_MDE_PERCENT
    result["effect"]["clears_ranked_mde"] = (
        result["effect"]["beagle_percent_of_score"] > RANKED_MDE_PERCENT
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
