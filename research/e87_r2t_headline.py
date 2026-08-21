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
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
HOST_PHASES = ("d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
               "d_chain_us", "readout_us", "commit_us", "upkeep_us")
HOST_GATE_US = 1500.0
FIELD = re.compile(r"(\w+)=([-\d.]+)")

# Per-draft head read of each arm, from the byte census in the assignment.
HEAD_BYTES = {
    "declared": 427_738_112,
    "derived25": 329_402_112,
    "derived15": 313_670_912,
}
BYTES_TO_SCORE_PCT = 0.0815  # 1 % of declared-head bytes -> % of candidate s/token


def clean_flags(tag: str) -> list[bool]:
    out = []
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        if line.startswith("mtp-trace: round="):
            rec = {k: float(v) for k, v in FIELD.findall(line.split(" arm=")[0])}
            out.append(sum(rec[p] for p in HOST_PHASES) < HOST_GATE_US)
    return out


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
    flags = {s["tag"]: clean_flags(s["tag"]) for s in stratum}
    n_rounds = min(len(v) for v in flags.values())

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
        }

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
        }

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
