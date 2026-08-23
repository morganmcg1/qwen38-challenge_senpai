#!/usr/bin/env python3
"""E143 C2: read the ABBA island-arm session and price it.

Reports the two metrics the assignment names for the C2 fork:

  e143_c2_realised_acceptance_delta_pp   measured LIVE, arm q minus arm all
  e143_c2_ranked_pct                     from measured seconds per token

and the exactness gate `e143_exactness_divergences`, which must be 0 on every
leg of both arms.

ABBA, so the arm contrast is `mean(legs 2,3) - mean(legs 1,4)` and monotone
thermal drift cancels to first order. The within-arm pair also gives a
replicate spread, which is the only honest noise floor available from one
session.

Usage: research/e143_c2_read.py [--prefix e143c2] [--out research/e143-c2.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import e143_value

OUT = Path("research/out")


def witness_is_arm(witness: str | None, arm: str) -> bool:
    return (witness or "").startswith(f"qwen-mtp-island-arm: {arm} ")


def read_meta(directory: Path) -> dict:
    meta = {}
    for line in (directory / "meta.txt").read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def read_rounds(directory: Path) -> list[dict]:
    path = directory / "trace.txt"
    if not path.is_file():
        return []
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("mtp-trace: round="):
            rounds.append({k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)})
    return rounds


def read_passes(directory: Path) -> list[dict[int, int]]:
    """position -> top-1 token id, one map per pass.

    One leg appends the reference, serial and timed passes to the same trace
    file, so the passes must be split at their `begin` markers before any two
    legs are compared position by position.
    """
    path = directory / "trace.txt"
    if not path.is_file():
        return []
    passes: list[dict[int, int]] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("mtp-trace: begin "):
            passes.append({})
            continue
        m = re.match(r"^mtp-row: pos=(\d+) ids=(-?\d+),(-?\d+)", line)
        if m and passes:
            passes[-1][int(m[1])] = int(m[2])
    return [p for p in passes if p]


def leg(tag: str) -> dict:
    directory = OUT / tag
    score = json.loads((directory / "score.json").read_text())
    metrics = score["metrics"]
    meta = read_meta(directory)
    rounds = read_rounds(directory)
    entry = {
        "tag": tag,
        "arm": meta.get("e124_arm"),
        "arm_witness": meta.get("e124_arm_witness"),
        "passed": bool(score.get("passed")),
        "decode_tokens": metrics["decode_tokens"],
        "all_tokens_matched": bool(metrics["all_tokens_matched"]),
        "residual_divergence_count": metrics["residual_divergence_count"],
        "effective_mean_draft_len": metrics["effective_mean_draft_len"],
        "accepted_draft_rate": metrics["accepted_draft_rate"],
        "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "serial_seconds_per_token": metrics["serial_seconds_per_token"],
        "mtp_decode_speedup": metrics["mtp_decode_speedup"],
        "head_provenance_sha256": metrics["head_provenance_sha256"],
        "rounds": len(rounds),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "official_or_ranked_score": meta.get("official_or_ranked_score"),
        "worker_sha256": meta.get("worker_sha256"),
        "base_sha": meta.get("base_sha"),
        "dirty_candidate_paths": meta.get("dirty_candidate_paths"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
    }
    if rounds:
        accepted = sum(r["acc"] for r in rounds)
        proposed = sum(r["d"] for r in rounds)
        entry["trace_accepted"] = accepted
        entry["trace_proposed"] = proposed
        # Per-STEP acceptance. A round that accepted its whole draft ran no
        # failing trial, so only a short round contributes a miss.
        misses = sum(1 for r in rounds if r["acc"] < r["d"])
        trials = accepted + misses
        entry["per_step_p"] = accepted / trials if trials else float("nan")
        entry["per_step_trials"] = trials
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="e143c2")
    parser.add_argument("--out", default="research/e143-c2.json")
    args = parser.parse_args()

    order = [f"{args.prefix}-1-all", f"{args.prefix}-2-q",
             f"{args.prefix}-3-q", f"{args.prefix}-4-all"]
    legs = [leg(tag) for tag in order]

    for entry in legs:
        expected = entry["tag"].rsplit("-", 1)[-1]
        if entry["arm"] != expected:
            raise SystemExit(f"e143-c2: {entry['tag']} recorded arm {entry['arm']}")
        if not witness_is_arm(entry["arm_witness"], expected):
            raise SystemExit(
                f"e143-c2: {entry['tag']} has no Rule 114 witness for arm "
                f"{expected}: {entry['arm_witness']!r}")

    # Rule 101 positive control. The witness reading is only informative if it
    # can come out negative, so run the same predicate against the arm the leg
    # did NOT request and require every leg to fail it. An inert selector makes
    # all four legs report `all`, which fails this control on legs 2 and 3.
    control = {
        e["tag"]: witness_is_arm(e["arm_witness"],
                                 "q" if e["arm"] == "all" else "all")
        for e in legs
    }
    control_passed = not any(control.values())
    witness_arms = sorted({e["arm"] for e in legs})

    # Exactness gate.
    divergences = sum(e["residual_divergence_count"] for e in legs)
    matched = all(e["all_tokens_matched"] for e in legs)
    tokens = {e["decode_tokens"] for e in legs}

    # The islands correct the PROPOSAL head, not the target, so both arms must
    # emit exactly the same tokens. Each leg's own exactness gate only compares
    # it with its own candidate-generated reference; this compares the arms.
    passes = {e["tag"]: read_passes(OUT / e["tag"]) for e in legs}
    cross = {}
    reference = passes[order[0]]
    for tag in order[1:]:
        other = passes[tag]
        entry = {"reference_passes": len(reference), "passes": len(other)}
        compared = mismatch = 0
        for a, b in zip(reference, other):
            shared = a.keys() & b.keys()
            compared += len(shared)
            mismatch += sum(1 for pos in shared if a[pos] != b[pos])
        entry["compared"] = compared
        entry["token_mismatches"] = mismatch
        cross[tag] = entry

    def arm_of(name: str) -> list[dict]:
        return [e for e in legs if e["arm"] == name]

    summary = {}
    for name in ("all", "q"):
        rows_here = arm_of(name)
        summary[name] = {
            "legs": [e["tag"] for e in rows_here],
            "mtp_seconds_per_token_mean": statistics.fmean(
                e["mtp_seconds_per_token"] for e in rows_here),
            "mtp_seconds_per_token_spread": abs(
                rows_here[0]["mtp_seconds_per_token"]
                - rows_here[-1]["mtp_seconds_per_token"]),
            "accepted_draft_rate_mean": statistics.fmean(
                e["accepted_draft_rate"] for e in rows_here),
            "effective_mean_draft_len_mean": statistics.fmean(
                e["effective_mean_draft_len"] for e in rows_here),
            "rounds_mean": statistics.fmean(e["rounds"] for e in rows_here),
            "per_step_p_mean": statistics.fmean(
                e["per_step_p"] for e in rows_here) if all(
                    "per_step_p" in e for e in rows_here) else None,
            "gpu_temp_entry_c": [e["gpu_temp_entry_c"] for e in rows_here],
            "gpu_temp_exit_c": [e["gpu_temp_exit_c"] for e in rows_here],
        }

    base_time = summary["all"]["mtp_seconds_per_token_mean"]
    cand_time = summary["q"]["mtp_seconds_per_token_mean"]
    # A FASTER candidate leg raises every affected ranked raw_p, so the raw
    # ratio gain is the fractional time REDUCTION. harness=ranked.
    time_pct = 100.0 * (base_time - cand_time) / base_time

    accept_delta_pp = 100.0 * (summary["q"]["accepted_draft_rate_mean"]
                               - summary["all"]["accepted_draft_rate_mean"])
    per_step_delta_pp = None
    if summary["q"]["per_step_p_mean"] is not None:
        per_step_delta_pp = 100.0 * (summary["q"]["per_step_p_mean"]
                                     - summary["all"]["per_step_p_mean"])

    # Rule 121 on the measured time effect. A head-numerics change touches
    # every prompt, and a uniform relative gain converts 1:1.
    median_pct = e143_value.median_pct_gain(
        {k: time_pct / 100.0 for k in e143_value.ANCHOR_572b2cc4})

    replicate_spread = max(summary["all"]["mtp_seconds_per_token_spread"],
                           summary["q"]["mtp_seconds_per_token_spread"])
    state = {
        "harness": "local",
        "design": "ABBA all,q,q,all; the arm contrast is mean(q) - mean(all)",
        "legs": legs,
        "arm_summary": summary,
        "cross_arm_token_agreement": cross,
        "rule_114_arm_witness_present": True,
        "rule_114_distinct_arms": witness_arms,
        "rule_101_wrong_arm_control": control,
        "rule_101_wrong_arm_control_passed": control_passed,
        "e143_exactness_divergences": divergences,
        "all_tokens_matched": matched,
        "decode_tokens": sorted(tokens),
        "e143_c2_realised_acceptance_delta_pp": accept_delta_pp,
        "e143_c2_realised_per_step_p_delta_pp": per_step_delta_pp,
        "e143_c2_time_pct_of_candidate_leg": time_pct,
        "e143_c2_ranked_pct": median_pct,
        "replicate_spread_seconds_per_token": replicate_spread,
        "replicate_spread_pct": 100.0 * replicate_spread / base_time,
        "prereg_point_pct": 0.35,
        "prereg_band_pct": [0.30, 0.42],
        "verdict": ("win" if median_pct >= 0.30
                    else "not useful" if median_pct > -0.30
                    else "refuted: the arm is slower"),
    }
    Path(args.out).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
