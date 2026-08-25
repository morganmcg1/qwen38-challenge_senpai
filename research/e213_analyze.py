#!/usr/bin/env python3
"""Reduce an E213 Stage-1 palindrome session to its decision statistics.

    usage: research/e213_analyze.py SESSION_DIR [OUT_JSON]

Three arms: `off` (shipped plan), `retuned` (the assigned rows-per-group family
with (6,4) in place of the impossible (6,5)) and `na6` (the register-boundary
control at m = 9). Every arm keeps G = 2 at m = 6, 7, 8 and 9, so no arm changes
the weight-pass count.

Decision statistics (RULE 394 -- round endpoint or leg absolute only):

  a) trajectory-paired per-round deltas of the trusted parent's
     `block_request_seconds`, split by served width and summed round-weighted;
  b) whole-leg absolute candidate `mtp_seconds_per_token` from score.json.

Pairing validity (RULE 396(b)): the arms run the identical depth schedule, so
every leg must produce the same round count and the same per-round `(d, acc)`
trajectory. The script asserts that identity and refuses to pair if it does not
hold.

Coverage: an arm whose moved widths never appear in the census executed the same
code as `off`, so its paired delta is pure noise. That case is a recorded
problem, never a null result for the mechanism.

Candidate-side `round_us` and the phase split are attribution only.

harness=local. Not gate qualified. Not an official or ranked score.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import statistics
import sys

TRACE_ROUND = re.compile(r"^mtp-trace: round=(\d+) ")

# Widths each arm moves. An arm changes nothing at any other width, so these are
# the only rounds where its mechanism can act.
ARM_WIDTHS = {"retuned": [6, 7, 8], "na6": [9]}


def parse_meta(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def parse_trace(path: pathlib.Path) -> list[dict]:
    rounds = []
    if not path.exists():
        return rounds
    for line in path.read_text().splitlines():
        if not TRACE_ROUND.match(line):
            continue
        fields = {}
        for token in line[len("mtp-trace: "):].split(" "):
            key, sep, value = token.partition("=")
            if sep:
                fields[key] = value
        rounds.append(fields)
    return rounds


def load_leg(directory: pathlib.Path) -> dict:
    meta = parse_meta(directory / "meta.txt")
    score = json.loads((directory / "score.json").read_text())
    timed = json.loads((directory / "reports" / "04-mtp-timed.json").read_text())
    trace = parse_trace(directory / "trace.txt")
    assert timed["verb"] == "mtp-timed" and not timed["is_serial_control"], (
        f"{directory}: report 04 is not the candidate MTP leg"
    )
    return {
        "dir": str(directory),
        "arm": meta["e213_arm"],
        "leg": int(meta["e213_leg"]),
        "expected_plan": meta["e213_expected_plan"],
        "observed_plans": meta["e213_observed_plans"],
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
        "worker_sha256": meta["worker_sha256_before"],
        "worker_digest_stable": meta["worker_digest_stable"] == "true",
        "candidate_sha": meta["candidate_sha"],
        "campaign_base_sha": meta["campaign_base_sha"],
        "dirty_candidate_paths": int(meta["dirty_candidate_paths"]),
        "host": meta["host"],
        "chip": meta["chip"],
        "tokens": int(meta["tokens"]),
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
        "official_or_ranked_score": meta["official_or_ranked_score"],
        "metallib_source_fingerprint": meta["metallib_source_fingerprint"],
        # trusted parent, per round
        "block_request_seconds": timed["block_request_seconds"],
        "effective_draft_lengths": timed["effective_draft_lengths"],
        "round_count": timed["round_count"],
        "decode_seconds": timed["decode_seconds"],
        "declared_rows_total": timed["declared_rows_total"],
        "reference_checked_row_total": timed["reference_checked_row_total"],
        "emitted_token_total": timed["emitted_token_total"],
        "rejected_draft_total": timed["rejected_draft_total"],
        "accepted_draft_total": timed["accepted_draft_total"],
        "residual_divergence_count": timed["residual_divergence_count"],
        "parity_all_ok": timed["parity_all_ok"],
        "verify_block_replayed_round_count": timed[
            "verify_block_replayed_round_count"
        ],
        # whole leg
        "mtp_seconds_per_token": score["metrics"]["mtp_seconds_per_token"],
        "serial_seconds_per_token": score["metrics"]["serial_seconds_per_token"],
        "mtp_decode_speedup": score["metrics"]["mtp_decode_speedup"],
        "effective_mean_draft_len": score["metrics"]["effective_mean_draft_len"],
        "accepted_draft_rate": score["metrics"]["accepted_draft_rate"],
        "all_tokens_matched": score["metrics"]["all_tokens_matched"],
        "decode_tokens": score["metrics"]["decode_tokens"],
        # candidate side, attribution only
        "trace": trace,
    }


def mtp_trace(leg: dict) -> list[dict]:
    """The trace covers the whole process, so it starts with the serial control
    rounds. The candidate MTP rounds are the tail; the trusted report's
    `effective_draft_lengths` identifies them exactly."""
    rounds = leg["round_count"]
    tail = leg["trace"][-rounds:] if leg["trace"] else []
    if len(tail) != rounds:
        return []
    if [int(r["d"]) for r in tail] != leg["effective_draft_lengths"]:
        return []
    return tail


def trajectory(leg: dict) -> list[tuple[int, int]]:
    """`(d, acc)` per round over the candidate MTP leg. The trace carries both;
    the trusted report carries `d` alone and is the cross-check."""
    tail = mtp_trace(leg)
    if tail:
        return [(int(r["d"]), int(r["acc"])) for r in tail]
    return [(d, -1) for d in leg["effective_draft_lengths"]]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(values),
        "sd": sd,
        "sem": sem,
        "ci95_lo": mean - 1.96 * sem,
        "ci95_hi": mean + 1.96 * sem,
    }


def paired_delta(
    base_legs: list[dict], arm_legs: list[dict], mask: list[bool] | None
) -> dict:
    """Per-round mean(base) - mean(arm) in milliseconds. Positive means the arm
    is faster, so the figure reads as recovered time."""
    rounds = base_legs[0]["round_count"]
    deltas = []
    for index in range(rounds):
        if mask is not None and not mask[index]:
            continue
        base = statistics.fmean(
            leg["block_request_seconds"][index] for leg in base_legs
        )
        arm = statistics.fmean(
            leg["block_request_seconds"][index] for leg in arm_legs
        )
        deltas.append((base - arm) * 1000.0)
    return summarize(deltas)


def round_weighted(per_width: dict, census: dict, rounds: int) -> float:
    """Round-weighted whole-leg figure in ms per round: the per-width mean
    recovered time, weighted by that width's share of all rounds. Widths the arm
    does not move contribute nothing by construction."""
    total = 0.0
    for width, stats in per_width.items():
        if stats.get("n"):
            total += stats["mean"] * census.get(int(width), 0)
    return total / rounds


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    session = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None

    legs = [load_leg(d) for d in sorted(session.glob("leg*-*")) if d.is_dir()]
    assert legs, f"no legs under {session}"
    legs.sort(key=lambda leg: leg["leg"])

    problems = []
    for leg in legs:
        if leg["observed_plans"] != leg["expected_plan"]:
            problems.append(
                f"leg {leg['leg']}: arm witness {leg['observed_plans']} "
                f"!= {leg['expected_plan']}"
            )
        if not leg["all_tokens_matched"] or leg["residual_divergence_count"]:
            problems.append(f"leg {leg['leg']}: exactness failed")
        if leg["declared_rows_total"] != leg["reference_checked_row_total"]:
            problems.append(f"leg {leg['leg']}: row ledger did not close")
        if not leg["worker_digest_stable"]:
            problems.append(f"leg {leg['leg']}: worker digest moved")
        if leg["dirty_candidate_paths"]:
            problems.append(f"leg {leg['leg']}: dirty candidate paths")
    digests = {leg["worker_sha256"] for leg in legs}
    if len(digests) != 1:
        problems.append(f"legs timed {len(digests)} distinct workers")

    reference = trajectory(legs[0])
    identical = all(trajectory(leg) == reference for leg in legs)
    if not identical:
        problems.append("(d, acc) trajectories differ across legs")

    by_arm: dict[str, list[dict]] = {}
    for leg in legs:
        by_arm.setdefault(leg["arm"], []).append(leg)
    assert "off" in by_arm, "session needs the off arm"

    widths = [d + 1 for d, _ in reference]
    census: dict[int, int] = {}
    for width in widths:
        census[width] = census.get(width, 0) + 1

    for arm, moved in ARM_WIDTHS.items():
        if arm not in by_arm:
            continue
        if not any(census.get(width) for width in moved):
            problems.append(
                f"no rounds at widths {moved}: the {arm} plan entries never "
                "ran, so this session cannot measure that arm"
            )

    result = {
        "experiment": "e213-ipg5-retune-m678",
        "harness": "local",
        "cool_gate_passed_real_gate": legs[0]["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": legs[0]["gate_qualified_for_timing"],
        "official_or_ranked_score": legs[0]["official_or_ranked_score"],
        "timing_source": "trusted-parent block_request_seconds",
        "session_order": [leg["arm"] for leg in legs],
        "candidate_sha": legs[0]["candidate_sha"],
        "campaign_base_sha": legs[0]["campaign_base_sha"],
        "worker_sha256": legs[0]["worker_sha256"],
        "host": legs[0]["host"],
        "chip": legs[0]["chip"],
        "metallib_source_fingerprint": legs[0]["metallib_source_fingerprint"],
        "tokens": legs[0]["tokens"],
        "round_count": legs[0]["round_count"],
        "trajectories_identical": identical,
        "width_census": {str(k): v for k, v in sorted(census.items())},
        "arm_widths": ARM_WIDTHS,
        "problems": problems,
        "gpu_temp_entry_c": [leg["gpu_temp_entry_c"] for leg in legs],
        "gpu_temp_exit_c": [leg["gpu_temp_exit_c"] for leg in legs],
        "gpu_temp_entry_spread_c": max(leg["gpu_temp_entry_c"] for leg in legs)
        - min(leg["gpu_temp_entry_c"] for leg in legs),
        "legs": [
            {
                key: leg[key]
                for key in (
                    "leg", "arm", "observed_plans", "mtp_seconds_per_token",
                    "serial_seconds_per_token", "mtp_decode_speedup",
                    "effective_mean_draft_len", "accepted_draft_rate",
                    "all_tokens_matched", "residual_divergence_count",
                    "declared_rows_total", "reference_checked_row_total",
                    "round_count", "decode_seconds", "gpu_temp_entry_c",
                    "gpu_temp_exit_c", "verify_block_replayed_round_count",
                )
            }
            for leg in legs
        ],
    }

    off_legs = by_arm["off"]
    arms = {}
    for arm, arm_legs in by_arm.items():
        if arm == "off":
            continue
        entry = {"legs": [leg["leg"] for leg in arm_legs]}
        if identical and not problems:
            per_width = {}
            for width in sorted(census):
                mask = [w == width for w in widths]
                per_width[str(width)] = paired_delta(off_legs, arm_legs, mask)
            entry["paired_per_width_ms_per_round_recovered"] = per_width
            entry["paired_all_rounds_ms_per_round_recovered"] = paired_delta(
                off_legs, arm_legs, None
            )
            moved = ARM_WIDTHS.get(arm, [])
            entry["paired_moved_widths_ms_per_round_recovered"] = paired_delta(
                off_legs, arm_legs, [w in moved for w in widths]
            )
            # The widths this arm does not touch. A non-zero effect there means
            # the change disturbed the shared kernel, not just its own entries.
            entry["paired_untouched_widths_ms_per_round_recovered"] = (
                paired_delta(
                    off_legs, arm_legs, [w not in moved for w in widths]
                )
            )
            entry["round_weighted_ms_per_round_recovered"] = round_weighted(
                {w: s for w, s in per_width.items() if int(w) in moved},
                census, len(widths),
            )
        base_spt = [leg["mtp_seconds_per_token"] for leg in off_legs]
        arm_spt = [leg["mtp_seconds_per_token"] for leg in arm_legs]
        base_mean = statistics.fmean(base_spt)
        arm_mean = statistics.fmean(arm_spt)
        entry["whole_leg_absolute"] = {
            "off_mtp_seconds_per_token": base_spt,
            "arm_mtp_seconds_per_token": arm_spt,
            "off_mean": base_mean,
            "arm_mean": arm_mean,
            "delta_seconds_per_token": base_mean - arm_mean,
            "relative_gain": (base_mean - arm_mean) / base_mean,
            "off_serial_seconds_per_token": [
                leg["serial_seconds_per_token"] for leg in off_legs
            ],
            "arm_serial_seconds_per_token": [
                leg["serial_seconds_per_token"] for leg in arm_legs
            ],
            "off_local_ratio": [leg["mtp_decode_speedup"] for leg in off_legs],
            "arm_local_ratio": [leg["mtp_decode_speedup"] for leg in arm_legs],
        }
        arms[arm] = entry
    result["arms"] = arms

    # Session-drift controls: the two legs of one arm sit at opposite ends of
    # the palindrome, so their paired difference is the drift floor every effect
    # must clear.
    if identical and not problems:
        drift = {}
        for arm, arm_legs in by_arm.items():
            if len(arm_legs) == 2:
                drift[arm] = paired_delta([arm_legs[0]], [arm_legs[1]], None)
        result["drift_controls_ms_per_round"] = drift

    # Attribution only: candidate-side round_us and the round phase split, at
    # the widths each arm moves.
    if all(mtp_trace(leg) for leg in legs):
        attribution = {}
        for arm, arm_legs in by_arm.items():
            if arm == "off":
                continue
            moved = ARM_WIDTHS.get(arm, [])
            mask = [w in moved for w in widths]
            fields = {}
            for field in (
                "round_us", "draft_build_us", "verify_build_us", "eval_wall_us",
                "readout_us", "commit_us", "upkeep_us",
            ):
                per_arm = {}
                for name, group in (("off", off_legs), (arm, arm_legs)):
                    values = [
                        float(r[field])
                        for leg in group
                        for index, r in enumerate(mtp_trace(leg))
                        if field in r and mask[index]
                    ]
                    per_arm[name] = summarize(values)
                per_arm["delta_us_moved_widths"] = (
                    per_arm["off"].get("mean", 0.0)
                    - per_arm[arm].get("mean", 0.0)
                )
                fields[field] = per_arm
            attribution[arm] = fields
        result["attribution_candidate_side"] = attribution

    text = json.dumps(result, indent=2, sort_keys=True)
    if out_path:
        out_path.write_text(text + "\n")
    print(text)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
