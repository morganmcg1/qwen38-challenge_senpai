#!/usr/bin/env python3
"""Reduce an E208 Stage-1 ABBA session to its decision statistics.

    usage: research/e208_analyze.py SESSION_DIR [OUT_JSON]

Decision statistics (RULE 394 -- round endpoint or leg absolute only):

  a) trajectory-paired per-round deltas of the trusted parent's
     `block_request_seconds`, restricted to the m = 9 rounds, and over all
     rounds for the round-weighted figure;
  b) whole-leg absolute candidate `mtp_seconds_per_token` from score.json.

Pairing validity (RULE 396(b)): the arms run the identical depth schedule, so
every leg must produce the same round count and the same per-round
`(d, acc)` trajectory. The script asserts that identity and refuses to pair if
it does not hold.

Candidate-side `round_us` from the trace is reported for attribution only, and
so is the phase split.

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
        "arm": meta["e208_arm"],
        "leg": int(meta["e208_leg"]),
        "expected_plan": meta["e208_expected_plan"],
        "observed_plans": meta["e208_observed_plans"],
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
        "worker_sha256": meta["worker_sha256_before"],
        "worker_digest_stable": meta["worker_digest_stable"] == "true",
        "base_sha": meta["base_sha"],
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


def trajectory(leg: dict) -> list[tuple[int, int]]:
    """`(d, acc)` per round. The trace carries both; the trusted report carries
    `d` alone, so the trace is preferred and the report is the cross-check."""
    if leg["trace"]:
        return [(int(r["d"]), int(r["acc"])) for r in leg["trace"]]
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
    off_legs: list[dict], on_legs: list[dict], mask: list[bool] | None
) -> dict:
    """Per-round mean(off) - mean(on) in milliseconds. Positive means the ON arm
    is faster, so the figure reads as recovered time."""
    rounds = off_legs[0]["round_count"]
    deltas = []
    for index in range(rounds):
        if mask is not None and not mask[index]:
            continue
        off = statistics.fmean(
            leg["block_request_seconds"][index] for leg in off_legs
        )
        on = statistics.fmean(
            leg["block_request_seconds"][index] for leg in on_legs
        )
        deltas.append((off - on) * 1000.0)
    return summarize(deltas)


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
    digests = {leg["worker_sha256"] for leg in legs}
    if len(digests) != 1:
        problems.append(f"legs timed {len(digests)} distinct workers")

    reference = trajectory(legs[0])
    identical = all(trajectory(leg) == reference for leg in legs)
    if not identical:
        problems.append("(d, acc) trajectories differ across legs")

    off_legs = [leg for leg in legs if leg["arm"] == "off"]
    on_legs = [leg for leg in legs if leg["arm"] == "on"]
    assert off_legs and on_legs, "session needs both arms"

    widths = [d + 1 for d, _ in reference]
    width_census = {}
    for width in widths:
        width_census[width] = width_census.get(width, 0) + 1
    is_nine = [width == 9 for width in widths]

    result = {
        "experiment": "e208-qmv-9row-ipg5",
        "harness": "local",
        "cool_gate_passed_real_gate": legs[0]["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": legs[0]["gate_qualified_for_timing"],
        "official_or_ranked_score": legs[0]["official_or_ranked_score"],
        "timing_source": "trusted-parent block_request_seconds",
        "session_order": [leg["arm"] for leg in legs],
        "base_sha": legs[0]["base_sha"],
        "worker_sha256": legs[0]["worker_sha256"],
        "host": legs[0]["host"],
        "chip": legs[0]["chip"],
        "metallib_source_fingerprint": legs[0]["metallib_source_fingerprint"],
        "tokens": legs[0]["tokens"],
        "round_count": legs[0]["round_count"],
        "trajectories_identical": identical,
        "width_census": {str(k): v for k, v in sorted(width_census.items())},
        "m9_round_share": width_census.get(9, 0) / len(widths),
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

    if identical and not problems:
        result["paired_m9_ms_per_round_recovered"] = paired_delta(
            off_legs, on_legs, is_nine
        )
        result["paired_all_rounds_ms_per_round_recovered"] = paired_delta(
            off_legs, on_legs, None
        )
        result["paired_non_m9_ms_per_round_recovered"] = paired_delta(
            off_legs, on_legs, [not flag for flag in is_nine]
        )
        # Session-drift control: the two same-arm legs sit at opposite ends of
        # the palindrome, so their paired difference is the drift floor the
        # effect must clear.
        if len(off_legs) == 2:
            result["drift_control_off_ms_per_round"] = paired_delta(
                [off_legs[0]], [off_legs[1]], None
            )
        if len(on_legs) == 2:
            result["drift_control_on_ms_per_round"] = paired_delta(
                [on_legs[0]], [on_legs[1]], None
            )

    off_spt = [leg["mtp_seconds_per_token"] for leg in off_legs]
    on_spt = [leg["mtp_seconds_per_token"] for leg in on_legs]
    off_mean = statistics.fmean(off_spt)
    on_mean = statistics.fmean(on_spt)
    result["whole_leg_absolute"] = {
        "off_mtp_seconds_per_token": off_spt,
        "on_mtp_seconds_per_token": on_spt,
        "off_mean": off_mean,
        "on_mean": on_mean,
        "delta_seconds_per_token": off_mean - on_mean,
        "relative_gain": (off_mean - on_mean) / off_mean,
        "off_serial_seconds_per_token": [
            leg["serial_seconds_per_token"] for leg in off_legs
        ],
        "on_serial_seconds_per_token": [
            leg["serial_seconds_per_token"] for leg in on_legs
        ],
        "off_local_ratio": [leg["mtp_decode_speedup"] for leg in off_legs],
        "on_local_ratio": [leg["mtp_decode_speedup"] for leg in on_legs],
    }

    # Attribution only: candidate-side round_us and the round phase split.
    if all(leg["trace"] for leg in legs):
        attribution = {}
        for field in (
            "round_us", "draft_build_us", "verify_build_us", "eval_wall_us",
            "readout_us", "commit_us", "upkeep_us",
        ):
            per_arm = {}
            for name, group in (("off", off_legs), ("on", on_legs)):
                values = [
                    float(r[field])
                    for leg in group
                    for index, r in enumerate(leg["trace"])
                    if field in r and is_nine[index]
                ]
                per_arm[name] = summarize(values)
            per_arm["delta_us_m9"] = (
                per_arm["off"].get("mean", 0.0) - per_arm["on"].get("mean", 0.0)
            )
            attribution[field] = per_arm
        result["attribution_m9_candidate_side"] = attribution

    text = json.dumps(result, indent=2, sort_keys=True)
    if out_path:
        out_path.write_text(text + "\n")
    print(text)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
