"""E154 R2 round-time accounting and the FINDING 281 bandwidth confrontation.

Reads the zero arms of an R2 session and the bandwidth probe artifact, then
writes the derived numbers the result and the W&B run both quote. Everything
here is arithmetic on measured quantities; nothing is fitted twice.

    python3 research/e154_r2_accounting.py \
        .mlxfast-private/e154/runs-r2a research/e154-artifacts

harness=local. Never an official or ranked score.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics
import sys
from collections import Counter

# FINDING 281, harness=ranked.
RANKED_INTERCEPT_US = 25409.0
RANKED_SLOPE_US_PER_TOKEN = 4291.0
# One full read of the scored linear weights: 25.622 G params at
# (0.5 + 4/64) bytes for affine 4-bit group-64.
WEIGHT_STREAM_GB = 14.412
# M4 Pro published peak.
HOST_SPEC_PEAK_GBPS = 273.0

# Measured by research/e154_r2_analyze.py on the same session.
PREEVAL_SLACK_US = 47424.0
PREVERIFY_SLACK_US = 6080.0
GPU_SLACK_US = 865.1413916074814
GPU_SLACK_BRACKET_US = (313.86443940790707, 1260.0184627530796)


def ols(xs, ys):
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    residual = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(residual / (n - 2) / sxx)
    return slope, intercept, se, sxy / math.sqrt(sxx * syy)


def load_zero_rounds(runs_dir: str) -> list[dict]:
    rounds = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*zero", "rounds.txt"))):
        for line in open(path):
            fields = dict(re.findall(r"(\w+)=([-\d.]+)", line))
            if fields:
                rounds.append({k: float(v) for k, v in fields.items()})
    return rounds


def main() -> None:
    runs_dir = sys.argv[1]
    out_dir = sys.argv[2]
    rounds = load_zero_rounds(runs_dir)
    modal_d = Counter(r["d"] for r in rounds).most_common(1)[0][0]
    modal = [r for r in rounds if r["d"] == modal_d]

    wall = statistics.fmean([r["wall_us"] for r in modal])
    process_cpu = statistics.fmean([r["process_cpu_us"] for r in modal])
    thread_cpu = statistics.fmean([r["thread_cpu_us"] for r in modal])

    # Acceptance at fixed draft depth. This is the only place in the session
    # where the number of emitted tokens moves while the verified row count
    # is held constant, so it is the only clean price for an acceptance point.
    acc_slope, acc_intercept, acc_se, acc_r = ols(
        [r["acc"] for r in modal], [r["wall_us"] for r in modal])

    rows_slope, rows_intercept, rows_se, rows_r = ols(
        [r["d"] + 1 for r in rounds], [r["wall_us"] for r in rounds])
    tok_slope, tok_intercept, tok_se, tok_r = ols(
        [r["acc"] + 1 for r in rounds], [r["wall_us"] for r in rounds])

    with open(os.path.join(out_dir, "e154_bandwidth.json")) as handle:
        bandwidth = json.load(handle)
    achievable = bandwidth["e154_host_achievable_read_gbps"]

    tokens_per_round = 512.0 / 78.0
    ranked_round = (RANKED_INTERCEPT_US
                    + RANKED_SLOPE_US_PER_TOKEN * tokens_per_round)

    # A bandwidth-limited weight stream must take longer on a slower host in
    # exact proportion. Both directions of that test are reported because
    # each one fails differently.
    weight_stream_on_this_host_us = WEIGHT_STREAM_GB / achievable * 1e6
    required_gbps_for_local_intercept = WEIGHT_STREAM_GB / rows_intercept * 1e6

    report = {
        "experiment": "e154-r2-round-accounting",
        "harness": "local",
        "official_or_ranked_score": False,
        "runs_dir": runs_dir,
        "modal_draft_count": modal_d,
        "n_modal_rounds": len(modal),
        "n_zero_rounds": len(rounds),

        "e154_round_wall_time_us": wall,
        "host_process_cpu_us": process_cpu,
        "host_thread_cpu_us": thread_cpu,
        "host_cpu_fraction_of_round": process_cpu / wall,

        "e154_scored_round_eval_boundaries": {
            "blocking_evals": dict(
                Counter(int(r["blocking_evals"]) for r in rounds)),
            "async_evals": dict(
                Counter(int(r["async_evals"]) for r in rounds)),
        },

        # The device can only be idle where injected device work is free, so
        # the GPU slack is an upper bound on device idle in the round.
        "device_busy_us_lower_bound": wall - GPU_SLACK_US,
        "device_busy_fraction_lower_bound": (wall - GPU_SLACK_US) / wall,
        "device_busy_fraction_worst_case": (
            (wall - GPU_SLACK_BRACKET_US[1]) / wall),

        # Where the host's wall clock goes. Exactly one blocking eval per
        # round, so any other host wait is device-driven back pressure.
        "host_wait_at_blocking_eval_us": PREEVAL_SLACK_US,
        "host_wait_during_enqueue_us": wall - process_cpu - PREEVAL_SLACK_US,
        "host_wait_during_enqueue_fraction": (
            (wall - process_cpu - PREEVAL_SLACK_US) / wall),

        "e154_acceptance_price_at_fixed_depth": {
            "us_per_accepted_token": acc_slope,
            "se": acc_se,
            "t_stat": acc_slope / acc_se,
            "r": acc_r,
            "n": len(modal),
            "at_draft_count": modal_d,
            "fraction_of_round_per_token": acc_slope / wall,
        },
        "e154_cost_law_regressor_test": {
            "verified_rows": {"slope_us": rows_slope, "se": rows_se,
                              "intercept_us": rows_intercept, "r": rows_r},
            "emitted_tokens": {"slope_us": tok_slope, "se": tok_se,
                               "intercept_us": tok_intercept, "r": tok_r},
            "dominant": "verified_rows" if abs(rows_r) > abs(tok_r)
                        else "emitted_tokens",
        },

        "e154_host_achievable_read_gbps": achievable,
        "host_spec_peak_gbps": HOST_SPEC_PEAK_GBPS,
        "host_fraction_of_spec_peak": achievable / HOST_SPEC_PEAK_GBPS,
        "e154_finding_281_bandwidth_test": {
            "ranked_required_gbps": WEIGHT_STREAM_GB / RANKED_INTERCEPT_US
                                    * 1e6,
            "ranked_over_local_bandwidth_required": (
                WEIGHT_STREAM_GB / RANKED_INTERCEPT_US * 1e6 / achievable),
            "weight_stream_on_this_host_us": weight_stream_on_this_host_us,
            "local_fixed_term_us": rows_intercept,
            "local_fixed_term_over_weight_stream": (
                rows_intercept / weight_stream_on_this_host_us),
            "required_gbps_for_local_fixed_term":
                required_gbps_for_local_intercept,
            "ranked_round_at_local_t_us": ranked_round,
            "local_round_over_ranked_round": wall / ranked_round,
            "implied_ranked_gbps_if_round_is_bandwidth_bound": (
                achievable * wall / ranked_round),
            "tokens_per_round": tokens_per_round,
        },

        "e154_cpu_over_gpu_slack_ratio": PREEVAL_SLACK_US / GPU_SLACK_US,
        "e154_cpu_slack_us_per_round": {"preeval": PREEVAL_SLACK_US,
                                        "preverify": PREVERIFY_SLACK_US},
        "e154_gpu_slack_us_per_round": GPU_SLACK_US,
    }

    path = os.path.join(out_dir, "e154_r2_accounting.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
