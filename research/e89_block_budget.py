#!/usr/bin/env python3
"""Split each traced round into host-CPU time and blocked time.

Reports, per host-state stratum, where the drafting thread's wall clock goes:
the decode asyncEval ladder inside the verify-build window, the single blocking
eval, and everything else. Blocked time is wall minus thread CPU for the same
span, so it is the time a spin-wait could convert into runnable time.
"""
import json
import os
import re
import statistics
import sys

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "research/out/e89r2"
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SLOW_CUT_US = 1246

FIELDS = [
    "round", "round_us", "verify_build_us", "eval_wall_us", "host_sum_us",
    "cpu_verify_ns", "cpu_eval_ns", "round_thread_cpu_ns", "host_thread_cpu_ns",
    "d_submit2_us", "cpu_submit2_ns",
]


def parse(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            if "mtp-trace:" not in line:
                continue
            row = {}
            for token in line.split():
                if "=" not in token:
                    continue
                key, _, value = token.partition("=")
                if key in FIELDS:
                    try:
                        row[key] = int(value)
                    except ValueError:
                        pass
            if len(row) == len(FIELDS):
                rows.append(row)
    return rows


def leg_dirs(prefix):
    parent = os.path.dirname(prefix) or "."
    stem = os.path.basename(prefix)
    out = []
    for name in sorted(os.listdir(parent)):
        if name.startswith(stem):
            trace = os.path.join(parent, name, "trace.txt")
            if os.path.exists(trace):
                out.append((name, trace))
    return out


def main():
    fast, slow = [], []
    for name, trace in leg_dirs(PREFIX):
        if name.endswith("-bg-1") or "-bg-" in name:
            continue
        for row in parse(trace):
            if row["round"] <= WARMUP:
                continue
            (slow if row["host_sum_us"] > SLOW_CUT_US else fast).append(row)

    report = {}
    for label, rows in (("fast", fast), ("slow", slow)):
        if not rows:
            continue
        def med(key, scale=1.0):
            return statistics.median(r[key] * scale for r in rows)

        round_us = med("round_us")
        verify_us = med("verify_build_us")
        eval_us = med("eval_wall_us")
        cpu_verify_us = med("cpu_verify_ns", 1e-3)
        cpu_eval_us = med("cpu_eval_ns", 1e-3)
        cpu_round_us = med("round_thread_cpu_ns", 1e-3)
        report[label] = {
            "n_rounds": len(rows),
            "round_us": round_us,
            "thread_cpu_us": cpu_round_us,
            "duty_pct": 100.0 * cpu_round_us / round_us,
            "verify_build_us": verify_us,
            "verify_build_cpu_us": cpu_verify_us,
            "verify_build_blocked_us": verify_us - cpu_verify_us,
            "eval_wall_us": eval_us,
            "eval_cpu_us": cpu_eval_us,
            "eval_blocked_us": eval_us - cpu_eval_us,
            "blocked_total_us": (verify_us - cpu_verify_us) + (eval_us - cpu_eval_us),
            "blocked_total_pct_of_round":
                100.0 * ((verify_us - cpu_verify_us) + (eval_us - cpu_eval_us)) / round_us,
            "duty_if_both_spun_pct":
                100.0 * (cpu_round_us + (verify_us - cpu_verify_us) + (eval_us - cpu_eval_us))
                / round_us,
            "duty_if_only_eval_spun_pct":
                100.0 * (cpu_round_us + (eval_us - cpu_eval_us)) / round_us,
            "host_sum_us": med("host_sum_us"),
            "d_submit2_us": med("d_submit2_us"),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


main()
