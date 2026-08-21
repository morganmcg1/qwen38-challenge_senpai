#!/usr/bin/env python3
"""E87 option B: collect the load-time derivation evidence from capture legs.

usage: research/e87_derivation_report.py PREFIX [--canonical DIGEST]
                                         [--report research/e87-derivation.json]

Answers two questions that a timed session cannot.

  determinism  Does the table the Swift runtime derives from the DECLARED head
               equal the table the offline screen priced? Every worker process
               of every leg reports `order_fnv1a64` and dumps its row order, so
               the check is one digest comparison plus a byte comparison across
               processes.
  placement    Does the derivation land inside the timed window? The build's
               own Swift wall clock is compared against the round-1 minus
               round-2 excess and against the whole charged decode window. A
               build inside the window would have to appear in one of those.

A capture leg needs MLXFAST_NO_SANDBOX=1: benchmark.sh gives the worker a
Seatbelt profile that denies every file write except /dev/null, so a scored leg
leaves both sinks empty by construction. research/e79_trace_leg.sh sets it.

The two sinks were removed from the candidate before the pre-submit chain, so
this tool now reads the archived `e87r2p1` capture legs and the report it
already wrote. Restore MLX_E87_DERIVED_LOG and MLX_E87_DERIVED_DUMP in
Qwen35.swift on a scratch branch to capture a new base or a new host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

BUILD = re.compile(
    r"e87 pid=(\d+) build ok leaves=(\d+) rows_per_leaf=(\d+) probes=(\d+) "
    r"probe_fraction=([\d.]+) iters=(\d+) centroid_bits=(\d+) "
    r"order_fnv1a64=([0-9a-f]{16}) build_seconds=([\d.]+)")
ROUND_US = re.compile(r"round_us=([\d.]+)")


def meta_value(tag: str, key: str) -> str | None:
    for line in (OUT / tag / "meta.txt").read_text().splitlines():
        if line.startswith(key + "="):
            return line.partition("=")[2]
    return None


def score_metric(tag: str, key: str):
    path = OUT / tag / "score.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("metrics", {}).get(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--canonical", default="efa704192133552d",
                    help="order_fnv1a64 of the offline partition the screen priced")
    ap.add_argument("--report", default="research/e87-derivation.json")
    args = ap.parse_args()

    tags = sorted(p.name for p in OUT.iterdir()
                  if p.name.startswith(args.prefix + "-") and (p / "derived.log").is_file())
    if not tags:
        raise SystemExit(f"no capture legs under {OUT} with prefix {args.prefix}-")

    builds, legs = [], []
    for tag in tags:
        text = (OUT / tag / "derived.log").read_text()
        entered = text.count("build entered")
        leg_builds = []
        for m in BUILD.finditer(text):
            pid = int(m.group(1))
            dump = OUT / tag / f"derived-order.bin.{pid}"
            leg_builds.append({
                "leg": tag,
                "pid": pid,
                "leaves": int(m.group(2)),
                "rows_per_leaf": int(m.group(3)),
                "probes": int(m.group(4)),
                "probe_fraction": float(m.group(5)),
                "iterations": int(m.group(6)),
                "centroid_bits": int(m.group(7)),
                "order_fnv1a64": m.group(8),
                "build_seconds": float(m.group(9)),
                "dump_bytes": dump.stat().st_size if dump.is_file() else None,
                "dump_sha256": (hashlib.sha256(dump.read_bytes()).hexdigest()
                                if dump.is_file() else None),
            })
        rounds = [float(x) for x in ROUND_US.findall((OUT / tag / "trace.txt").read_text())]
        legs.append({
            "leg": tag,
            "arm": meta_value(tag, "e87_arm"),
            "probe_fraction": meta_value(tag, "e87_probe_fraction"),
            "worker_processes_that_built": len(leg_builds),
            "build_entered_count": entered,
            "one_build_per_process": entered == len(leg_builds),
            "rounds": len(rounds),
            "round1_us": rounds[0] if rounds else None,
            "round2_us": rounds[1] if len(rounds) > 1 else None,
            # A build inside the timed window would have to show up here: it is
            # three orders of magnitude larger than a round.
            "round1_excess_over_round2_us": (rounds[0] - rounds[1]
                                             if len(rounds) > 1 else None),
            "summed_round_us": sum(rounds) if rounds else None,
            "mtp_seconds_per_token": score_metric(tag, "mtp_seconds_per_token"),
            "all_tokens_matched": score_metric(tag, "all_tokens_matched"),
            "residual_divergence_count": score_metric(tag, "residual_divergence_count"),
            "head_provenance_sha256": score_metric(tag, "head_provenance_sha256"),
            "effective_mean_draft_len": score_metric(tag, "effective_mean_draft_len"),
            "accepted_draft_rate": score_metric(tag, "accepted_draft_rate"),
            "gpu_temp_entry_c": meta_value(tag, "gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta_value(tag, "gpu_temp_exit_c"),
        })
        builds.extend(leg_builds)

    digests = sorted({b["order_fnv1a64"] for b in builds})
    dumps = sorted({b["dump_sha256"] for b in builds if b["dump_sha256"]})
    seconds = [b["build_seconds"] for b in builds]

    charged = [l["mtp_seconds_per_token"] * 512 for l in legs
               if l["mtp_seconds_per_token"]]
    worst_excess = max((l["round1_excess_over_round2_us"] for l in legs
                        if l["round1_excess_over_round2_us"] is not None), default=None)
    build_us = max(seconds) * 1e6

    report = {
        "experiment": "e87-coarse-draft-shortlist-traffic",
        "question": "is the derived index deterministic, canonical, and untimed",
        "harness": "local",
        "prefix": args.prefix,
        "base_sha": meta_value(tags[0], "base_sha"),
        "worker_sha256": meta_value(tags[0], "worker_sha256"),
        "cli_sha256": meta_value(tags[0], "cli_sha256"),
        "metallib_source_fingerprint": meta_value(tags[0], "metallib_source_fingerprint"),
        "host": meta_value(tags[0], "host"),
        "chip": meta_value(tags[0], "chip"),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "sandbox_disabled_for_capture": True,
        "determinism": {
            "canonical_order_fnv1a64": args.canonical,
            "runtime_order_fnv1a64_values": digests,
            "matches_canonical": digests == [args.canonical],
            "processes": len(builds),
            "distinct_dump_sha256": dumps,
            "dumps_identical_across_processes": len(dumps) == 1,
        },
        "placement": {
            "build_seconds_min": min(seconds),
            "build_seconds_max": max(seconds),
            "build_seconds_mean": st.mean(seconds),
            "worst_round1_excess_over_round2_us": worst_excess,
            "worst_excess_as_fraction_of_build": (worst_excess / build_us
                                                  if worst_excess is not None else None),
            "charged_window_seconds": charged,
            "charged_window_if_build_were_timed_seconds": [c + max(seconds)
                                                           for c in charged],
            "build_is_outside_timed_window": (worst_excess is not None
                                              and worst_excess < 0.05 * build_us),
            "builds_once_per_process": all(l["one_build_per_process"] for l in legs),
        },
        "builds": builds,
        "legs": legs,
    }
    print(json.dumps(report, indent=2))
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
