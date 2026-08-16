#!/usr/bin/env python3
"""Stall-guardrail statistics for the native-MTP decode leg.

The guardrail the advisor asked about reads `*_after_first`, i.e. it excludes
the first block of a leg (shape build, warm miss) and then compares the worst
remaining block against the median remaining block. Two independent views are
produced here:

  worker side  -- `round_us` from `MLX_QWEN_MTP_TRACE=1` round records, which
                  is the time the session itself spent in the round.
  parent side  -- `block_request_seconds` from CLI reports retained by
                  `research/capture-cli.sh`, which additionally contains the
                  worker-protocol overhead the guardrail actually sees.

The worker view is available for any traced run already on disk; the parent
view needs the capture shim. Where both exist they should agree in shape.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

PHASE_MARKERS = (
    ("reference", "generating the MTP reference rows"),
    ("serial", "measuring the TRUE serial control"),
    ("mtp", "measuring native-MTP decode"),
)

ROUND_RE = re.compile(r"mtp-trace: round=(\d+) .*?\bround_us=(\d+)")
DEPTH_RE = re.compile(r"\bd=(\d+)")
ACC_RE = re.compile(r"\bacc=(\d+)")


def phase_of(line: str, current: str) -> str:
    for name, marker in PHASE_MARKERS:
        if marker in line:
            return name
    return current


def parse_trace(path: str) -> dict[str, list[dict]]:
    phases: dict[str, list[dict]] = {name: [] for name, _ in PHASE_MARKERS}
    current = "reference"
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            current = phase_of(line, current)
            match = ROUND_RE.search(line)
            if not match:
                continue
            depth = DEPTH_RE.search(line)
            accepted = ACC_RE.search(line)
            phases[current].append(
                {
                    "round": int(match.group(1)),
                    "seconds": int(match.group(2)) / 1e6,
                    "depth": int(depth.group(1)) if depth else None,
                    "accepted": int(accepted.group(1)) if accepted else None,
                }
            )
    return phases


def parse_capture(directory: str) -> list[dict]:
    reports = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (ValueError, OSError):
            continue
        blocks = find_blocks(payload)
        if blocks:
            reports.append(
                {
                    "file": os.path.basename(path),
                    "blocks": [{"round": i, "seconds": float(v)} for i, v in enumerate(blocks)],
                    "decode_seconds": find_scalar(payload, "decode_seconds"),
                }
            )
    return reports


def find_blocks(payload) -> list | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "block_request_seconds" and isinstance(value, list) and value:
                return value
            found = find_blocks(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_blocks(item)
            if found:
                return found
    return None


def find_scalar(payload, key: str):
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name == key and isinstance(value, (int, float)):
                return float(value)
            found = find_scalar(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_scalar(item, key)
            if found is not None:
                return found
    return None


def guardrail(rounds: list[dict], prefill_seconds: float | None = None) -> dict:
    if not rounds:
        return {"count": 0}
    seconds = [r["seconds"] for r in rounds]
    first = seconds[0]
    rest = seconds[1:]
    out = {
        "count": len(seconds),
        "total_seconds": sum(seconds),
        "first_seconds": first,
    }
    if prefill_seconds is not None:
        out["prefill_seconds"] = prefill_seconds
        out["first_seconds_excluding_prefill"] = first - prefill_seconds
    if not rest:
        return out
    p50 = statistics.median(rest)
    worst = max(rest)
    worst_at = rest.index(worst)
    out.update(
        {
            "p50_after_first": p50,
            "p90_after_first": sorted(rest)[int(0.9 * (len(rest) - 1))],
            "max_after_first": worst,
            "max_after_first_round_index": worst_at + 1,
            "max_over_p50_after_first": worst / p50 if p50 else None,
            "first_over_p50_after_first": first / p50 if p50 else None,
            "max_round_depth": rounds[worst_at + 1].get("depth"),
            "max_round_accepted": rounds[worst_at + 1].get("accepted"),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace")
    parser.add_argument("--capture-dir")
    parser.add_argument(
        "--prefill-seconds",
        type=float,
        help="measured seed-prefill term charged to the first block",
    )
    parser.add_argument("--label", default="run")
    parser.add_argument("--out")
    args = parser.parse_args()

    result: dict = {"label": args.label, "guardrail_threshold": 4.0}

    if args.trace:
        phases = parse_trace(args.trace)
        result["worker_side"] = {
            name: guardrail(
                rounds,
                args.prefill_seconds if name == "mtp" else None,
            )
            for name, rounds in phases.items()
            if rounds
        }

    if args.capture_dir:
        result["parent_side"] = [
            {
                "file": report["file"],
                "decode_seconds": report["decode_seconds"],
                **guardrail(report["blocks"], args.prefill_seconds),
            }
            for report in parse_capture(args.capture_dir)
        ]

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
