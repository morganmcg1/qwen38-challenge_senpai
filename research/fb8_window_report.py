#!/usr/bin/env python3
"""FB8 window-labelled arm report.

One report per arm, at one decode window, answering the five FB8 questions that
the r1 256-token tables could not:

  1. schedule counters (rounds, depth histogram, accepted tokens per round,
     reject round rate) at this window;
  2. the `*_after_first` stall guardrail RE-TAKEN at this window, reported with
     the round count it was drawn from, the depth of the max round and of the
     p50 round, and the max round's index as a fraction of total rounds;
  3. the same guardrail rebased onto the ranked 4-bit head with the FB7
     per-draft delta, `(M - k*d_max) / (P - k*d_p50)`;
  4. the acceptance split before/after a decode index (the public long-copy
     fixture emits EOS around 301, after which the continuation is degenerate
     repetitive text whose acceptance is not representative);
  5. the parent-side view of 2 when a `research/capture-cli.sh` report is given.

Worker-side round times come from `MLX_QWEN_MTP_TRACE=1`; parent-side block
times come from a retained CLI report's `block_request_seconds`.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics

PHASE_MARKERS = (
    ("reference", "generating the MTP reference rows"),
    ("serial", "measuring the TRUE serial control"),
    ("mtp", "measuring native-MTP decode"),
)

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?\bround_us=(\d+)")
CAP_RE = re.compile(r"\bcap=(\d+)")
STREAK_RE = re.compile(r"\bstreak_in=(\d+)")

# FB7: bf16 local head payload minus declared 4-bit head payload, over the
# measured 227 GB/s backbone read bandwidth -- the per-draft cost the ranked
# head does NOT pay.
RANKED_HEAD_DELTA_MS = (849398784 - 238934093) / 227e9 * 1e3


def phase_of(line: str, current: str) -> str:
    for name, marker in PHASE_MARKERS:
        if marker in line:
            return name
    return current


def parse_rounds(path: str) -> list[dict]:
    """Round records of the timed native-MTP phase, in order."""
    rounds: list[dict] = []
    current = "reference"
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            current = phase_of(line, current)
            if current != "mtp":
                continue
            match = ROUND_RE.search(line)
            if not match:
                continue
            cap = CAP_RE.search(line)
            streak = STREAK_RE.search(line)
            rounds.append(
                {
                    "round": int(match.group(1)),
                    "depth": int(match.group(2)),
                    "accepted": int(match.group(3)),
                    "ms": int(match.group(4)) / 1e3,
                    "cap": int(cap.group(1)) if cap else None,
                    "streak_in": int(streak.group(1)) if streak else None,
                }
            )
    start = 0
    for position, record in enumerate(rounds):
        record["position"] = position
        record["start_index"] = start
        record["committed"] = 1 + record["accepted"]
        start += record["committed"]
        record["end_index"] = start
        record["rejected"] = record["depth"] - record["accepted"]
    return rounds


def histogram(values) -> dict:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {str(key): counts[key] for key in sorted(counts)}


def p50_member(records: list[dict]) -> dict:
    """The record AT the median round time (lower median for even counts)."""
    ordered = sorted(records, key=lambda r: r["ms"])
    return ordered[(len(ordered) - 1) // 2]


def guardrail(records: list[dict], total_rounds: int) -> dict:
    """`*_after_first` guardrail plus the FB8 provenance fields."""
    if len(records) < 2:
        return {"available": False, "n_after_first": len(records)}
    after = records[1:]
    times = [r["ms"] for r in after]
    worst = max(after, key=lambda r: r["ms"])
    median_record = p50_member(after)
    p50 = statistics.median(times)
    ratio = worst["ms"] / p50
    d_max = worst["depth"]
    d_p50 = median_record["depth"]
    ranked_max = worst["ms"] - RANKED_HEAD_DELTA_MS * d_max
    ranked_p50 = p50 - RANKED_HEAD_DELTA_MS * d_p50
    return {
        "available": True,
        "total_rounds": total_rounds,
        "n_after_first": len(after),
        "first_ms": records[0]["ms"],
        "p50_after_first_ms": p50,
        "max_after_first_ms": worst["ms"],
        "ratio_local": ratio,
        "first_over_p50": records[0]["ms"] / p50,
        "depth_of_max_round": d_max,
        "accepted_in_max_round": worst["accepted"],
        "max_round_rejected": worst["rejected"] > 0,
        "depth_of_p50_round": d_p50,
        "max_round_index": worst["position"],
        "max_round_position_fraction": (
            worst["position"] / max(total_rounds - 1, 1)),
        "max_is_last_round": worst["position"] == records[-1]["position"],
        "ranked_head_delta_ms_per_draft": RANKED_HEAD_DELTA_MS,
        "ratio_ranked": ranked_max / ranked_p50,
        "ratio_shift_ranked_minus_local": ranked_max / ranked_p50 - ratio,
    }


def split_stats(records: list[dict], label: str) -> dict:
    drafted = sum(r["depth"] for r in records)
    accepted = sum(r["accepted"] for r in records)
    rejected = sum(r["rejected"] for r in records)
    reject_rounds = sum(1 for r in records if r["rejected"] > 0)
    committed = sum(r["committed"] for r in records)
    n = len(records)
    return {
        "segment": label,
        "rounds": n,
        "committed_tokens": committed,
        "drafted_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "accepted_draft_rate": (accepted / drafted) if drafted else None,
        "reject_round_rate": (reject_rounds / n) if n else None,
        "accepted_tokens_per_round": (accepted / n) if n else None,
        "committed_tokens_per_round": (committed / n) if n else None,
        "rounds_per_token": (n / committed) if committed else None,
        "mean_depth": (drafted / n) if n else None,
        "depth_histogram": histogram(r["depth"] for r in records),
        "mean_round_ms": statistics.fmean(r["ms"] for r in records) if n else None,
        "ms_per_committed_token": (
            sum(r["ms"] for r in records) / committed) if committed else None,
    }


def parent_blocks(path: str) -> dict:
    with open(path, "r") as handle:
        report = json.load(handle)
    blocks = (report.get("block_request_seconds")
              or report.get("blockRequestSeconds"))
    if not blocks:
        return {"available": False, "path": path}
    records = [{"round": i, "position": i, "ms": value * 1e3, "depth": 0,
                "accepted": 0, "rejected": 0}
               for i, value in enumerate(blocks)]
    rail = guardrail(records, len(records))
    rail["path"] = path
    rail["decode_seconds"] = report.get("decode_seconds")
    rail["blocks"] = len(blocks)
    # Depth is unknown parent-side; the ranked rebase is meaningless without it.
    for key in ("ratio_ranked", "ratio_shift_ranked_minus_local",
                "depth_of_max_round", "depth_of_p50_round",
                "accepted_in_max_round", "max_round_rejected",
                "ranked_head_delta_ms_per_draft"):
        rail.pop(key, None)
    return rail


def parent_blocks_with_depths(path: str, records: list[dict]) -> dict:
    """Parent-side guardrail with worker round depths joined in by position."""
    with open(path, "r") as handle:
        report = json.load(handle)
    blocks = (report.get("block_request_seconds")
              or report.get("blockRequestSeconds"))
    if not blocks or len(blocks) != len(records):
        return {"available": False, "path": path,
                "blocks": len(blocks) if blocks else 0,
                "worker_rounds": len(records)}
    joined = [{"round": i, "position": i, "ms": blocks[i] * 1e3,
               "depth": records[i]["depth"],
               "accepted": records[i]["accepted"],
               "rejected": records[i]["rejected"]}
              for i in range(len(blocks))]
    rail = guardrail(joined, len(joined))
    rail["path"] = path
    rail["decode_seconds"] = report.get("decode_seconds")
    return rail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--window", type=int, required=True,
                        help="decode tokens the leg was asked for")
    parser.add_argument("--window-role", required=True,
                        choices=["inner-loop-screen", "directional-screen",
                                 "ranked-equivalent-headline"])
    parser.add_argument("--eos-index", type=int, default=301,
                        help="decode index where the fixture emits EOS")
    parser.add_argument("--parent-mtp", default=None,
                        help="retained CLI report for the MTP timed leg")
    parser.add_argument("--parent-serial", default=None,
                        help="retained CLI report for the serial control leg")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    records = parse_rounds(args.trace)
    if not records:
        raise SystemExit(f"no timed MTP rounds in {args.trace}")

    cut = args.eos_index
    pre = [r for r in records if r["end_index"] <= cut]
    post = [r for r in records if r["start_index"] >= cut]
    straddle = [r for r in records
                if r["start_index"] < cut < r["end_index"]]

    report = {
        "label": args.label,
        "trace": args.trace,
        "window_tokens": args.window,
        "window_role": args.window_role,
        "eos_index": cut,
        "committed_tokens_total": records[-1]["end_index"],
        "overall": split_stats(records, "all"),
        "pre_eos": split_stats(pre, f"decode index < {cut}"),
        "post_eos": split_stats(post, f"decode index >= {cut}"),
        "straddling_round": (
            {k: straddle[0][k] for k in
             ("round", "depth", "accepted", "start_index", "end_index")}
            if straddle else None),
        "guardrail_worker": guardrail(records, len(records)),
    }
    if args.parent_mtp:
        report["guardrail_parent_mtp"] = parent_blocks_with_depths(
            args.parent_mtp, records)
    if args.parent_serial:
        report["guardrail_parent_serial"] = parent_blocks(args.parent_serial)

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
