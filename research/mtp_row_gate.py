#!/usr/bin/env python3
"""Hexfloat row gate + round-schedule analysis for one local Qwen MTP run.

Consumes the stderr trace produced by

    MLX_QWEN_MTP_TRACE=1 ./benchmark-qwen-mtp.sh --local-iterate 2>trace.log

(worker stderr must be forwarded by the parent) plus the wrapper's score.json,
and reports:

  * bit-exactness of every MTP-leg top-2 row against the SERIAL leg's row at
    the same absolute position, bucketed by the verify width that produced it;
  * the round schedule: depth histogram, reject rate, accepted tokens per
    round, rounds per token;
  * the wrapper's timing metrics.

Optionally logs the whole record to W&B.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

ROW_RE = re.compile(r"mtp-row: pos=(\d+) ids=(-?\d+),(-?\d+) v=(\S+)")
ROUND_RE = re.compile(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)")
BEGIN_RE = re.compile(r"mtp-trace: begin (.*)")
KV_RE = re.compile(r"(\w+)=(\S+)")

PHASE_MARKERS = (
    ("generating the MTP reference rows", "reference"),
    ("measuring the TRUE serial control", "serial"),
    ("measuring native-MTP decode", "mtp"),
)


@dataclass
class Row:
    pos: int
    ids: tuple[int, int]
    values: tuple[str, str]
    width: int | None = None
    round_index: int | None = None


@dataclass
class Round:
    index: int
    depth: int
    accepted: int
    timings: dict[str, int]
    rows: list[Row] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.depth + 1


@dataclass
class Phase:
    name: str
    rows: list[Row] = field(default_factory=list)
    rounds: list[Round] = field(default_factory=list)


def parse_trace(path: str) -> dict[str, Phase]:
    phases: dict[str, Phase] = {}
    current: Phase | None = None
    pending: list[Row] = []

    def flush_depth0(phase: Phase) -> None:
        for row in pending:
            row.width = 1
            phase.rows.append(row)
        pending.clear()

    with open(path, "r", errors="replace") as handle:
        for line in handle:
            for marker, name in PHASE_MARKERS:
                if marker in line:
                    if current is not None:
                        flush_depth0(current)
                    current = phases.setdefault(name, Phase(name))
            if current is None:
                continue
            row_match = ROW_RE.search(line)
            if row_match:
                pending.append(
                    Row(
                        pos=int(row_match.group(1)),
                        ids=(int(row_match.group(2)), int(row_match.group(3))),
                        values=tuple(row_match.group(4).split(",")),
                    )
                )
                continue
            round_match = ROUND_RE.search(line)
            if round_match:
                index = int(round_match.group(1))
                depth = int(round_match.group(2))
                accepted = int(round_match.group(3))
                timings = {
                    key: int(value)
                    for key, value in KV_RE.findall(round_match.group(4))
                    if value.isdigit()
                }
                rnd = Round(index=index, depth=depth, accepted=accepted, timings=timings)
                take = accepted + 1
                own = pending[-take:] if take <= len(pending) else pending[:]
                leftover = pending[: len(pending) - len(own)]
                for row in leftover:
                    row.width = 1
                    current.rows.append(row)
                for row in own:
                    row.width = depth + 1
                    row.round_index = index
                    current.rows.append(row)
                    rnd.rows.append(row)
                pending.clear()
                current.rounds.append(rnd)
                continue
            if BEGIN_RE.search(line) and current is not None:
                flush_depth0(current)
    if current is not None:
        flush_depth0(current)
    return phases


def row_gate(serial: Phase, mtp: Phase) -> dict:
    reference = {row.pos: row for row in serial.rows}
    per_width_total: Counter[int] = Counter()
    per_width_bad: Counter[int] = Counter()
    per_width_id_bad: Counter[int] = Counter()
    mismatches: list[dict] = []
    unmatched = 0
    for row in mtp.rows:
        ref = reference.get(row.pos)
        if ref is None:
            unmatched += 1
            continue
        width = row.width or 0
        per_width_total[width] += 1
        values_equal = row.values == ref.values
        ids_equal = row.ids == ref.ids
        if not values_equal:
            per_width_bad[width] += 1
            if len(mismatches) < 40:
                mismatches.append(
                    {
                        "pos": row.pos,
                        "width": width,
                        "round": row.round_index,
                        "mtp_ids": row.ids,
                        "serial_ids": ref.ids,
                        "mtp_values": row.values,
                        "serial_values": ref.values,
                    }
                )
        if not ids_equal:
            per_width_id_bad[width] += 1
    widths = sorted(per_width_total)
    return {
        "serial_rows": len(serial.rows),
        "mtp_rows": len(mtp.rows),
        "compared_rows": sum(per_width_total.values()),
        "unmatched_positions": unmatched,
        "value_mismatches": sum(per_width_bad.values()),
        "id_mismatches": sum(per_width_id_bad.values()),
        "per_width": {
            str(width): {
                "compared": per_width_total[width],
                "value_mismatches": per_width_bad[width],
                "id_mismatches": per_width_id_bad[width],
                "bit_exact": per_width_bad[width] == 0,
            }
            for width in widths
        },
        "mismatch_samples": mismatches,
    }


def schedule_stats(phase: Phase) -> dict:
    depths = Counter()
    accepted_total = 0
    drafted_total = 0
    rejecting_rounds = 0
    depth0_rounds = max(
        0, len([row for row in phase.rows if row.width == 1 and row.round_index is None])
    )
    timing_totals: dict[str, int] = defaultdict(int)
    for rnd in phase.rounds:
        depths[rnd.depth] += 1
        accepted_total += rnd.accepted
        drafted_total += rnd.depth
        if rnd.accepted < rnd.depth:
            rejecting_rounds += 1
        for key, value in rnd.timings.items():
            timing_totals[key] += value
    depths[0] += depth0_rounds
    rounds = sum(depths.values())
    tokens = accepted_total + rounds
    return {
        "rounds": rounds,
        "drafting_rounds": len(phase.rounds),
        "depth_histogram": {str(depth): count for depth, count in sorted(depths.items())},
        "mean_depth": (sum(d * c for d, c in depths.items()) / rounds) if rounds else 0.0,
        "drafted_tokens": drafted_total,
        "accepted_tokens": accepted_total,
        "rejected_tokens": drafted_total - accepted_total,
        "accept_rate": (accepted_total / drafted_total) if drafted_total else 0.0,
        "reject_round_rate": (rejecting_rounds / rounds) if rounds else 0.0,
        "accepted_tokens_per_round": (accepted_total / rounds) if rounds else 0.0,
        "tokens_per_round": (tokens / rounds) if rounds else 0.0,
        "rounds_per_token": (rounds / tokens) if tokens else 0.0,
        "mean_round_us": (timing_totals.get("round_us", 0) / len(phase.rounds))
        if phase.rounds
        else 0.0,
        "mean_eval_wall_us": (timing_totals.get("eval_wall_us", 0) / len(phase.rounds))
        if phase.rounds
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace")
    parser.add_argument("--score")
    parser.add_argument("--label", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--config", default="{}")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    record: dict = {"label": args.label, "notes": args.notes}
    record["config"] = json.loads(args.config)

    if args.score:
        with open(args.score) as handle:
            score = json.load(handle)
        record["score"] = score
        record["metrics"] = score.get("metrics", {})

    if args.trace:
        phases = parse_trace(args.trace)
        record["phases"] = {
            name: schedule_stats(phase) for name, phase in phases.items()
        }
        if "serial" in phases and "mtp" in phases:
            record["row_gate"] = row_gate(phases["serial"], phases["mtp"])

    text = json.dumps(record, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)

    if args.wandb:
        import wandb

        flat: dict = {}

        def flatten(prefix: str, value) -> None:
            if isinstance(value, dict):
                for key, sub in value.items():
                    flatten(f"{prefix}/{key}" if prefix else str(key), sub)
            elif isinstance(value, (int, float, bool)):
                flat[prefix] = value

        flatten("", record.get("metrics", {}))
        flatten("", {"phases": record.get("phases", {})})
        gate = dict(record.get("row_gate", {}))
        gate.pop("mismatch_samples", None)
        flatten("", {"row_gate": gate})
        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="qwen38-mlx-challenge-senpai",
            name=args.label,
            notes=args.notes,
            config=record["config"],
        )
        run.log(flat)
        run.summary.update(flat)
        artifact_payload = {
            "row_gate_mismatch_samples": record.get("row_gate", {}).get(
                "mismatch_samples", []
            )
        }
        run.summary["mismatch_samples_json"] = json.dumps(artifact_payload)[:8000]
        print(f"wandb_run_url={run.url}")
        print(f"wandb_run_id={run.id}")
        run.finish()


if __name__ == "__main__":
    main()
