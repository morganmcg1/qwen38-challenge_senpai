#!/usr/bin/env python3
"""E101 -- itemise the `draft_head` phase of one MTP round from a census leg.

    usage: research/e101_head_census.py CENSUS_JSONL [--width=9] [--drafts=8]
                                        [--phase=draft_head] [--json=OUT]

`research/e95_verify_census.py kernels` hard-codes `target_verify`, so this
reader exists to read the same `exclusive_kernels` table for the proposal head.
It reports isolated GPU time per signature, per round and per draft.

An `exclusive_kernels` entry is only a single kernel when the leg ran with
`MLX_E58_BUFFER_LIMIT_OPS=0`. With `OPS=1` MLX still admits a second op before
it commits, so a signature that names two kernels is a PAIR and its time is the
pair's time. The reader prints the per-signature kernel count so a pair is never
quoted as a single kernel by accident.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict


def load(path):
    legs = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "gputime":
                legs[record["pid"]].append(record)
    return legs


def pick(legs, key):
    best, best_n = None, 0
    for pid, snapshots in legs.items():
        n = sum(1 for s in snapshots if key in s.get("by_width_phase", {}))
        if n > best_n:
            best, best_n = pid, n
    return best, best_n


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    width, drafts, phase, out_json = 9, 8, "draft_head", None
    for argument in args[1:]:
        if argument.startswith("--width="):
            width = int(argument.split("=", 1)[1])
        elif argument.startswith("--drafts="):
            drafts = int(argument.split("=", 1)[1])
        elif argument.startswith("--phase="):
            phase = argument.split("=", 1)[1]
        elif argument.startswith("--json="):
            out_json = argument.split("=", 1)[1]

    legs = load(path)
    key = f"w{width}|{phase}"
    pid, rounds = pick(legs, key)
    if pid is None:
        available = sorted({k for snapshots in legs.values() for s in snapshots
                            for k in s.get("by_width_phase", {})})
        print(f"no {key} snapshots; available: {available}")
        return 1

    snapshots = [s for s in legs[pid] if key in s.get("by_width_phase", {})]
    phase_ns = sum(s["by_width_phase"][key]["gpu_ns"] for s in snapshots)
    phase_disp = sum(s["by_width_phase"][key]["dispatches"] for s in snapshots)
    prefix = key + "|"

    def table(field):
        buckets = defaultdict(lambda: [0, 0])
        for snapshot in snapshots:
            for entry_key, bucket in snapshot.get(field, {}).items():
                if entry_key.startswith(prefix):
                    signature = entry_key[len(prefix):]
                    buckets[signature][0] += bucket["buffers"]
                    buckets[signature][1] += bucket["gpu_ns"]
        rows = [(gpu_ns / 1e3 / rounds, gpu_ns / 1e3 / rounds / drafts,
                 buffers / rounds / drafts, signature.count(",") + 1, signature)
                for signature, (buffers, gpu_ns) in buckets.items()]
        rows.sort(reverse=True)
        return rows

    print(f"=== {path}  pid={pid}  {key}  rounds={rounds}  drafts/round={drafts}")
    print(f"    phase in-situ: {phase_ns / 1e3 / rounds:.2f} us/round, "
          f"{phase_disp / rounds:.1f} dispatches/round, "
          f"{phase_ns / 1e3 / rounds / drafts:.2f} us/draft")

    payload = {}
    for field, label in (("exclusive_kernels", "ISOLATED single-kernel buffers"),
                         ("signatures", "WHOLE-BUFFER signatures")):
        rows = table(field)
        print(f"--- {label} ({field})")
        print(f"    {'us/round':>10} {'us/draft':>9} {'buf/draft':>10}"
              f" {'kern':>5}  signature")
        for us_round, us_draft, buf_draft, kernels, signature in rows:
            print(f"    {us_round:10.2f} {us_draft:9.3f} {buf_draft:10.3f} "
                  f"{kernels:5d}  {signature[:150]}")
        total = sum(r[0] for r in rows)
        print(f"    TOTAL {total:.2f} us/round = {total / drafts:.3f} us/draft")
        payload[field] = [{"us_per_round": r[0], "us_per_draft": r[1],
                           "buffers_per_draft": r[2],
                           "kernels_in_signature": r[3],
                           "signature": r[4]} for r in rows]

    if out_json:
        pathlib.Path(out_json).write_text(json.dumps({
            "census": path,
            "pid": pid,
            "width": width,
            "phase": phase,
            "drafts_per_round": drafts,
            "rounds": rounds,
            "phase_us_per_round": phase_ns / 1e3 / rounds,
            "phase_dispatches_per_round": phase_disp / rounds,
            **payload,
        }, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
