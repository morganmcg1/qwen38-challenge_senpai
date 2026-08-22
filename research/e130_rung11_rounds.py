#!/usr/bin/env python3
"""E130 rung 11, F18: split each ladder leg into round-1 excess and steady state.

The ladder's headline is A, the leg mean candidate seconds per token. F18 asks
whether the wired-residency arm moves the whole decode or only the first round
after the clock starts, because a rival submission is reported to exploit the
`Memory.clearCache()` at Qwen36MTPBlockSession.swift:235 that empties the MLX
pool so round 1 re-allocates from the OS.

    A = leg mean candidate seconds per token, from the score report
    B = median(round_us[2..N]) / tokens_per_round          steady state only
    C = round_us[1] - median(round_us[2..N])               round-1 excess

B is put back on a seconds-per-token axis by dividing by the leg's own mean
accepted tokens per round, so B and A carry the same units and a B/A ratio near
one means the trace and the report agree about the leg.

C is reported in MILLISECONDS PER LEG, never as a percentage. A one-time cost
divided by 512 tokens is a number that looks like a rate and is not one.

SOURCE. Only `mtp-anchor:` lines are read. They are the only trace line that
carries a pid, and one leg's trace file collects every worker the wrapper
spawns under O_APPEND, so a pid is the only safe grouping key. `mtp-trace:`
carries the same round_us with no pid and is deliberately ignored.

The depth-0 serial path returns before the anchor emit, so a serial worker
contributes no anchors at all. Every anchor in a leg belongs to a drafting
worker.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

ANCHOR = re.compile(r"^mtp-anchor:\s+(.*)$")
FIELD = re.compile(r"(\w+)=(-?\d+)")


def parse_anchor_lines(path: pathlib.Path) -> list[dict]:
    rounds = []
    with path.open("r", errors="replace") as handle:
        for line in handle:
            match = ANCHOR.match(line)
            if not match:
                continue
            fields = {k: int(v) for k, v in FIELD.findall(match.group(1))}
            if "t_round0" not in fields or "t_tail_done" not in fields:
                continue
            fields["round_us"] = (
                fields["t_tail_done"] - fields["t_round0"]) / 1000.0
            rounds.append(fields)
    return rounds


def group_by_pid(rounds: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for entry in rounds:
        groups.setdefault(entry.get("pid", -1), []).append(entry)
    for entries in groups.values():
        entries.sort(key=lambda e: e["t_round0"])
    return groups


def decompose(entries: list[dict]) -> dict:
    """A leg's round-1 excess and steady-state median, in microseconds."""
    series = [e["round_us"] for e in entries]
    first = series[0]
    tail = series[1:]
    steady = statistics.median(tail) if tail else float("nan")
    accepted = [e.get("acc", 0) for e in entries]
    # Tokens a round commits: the primary plus every accepted draft.
    committed = [a + 1 for a in accepted]
    steady_committed = (
        statistics.mean(committed[1:]) if len(committed) > 1 else float("nan"))
    return {
        "rounds": len(series),
        "round1_us": first,
        "steady_median_us": steady,
        "steady_mean_us": statistics.mean(tail) if tail else float("nan"),
        "steady_sd_us": (
            statistics.stdev(tail) if len(tail) > 1 else float("nan")),
        "round1_excess_us": first - steady,
        "committed_tokens": sum(committed),
        "steady_mean_committed_tokens_per_round": steady_committed,
        "steady_seconds_per_token": (
            steady / steady_committed / 1e6
            if steady_committed and steady_committed == steady_committed
            else float("nan")),
        "mean_draft_depth": (
            statistics.mean([e.get("d", 0) for e in entries])),
        "mean_accepted": statistics.mean(accepted) if accepted else 0.0,
    }


def read_leg(out_dir: pathlib.Path) -> dict:
    trace = out_dir / "trace.txt"
    result: dict = {"tag": out_dir.name, "trace_path": str(trace)}
    if not trace.exists():
        result["error"] = "no trace file"
        return result
    rounds = parse_anchor_lines(trace)
    if not rounds:
        result["error"] = "no mtp-anchor lines"
        return result
    groups = group_by_pid(rounds)
    result["pids"] = {
        str(pid): {"rounds": len(entries),
                   "first_t_round0": entries[0]["t_round0"],
                   "mean_d": statistics.mean(
                       [e.get("d", 0) for e in entries])}
        for pid, entries in groups.items()
    }
    # The timed MTP worker is the last drafting worker the leg starts. A
    # reference or verify worker that drafts would start earlier, so ordering
    # by first anchor is the discriminator; the pid map above is printed so a
    # leg with an unexpected worker count is visible rather than silently
    # collapsed.
    drafting = {pid: e for pid, e in groups.items()
                if statistics.mean([x.get("d", 0) for x in e]) > 0}
    chosen = drafting if drafting else groups
    pid = max(chosen, key=lambda p: chosen[p][0]["t_round0"])
    result["timed_pid"] = pid
    result["drafting_pids"] = sorted(drafting)
    result.update(decompose(chosen[pid]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True,
                        help="leg tag prefix under research/out")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dump", action="store_true",
                        help="print every round of the chosen pid per leg")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent / "research" / "out"
    if not root.exists():
        root = pathlib.Path("research/out")
    legs = sorted(p for p in root.glob(f"{args.prefix}-*") if p.is_dir())
    if not legs:
        print(f"no legs under {root} matching {args.prefix}-*", file=sys.stderr)
        return 2

    records = []
    for leg in legs:
        record = read_leg(leg)
        records.append(record)
        if "error" in record:
            print(f"{record['tag']:<28} ERROR {record['error']}")
            continue
        print(f"{record['tag']:<28} pid={record['timed_pid']} "
              f"rounds={record['rounds']:>3} "
              f"round1={record['round1_us'] / 1000:>9.2f} ms  "
              f"steady={record['steady_median_us'] / 1000:>8.2f} ms  "
              f"C={record['round1_excess_us'] / 1000:>9.2f} ms  "
              f"B={record['steady_seconds_per_token']:.9f} s/tok")
        if args.dump:
            trace = pathlib.Path(record["trace_path"])
            entries = group_by_pid(parse_anchor_lines(trace))[
                record["timed_pid"]]
            for index, entry in enumerate(entries, start=1):
                print(f"    r{index:<3} d={entry.get('d')} "
                      f"acc={entry.get('acc')} "
                      f"round_us={entry['round_us']:.1f}")

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"legs": records}, indent=2) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
