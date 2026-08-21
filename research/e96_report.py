#!/usr/bin/env python3
"""Summarise E96 legs: per-round trace statistics beside the score metrics.

    usage: research/e96_report.py TAG [TAG ...]

Each leg contributes its rounds keyed by (d, acc), because the repair path a
round takes depends on both. A comparison between two arms is only valid
inside one such bucket.
"""
import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
FIELDS = (
    "round_us",
    "verify_build_us",
    "eval_wall_us",
    "draft_build_us",
    "readout_us",
    "commit_us",
    "upkeep_us",
)


def parse_meta(tag):
    meta = {}
    path = OUT / tag / "meta.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                meta[key] = value
    return meta


def parse_rounds(tag):
    rounds = []
    path = OUT / tag / "trace.txt"
    if not path.exists():
        return rounds
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        record = {}
        for token in line[len("mtp-trace: "):].split():
            key, _, value = token.partition("=")
            try:
                record[key] = int(value)
            except ValueError:
                continue
        rounds.append(record)
    return rounds


def score(tag):
    path = OUT / tag / "score.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("metrics", {})


def summarise(tag, warmup=1):
    meta = parse_meta(tag)
    rounds = parse_rounds(tag)
    kept = [r for r in rounds if r.get("round", 0) > warmup]
    buckets = {}
    for record in kept:
        key = (record.get("d"), record.get("acc"))
        buckets.setdefault(key, []).append(record)
    return {
        "tag": tag,
        "step_mode": meta.get("step_mode"),
        "tg_y": meta.get("tg_y"),
        "force_drafts": meta.get("force_drafts"),
        "tokens": meta.get("tokens"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "exit": meta.get("exit"),
        "rounds_total": len(rounds),
        "rounds_kept": len(kept),
        "score": score(tag),
        "buckets": buckets,
    }


def stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    tags = sys.argv[1:]
    if not tags:
        print(__doc__)
        return 2
    legs = [summarise(tag) for tag in tags]
    for leg in legs:
        metrics = leg["score"]
        print(
            f"{leg['tag']:<22} step={leg['step_mode']:<7} y={leg['tg_y']:<3}"
            f" d={leg['force_drafts']:<3} rounds={leg['rounds_kept']:<4}"
            f" exit={leg['exit']}"
            f" mtp_spt={metrics.get('mtp_seconds_per_token')}"
            f" serial_spt={metrics.get('serial_seconds_per_token')}"
            f" matched={metrics.get('all_tokens_matched')}"
            f" acc_rate={metrics.get('accepted_draft_rate')}"
            f" temp={leg['gpu_temp_entry_c']}->{leg['gpu_temp_exit_c']}"
        )
        for key in sorted(leg["buckets"], key=lambda k: (k[0] or 0, k[1] or 0)):
            records = leg["buckets"][key]
            row = [f"  d={key[0]} acc={key[1]} n={len(records)}"]
            for field in FIELDS:
                summary = stats([r[field] for r in records if field in r])
                if summary:
                    row.append(f"{field}={summary['mean']:.0f}")
            print(" ".join(row))
    print()
    print("== arm aggregate over identical (d, acc) buckets ==")
    by_arm = {}
    for leg in legs:
        arm = f"{leg['step_mode']}/y{leg['tg_y']}"
        for key, records in leg["buckets"].items():
            by_arm.setdefault(arm, {}).setdefault(key, []).extend(records)
    shared = None
    for arm, buckets in by_arm.items():
        keys = set(buckets)
        shared = keys if shared is None else (shared & keys)
    for key in sorted(shared or [], key=lambda k: (k[0] or 0, k[1] or 0)):
        print(f"-- bucket d={key[0]} acc={key[1]}")
        reference = None
        for arm in by_arm:
            records = by_arm[arm][key]
            summary = {
                field: stats([r[field] for r in records if field in r])
                for field in FIELDS
            }
            mean = summary["round_us"]["mean"]
            if reference is None:
                reference = mean
            delta = mean - reference
            print(
                f"   {arm:<12} n={summary['round_us']['n']:<4}"
                f" round_us={mean:9.1f}"
                f" sd={summary['round_us']['stdev']:7.1f}"
                f" delta={delta:9.1f} ({100 * delta / reference:+.2f}%)"
                f" verify_build={summary['verify_build_us']['mean']:9.1f}"
                f" eval_wall={summary['eval_wall_us']['mean']:9.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
