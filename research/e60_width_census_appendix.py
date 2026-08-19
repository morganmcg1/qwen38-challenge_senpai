#!/usr/bin/env python3
"""Per-width dispatch census d(M) for M = 1..9, split by kernel family.

The advisor asked for this as a formatting change over data E58 already holds.
`dispatches_per_round` is a mean over a leg's own width mix, so it cannot be
carried to a leg that runs a different mix. This appendix reports the per-width
table underneath that mean.

Three properties of the source data bound what it can say, and the script
proves each from the files rather than asserting it.

* Command-buffer geometry does not change dispatch counts per round. Three
  E58 census legs share one 64-token window and differ only in the
  command-buffer setting, so they isolate that axis exactly.
* Window length DOES change d(M) at a fixed width. The 512-token census and
  the 64-token censuses disagree at several widths, so d(M) is not a pure
  function of width, and the two windows are reported separately.
* Only widths the candidate actually chose appear. Width 3 was never
  scheduled in any E58 leg, so d(3) is unknown and is printed as unknown.

Round width M counts the primary token plus its drafts, so M = 1 + drafts and
the eight-draft ceiling makes M = 9 the maximum. A leg's mean draft length is
therefore its mean width minus one.

usage: research/e60_width_census_appendix.py [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

ARTIFACTS = pathlib.Path("research/e58-artifacts")


def load() -> list[dict]:
    censuses = []
    for path in sorted(glob.glob(str(ARTIFACTS / "*census*.json"))):
        payload = json.load(open(path))
        probe = next(
            (
                event
                for event in payload.get("install_events", [])
                if event.get("event") == "buffer_limit_probe"
            ),
            None,
        )
        censuses.append(
            {
                "name": pathlib.Path(path).name,
                "geometry": (
                    f"{probe['requested_mb']} MiB / {probe['requested_ops']} ops"
                    if probe
                    else "default (128 MiB / 64 ops, trusted low profile)"
                ),
                "legs": {leg["leg"]: leg for leg in payload["legs"]},
            }
        )
    return censuses


def window_of(census: dict) -> int:
    return census["legs"]["serial(depth0)"]["rounds"]


SEED_TOKENS = 512
KL_BUCKET = 128


def load_rounds() -> list[dict]:
    """Per-round records from the raw census logs, keyed by (M, kL).

    kL is the key length the target sees while it verifies the round:
    seed + tokens already committed + the M rows under verification. A round
    record carries its own width and accepted count, so the committed prefix is
    a running sum and needs no extra measurement.
    """
    rounds = []
    for path in sorted(glob.glob("research/out/e58-*/census.jsonl")):
        by_pid: dict[int, list[dict]] = {}
        for line in open(path):
            event = json.loads(line)
            if event.get("event") == "round":
                by_pid.setdefault(event["pid"], []).append(event)
        for events in by_pid.values():
            events.sort(key=lambda e: e["round"])
            leg = "serial(depth0)" if max(e["width"] for e in events) == 1 else "candidate(mtp)"
            committed = 0
            for index, event in enumerate(events):
                width = event["width"]
                previous = events[index - 1] if index else None
                rounds.append(
                    {
                        "source": pathlib.Path(path).parent.name,
                        "leg": leg,
                        "window": sum(1 + e["accepted"] for e in events),
                        "round": event["round"],
                        "M": width,
                        "committed": committed,
                        "kL": SEED_TOKENS + committed + width,
                        "rejected_in_previous_round": (
                            None
                            if previous is None
                            else (previous["width"] - 1) - previous["accepted"]
                        ),
                        "dispatches": sum(
                            phase["dispatches"] for phase in event["phases"].values()
                        ),
                        "families": {
                            family: count
                            for phase in event["phases"].values()
                            for family, count in phase["kernels"].items()
                        },
                    }
                )
                committed += 1 + event["accepted"]
    return rounds


def report_rollback(rounds: list[dict]) -> dict:
    """Split d at fixed M by the previous round's rejected-draft count.

    M is set by the previous round's accepted count, so a round that follows a
    rejection is systematically narrow AND carries that rejection's recurrent
    state repair. Width and repair work are therefore confounded in any table
    keyed on M alone.
    """
    candidate = [r for r in rounds if r["leg"] == "candidate(mtp)"]
    table: dict[tuple[int, int], dict] = {}
    for record in candidate:
        rejected = record["rejected_in_previous_round"]
        if rejected is None:
            continue
        entry = table.setdefault(
            (record["M"], rejected), {"rounds": 0, "dispatches": 0, "gdn": 0}
        )
        entry["rounds"] += 1
        entry["dispatches"] += record["dispatches"]
        entry["gdn"] += sum(
            count
            for family, count in record["families"].items()
            if "gated_delta" in family or "gdn" in family
        )

    print()
    print("d split by the PREVIOUS round's rejected drafts, candidate leg")
    print("M | rejected in previous round | rounds | d | gated-DeltaNet dispatches")
    for (width, rejected) in sorted(table):
        entry = table[(width, rejected)]
        print(
            f"{width} | {rejected} | {entry['rounds']} | "
            f"{entry['dispatches'] / entry['rounds']:.2f} | "
            f"{entry['gdn'] / entry['rounds']:.2f}"
        )
    return {
        f"M{width}_rej{rejected}": {
            "M": width,
            "rejected_in_previous_round": rejected,
            "rounds": entry["rounds"],
            "dispatches_per_round": entry["dispatches"] / entry["rounds"],
            "gdn_dispatches_per_round": entry["gdn"] / entry["rounds"],
        }
        for (width, rejected), entry in sorted(table.items())
    }


def kL_table(rounds: list[dict]) -> dict[tuple[int, int], dict]:
    table: dict[tuple[int, int], dict] = {}
    for record in rounds:
        key = (record["M"], record["kL"] // KL_BUCKET * KL_BUCKET)
        entry = table.setdefault(key, {"rounds": 0, "dispatches": 0, "windows": set()})
        entry["rounds"] += 1
        entry["dispatches"] += record["dispatches"]
        entry["windows"].add(record["window"])
    for entry in table.values():
        entry["d"] = entry["dispatches"] / entry["rounds"]
        entry["windows"] = sorted(entry["windows"])
    return table


def report_kL(rounds: list[dict]) -> dict:
    """Test whether d(M) stops disagreeing across windows once kL is held fixed."""
    candidate = [r for r in rounds if r["leg"] == "candidate(mtp)"]
    table = kL_table(candidate)

    print()
    print("d(M, kL_bucket), candidate leg, all E58 censuses pooled")
    print(f"kL = {SEED_TOKENS} + tokensCommitted + M, bucketed at {KL_BUCKET}")
    print("M | kL bucket | rounds | d | source windows")
    for (width, bucket) in sorted(table):
        entry = table[(width, bucket)]
        print(
            f"{width} | {bucket}-{bucket + KL_BUCKET - 1} | {entry['rounds']} | "
            f"{entry['d']:.2f} | {entry['windows']}"
        )

    print()
    print("Widths whose whole-leg d(M) disagreed across windows, re-read at fixed kL:")
    resolved = {}
    for width in sorted({w for w, _ in table}):
        buckets = {b: table[(width, b)] for w, b in table if w == width}
        if len(buckets) < 2:
            continue
        values = [entry["d"] for entry in buckets.values()]
        resolved[width] = {
            "buckets": {
                str(b): {"d": e["d"], "rounds": e["rounds"], "windows": e["windows"]}
                for b, e in buckets.items()
            },
            "spread": max(values) - min(values),
        }
        print(f"  M={width}: spread across kL buckets {max(values) - min(values):+.2f}")
        for bucket in sorted(buckets):
            entry = buckets[bucket]
            print(
                f"    kL {bucket}-{bucket + KL_BUCKET - 1}: d={entry['d']:.2f} "
                f"over {entry['rounds']} rounds, windows {entry['windows']}"
            )

    crossings = sorted({r["kL"] for r in candidate if r["kL"] >= 1024})
    print()
    print(
        f"rounds at kL >= 1024 (the two-pass SDPA predicate): "
        f"{len(crossings)} distinct kL values, "
        f"{sum(1 for r in candidate if r['kL'] >= 1024)} of {len(candidate)} rounds"
    )
    return {
        "bucket_width": KL_BUCKET,
        "table": {
            f"M{width}_kL{bucket}": {
                "M": width,
                "kL_bucket_start": bucket,
                "rounds": entry["rounds"],
                "dispatches_per_round": entry["d"],
                "source_windows": entry["windows"],
            }
            for (width, bucket), entry in sorted(table.items())
        },
        "cross_window_widths": resolved,
        "rounds_at_or_above_1024": sum(1 for r in candidate if r["kL"] >= 1024),
        "candidate_rounds": len(candidate),
    }


def prove_geometry_invariance(censuses: list[dict]) -> dict:
    """Compare only legs that share a window, so geometry is the sole variable."""
    by_window: dict[int, list[dict]] = {}
    for census in censuses:
        by_window.setdefault(window_of(census), []).append(census)

    report = {"groups": [], "invariant": True}
    for window, group in sorted(by_window.items()):
        if len(group) < 2:
            continue
        entry = {
            "window_tokens": window,
            "settings": [c["geometry"] for c in group],
            "dispatches_per_round": {},
            "dispatches_per_commit": {},
            "invariant": True,
        }
        for leg_name in ("serial(depth0)", "candidate(mtp)"):
            per_round = [c["legs"][leg_name]["dispatches_per_round_mean"] for c in group]
            per_commit = [c["legs"][leg_name]["dispatches_per_commit"] for c in group]
            entry["dispatches_per_round"][leg_name] = per_round
            entry["dispatches_per_commit"][leg_name] = per_commit
            if len(set(per_round)) != 1:
                entry["invariant"] = False
                report["invariant"] = False
        report["groups"].append(entry)
    return report


def width_table(leg: dict) -> dict[int, dict]:
    return {
        int(width): {
            "rounds": entry["rounds"],
            "dispatches_per_round": entry["dispatches_per_round"],
            "families_per_round": entry["families_per_round"],
        }
        for width, entry in leg["widths"].items()
    }


def merge_candidate_widths(censuses: list[dict], window: int) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for census in censuses:
        if window_of(census) != window:
            continue
        for width, entry in width_table(census["legs"]["candidate(mtp)"]).items():
            merged.setdefault(width, entry)
    return merged


def print_table(title: str, table: dict[int, dict], families: list[str]) -> None:
    print()
    print(title)
    print(" | ".join(["M", "rounds", "d(M)"] + families))
    for width in sorted(table):
        entry = table[width]
        row = [str(width), str(entry["rounds"]), f"{entry['dispatches_per_round']:.2f}"]
        row += [f"{entry['families_per_round'].get(f, 0.0):.2f}" for f in families]
        print(" | ".join(row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    censuses = load()
    invariance = prove_geometry_invariance(censuses)

    long_census = max(censuses, key=window_of)
    long_serial = width_table(long_census["legs"]["serial(depth0)"])
    long_candidate = width_table(long_census["legs"]["candidate(mtp)"])
    short_window = min(window_of(c) for c in censuses)
    short_candidate = merge_candidate_widths(censuses, short_window)

    families = sorted(
        {
            family
            for table in (long_serial, long_candidate, short_candidate)
            for entry in table.values()
            for family in entry["families_per_round"]
        }
    )

    print("E58 per-width dispatch census d(M), M = 1..9")
    print("M counts the primary token plus its drafts, so M = 1 + drafts.")
    print()
    print("Command-buffer geometry axis (one shared window, setting is the only variable):")
    for group in invariance["groups"]:
        print(f"  window {group['window_tokens']} tokens, settings {group['settings']}")
        for leg_name, values in group["dispatches_per_round"].items():
            print(f"    {leg_name:16s} dispatches per round  {values}")
        for leg_name, values in group["dispatches_per_commit"].items():
            print(f"    {leg_name:16s} dispatches per commit {values}")
        print(f"    invariant: {group['invariant']}")

    observed = set(long_serial) | set(long_candidate) | set(short_candidate)
    missing = [m for m in range(1, 10) if m not in observed]
    print()
    print(f"UNOBSERVED widths (never scheduled in any E58 leg): {missing or 'none'}")

    print_table(
        f"d(M) from the {window_of(long_census)}-token census "
        f"({long_census['geometry']}), serial leg:",
        long_serial,
        families,
    )
    print_table(
        f"d(M) from the {window_of(long_census)}-token census, candidate leg:",
        long_candidate,
        families,
    )
    print_table(
        f"d(M) from the {short_window}-token censuses, candidate leg:",
        short_candidate,
        families,
    )

    print()
    print("Same width, different window (d(M) is NOT a pure function of width):")
    for width in sorted(set(long_candidate) & set(short_candidate)):
        long_value = long_candidate[width]["dispatches_per_round"]
        short_value = short_candidate[width]["dispatches_per_round"]
        print(
            f"  M={width}: {window_of(long_census)}-token {long_value:.2f}"
            f"  vs {short_window}-token {short_value:.2f}"
            f"  delta {long_value - short_value:+.2f}"
        )

    rounds = sum(e["rounds"] for e in long_candidate.values())
    mean_width = sum(w * e["rounds"] for w, e in long_candidate.items()) / rounds
    mean_dispatches = (
        sum(e["dispatches_per_round"] * e["rounds"] for e in long_candidate.values())
        / rounds
    )
    recorded = long_census["legs"]["candidate(mtp)"]["dispatches_per_round_mean"]
    print()
    print(f"{window_of(long_census)}-token candidate leg: {rounds} rounds")
    print(f"  round-weighted mean width M      = {mean_width:.4f}")
    print(f"  implied mean draft length M - 1  = {mean_width - 1:.4f}")
    print(f"  round-weighted mean d(M)         = {mean_dispatches:.2f}")
    print(f"  leg's own recorded mean          = {recorded:.2f}")

    per_round = load_rounds()
    kL = report_kL(per_round)
    rollback = report_rollback(per_round)

    if args.json:
        out = {
            "geometry_invariance": invariance,
            "kL_census": kL,
            "rollback_split": rollback,
            "unobserved_widths": missing,
            "families": families,
            "long_window_tokens": window_of(long_census),
            "short_window_tokens": short_window,
            "serial_long_window": {str(k): v for k, v in long_serial.items()},
            "candidate_long_window": {str(k): v for k, v in long_candidate.items()},
            "candidate_short_window": {str(k): v for k, v in short_candidate.items()},
            "candidate_long_window_mean_width": mean_width,
            "candidate_long_window_mean_draft_length": mean_width - 1,
            "candidate_long_window_mean_dispatches": mean_dispatches,
            "candidate_long_window_recorded_mean": recorded,
            "censuses": [
                {"name": c["name"], "geometry": c["geometry"], "window": window_of(c)}
                for c in censuses
            ],
        }
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
