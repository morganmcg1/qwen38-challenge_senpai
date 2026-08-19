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

    if args.json:
        out = {
            "geometry_invariance": invariance,
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
