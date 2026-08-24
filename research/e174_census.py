#!/usr/bin/env python3
"""E174 step-0 census report: the live chunk-sum fill surface, per round.

`xs_hit` and `xs_fill` are cumulative process counters, so the per-round
surface is the difference between consecutive round lines. Round 0 is skipped:
it carries whatever the untimed warm left in the counters.

Each arm is the other arm's positive control. `on` must show hits and a partial
fill count; `off` must show zero hits and the whole table-paying surface. An
arm that passes both expectations means the switch never reached the worker.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

OUT = pathlib.Path(__file__).resolve().parent / "out"


def rounds(path: pathlib.Path) -> list[tuple[int, int, int]]:
    """`(round, xs_hit, xs_fill)` for every traced round, in order."""
    seen: list[tuple[int, int, int]] = []
    if not path.exists():
        return seen
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        got = {}
        for key in ("round", "xs_hit", "xs_fill"):
            m = re.search(rf"\b{key}=(\d+)", line)
            if m:
                got[key] = int(m.group(1))
        if len(got) == 3:
            seen.append((got["round"], got["xs_hit"], got["xs_fill"]))
    return seen


def per_round(path: pathlib.Path) -> dict[str, object]:
    seen = rounds(path)
    if len(seen) < 3:
        return {"rounds_traced": len(seen)}
    hits = [b[1] - a[1] for a, b in zip(seen, seen[1:])]
    fills = [b[2] - a[2] for a, b in zip(seen, seen[1:])]
    return {
        "rounds_traced": len(seen),
        "rounds_differenced": len(hits),
        "hit_per_round_median": statistics.median(hits),
        "fill_per_round_median": statistics.median(fills),
        "hit_per_round_set": sorted(set(hits)),
        "fill_per_round_set": sorted(set(fills)),
        "table_paying_cells_median": statistics.median(
            [h + f for h, f in zip(hits, fills)]
        ),
    }


def meta(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


def score(path: pathlib.Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="c1")
    args = ap.parse_args()

    report: dict[str, object] = {
        "experiment": "e174-xsums-epilogue-extension",
        "step": "step-0 fill census",
        "harness": "local",
        "gate_qualified_for_timing": False,
        "arms": {},
    }

    for arm in ("on", "off"):
        out = OUT / f"e174{args.label}{arm}"
        m = meta(out / "meta.txt")
        s = score(out / "score.json")
        metrics = s.get("metrics", {}) if isinstance(s, dict) else {}
        report["arms"][arm] = {
            "tag": f"e174{args.label}{arm}",
            "census": per_round(out / "trace.txt"),
            "base_sha": m.get("base_sha"),
            "worker_sha256": m.get("worker_sha256"),
            "metallib_source_fingerprint": m.get("metallib_source_fingerprint"),
            "host": m.get("host"),
            "chip": m.get("chip"),
            "tokens": m.get("tokens"),
            "dirty_candidate_paths": m.get("dirty_candidate_paths"),
            "gpu_temp_entry_c": m.get("gpu_temp_entry_c"),
            "all_tokens_matched": metrics.get("all_tokens_matched"),
            "effective_mean_draft_len": metrics.get("effective_mean_draft_len"),
            "accepted_draft_rate": metrics.get("accepted_draft_rate"),
            "mtp_seconds_per_token": metrics.get("mtp_seconds_per_token"),
        }

    on = report["arms"]["on"]["census"]
    off = report["arms"]["off"]["census"]
    on_hits = on.get("hit_per_round_median")
    on_fills = on.get("fill_per_round_median")
    off_hits = off.get("hit_per_round_median")
    off_fills = off.get("fill_per_round_median")

    checks = {
        "on_arm_publishes": on_hits is not None and on_hits > 0,
        "on_arm_has_unserved_cells": on_fills is not None and on_fills > 0,
        "off_arm_publishes_nothing": off_hits == 0,
        "off_arm_fills_whole_surface": (
            off_fills is not None
            and on_hits is not None
            and on_fills is not None
            and off_fills == on_hits + on_fills
        ),
        # The control: an arm that satisfies the other arm's expectation means
        # the switch never reached the worker.
        "arms_are_distinguishable": on_hits != off_hits and on_fills != off_fills,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    if on_hits is not None and on_fills is not None:
        report["unserved_share_of_table_paying_surface"] = round(
            on_fills / (on_hits + on_fills), 6
        )

    path = OUT / f"e174-census-{args.label}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
