#!/usr/bin/env python3
"""E174 step-1 dedup census report.

How many standalone chunk-sum table fills in one MTP round rebuild a table an
earlier fill in the SAME round already built?

`xs_fill` and `xs_uniq` are cumulative process counters, so the per-round
surface is the difference between consecutive round lines. Round 0 is skipped:
it carries whatever the untimed warm left in the counters.

`xs_dedup` is not cumulative. It describes exactly the round its line reports,
because the block session clears the census immediately after writing the line.

The report prices the duplicates against the E174 screen's per-dispatch
bracket. That bracket is a LOCAL, matched, absolute candidate-MTP coefficient.
It is not a ranked coefficient and it is not converted into one here.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics

OUT = pathlib.Path(__file__).resolve().parent / "out"

# E174 screen, W&B run 7ueick4f: one standalone fill dispatch plus its host
# kernel record costs this many microseconds of candidate MTP wall time. The
# bounds cross (5.505 us/cell from `o - s`, 3.820 us/cell from `f - r`), so the
# screen reports a bracket and never a point.
US_PER_DISPATCH_LO = 4.0
US_PER_DISPATCH_HI = 5.3


def trace_rounds(path: pathlib.Path) -> list[dict[str, object]]:
    seen: list[dict[str, object]] = []
    if not path.exists():
        return seen
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        row: dict[str, object] = {}
        for key in ("round", "xs_hit", "xs_fill", "xs_uniq", "round_us"):
            m = re.search(rf"\b{key}=(\d+)", line)
            if m:
                row[key] = int(m.group(1))
        m = re.search(r"\bxs_dedup=(\S+)", line)
        if m:
            row["xs_dedup"] = m.group(1)
        if len(row) == 6:
            seen.append(row)
    return seen


def parse_dedup(text: str) -> dict[str, object] | None:
    """`selfcheck:calls/distinct;KxM:calls/uniq,...;Nx=count,...`."""
    if text == "off":
        return None
    head, _, rest = text.partition(":")
    totals, _, rest = rest.partition(";")
    shapes_text, _, bars_text = rest.partition(";")
    calls, _, distinct = totals.partition("/")
    shapes = {}
    for item in filter(None, shapes_text.split(",")):
        name, _, counts = item.partition(":")
        c, _, u = counts.partition("/")
        shapes[name] = (int(c), int(u))
    bars = {}
    for item in filter(None, bars_text.split(",")):
        name, _, count = item.partition("=")
        bars[int(name.rstrip("x"))] = int(count)
    return {
        "selfcheck": head,
        "calls": int(calls),
        "distinct": int(distinct),
        "shapes": shapes,
        "histogram": bars,
    }


def arm_report(tag: str) -> dict[str, object]:
    out = OUT / tag
    seen = trace_rounds(out / "trace.txt")
    report: dict[str, object] = {"tag": tag, "rounds_traced": len(seen)}
    if len(seen) < 3:
        report["error"] = "too few traced rounds"
        return report

    fills = [b["xs_fill"] - a["xs_fill"] for a, b in zip(seen, seen[1:])]
    uniq = [b["xs_uniq"] - a["xs_uniq"] for a, b in zip(seen, seen[1:])]
    hits = [b["xs_hit"] - a["xs_hit"] for a, b in zip(seen, seen[1:])]
    round_us = [b["round_us"] for b in seen[1:]]
    dups = [f - u for f, u in zip(fills, uniq)]

    # `xs_dedup` describes its own line, so it is read from the same rounds the
    # differences describe rather than differenced.
    parsed = [parse_dedup(str(b["xs_dedup"])) for b in seen[1:]]
    parsed = [p for p in parsed if p is not None]
    selfchecks = sorted({str(p["selfcheck"]) for p in parsed})

    # Every round's counter difference must agree with that round's own
    # self-describing summary. A disagreement means the census did not clear
    # between rounds and nothing below is readable.
    agree = all(
        p["calls"] == f and p["distinct"] == u
        for p, f, u in zip(parsed, fills, uniq)
    )

    shape_calls: collections.Counter = collections.Counter()
    shape_uniq: collections.Counter = collections.Counter()
    histogram: collections.Counter = collections.Counter()
    for p in parsed:
        for name, (c, u) in p["shapes"].items():  # type: ignore[union-attr]
            shape_calls[name] += c
            shape_uniq[name] += u
        for n, c in p["histogram"].items():  # type: ignore[union-attr]
            histogram[n] += c

    fill_med = statistics.median(fills)
    dup_med = statistics.median(dups)
    round_med = statistics.median(round_us)
    fraction = dup_med / fill_med if fill_med else 0.0

    report.update(
        {
            "rounds_differenced": len(fills),
            "selfcheck_values": selfchecks,
            "selfcheck_ok": selfchecks == ["ok"],
            "counter_summary_agree": agree,
            "hit_per_round_median": statistics.median(hits),
            "fill_per_round_median": fill_med,
            "uniq_per_round_median": statistics.median(uniq),
            "duplicate_per_round_median": dup_med,
            "duplicate_fraction_of_fills": round(fraction, 6),
            "fill_per_round_set": sorted(set(fills)),
            "duplicate_per_round_set": sorted(set(dups)),
            "round_us_median": round_med,
            "shape_calls_per_round": {
                k: round(v / len(parsed), 3) for k, v in sorted(shape_calls.items())
            },
            "shape_uniq_per_round": {
                k: round(v / len(parsed), 3) for k, v in sorted(shape_uniq.items())
            },
            "shape_duplicates_per_round": {
                k: round((shape_calls[k] - shape_uniq[k]) / len(parsed), 3)
                for k in sorted(shape_calls)
                if shape_calls[k] != shape_uniq[k]
            },
            "duplicates_per_activation_histogram": {
                f"{k}x": round(v / len(parsed), 3) for k, v in sorted(histogram.items())
            },
            "local_value_us_per_round_lo": round(dup_med * US_PER_DISPATCH_LO, 2),
            "local_value_us_per_round_hi": round(dup_med * US_PER_DISPATCH_HI, 2),
            "local_value_pct_of_round_lo": round(
                100.0 * dup_med * US_PER_DISPATCH_LO / round_med, 4
            )
            if round_med
            else None,
            "local_value_pct_of_round_hi": round(
                100.0 * dup_med * US_PER_DISPATCH_HI / round_med, 4
            )
            if round_med
            else None,
        }
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="d1")
    args = ap.parse_args()

    arms = {arm: arm_report(f"e174dedup{args.label}{arm}") for arm in ("s", "o")}

    gate_ok = all(
        a.get("selfcheck_ok") and a.get("counter_summary_agree") for a in arms.values()
    )
    # Arm `o` is arm `s`'s positive control on the switch: the sidecar-off arm
    # must serve no cell and must send the whole table-paying surface through
    # the standalone fill.
    switch_ok = (
        arms["s"].get("hit_per_round_median", 0) > 0
        and arms["o"].get("hit_per_round_median", 1) == 0
        and arms["o"].get("fill_per_round_median", 0)
        > arms["s"].get("fill_per_round_median", 0)
    )

    report = {
        "experiment": "E174-step1-dedup-census",
        "timed": False,
        "gate_qualified_for_timing": False,
        "harness": "local",
        "us_per_dispatch_bracket": [US_PER_DISPATCH_LO, US_PER_DISPATCH_HI],
        "us_per_dispatch_source": "E174 screen, wandb run 7ueick4f, local matched",
        "instrument_selfcheck_ok": gate_ok,
        "arm_switch_control_ok": switch_ok,
        "arms": arms,
        "decision_threshold_duplicate_fraction": 0.20,
    }
    for arm, a in arms.items():
        f = a.get("duplicate_fraction_of_fills")
        if f is not None:
            report[f"arm_{arm}_meets_20pct"] = f >= 0.20

    path = OUT / f"e174-dedup-census-{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
