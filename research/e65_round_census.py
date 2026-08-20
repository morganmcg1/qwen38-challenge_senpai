#!/usr/bin/env python3
"""E65 rung 0: per-round cold-first-touch census from one MTP phase trace.

The instrument is the round-latency series, not the leg. A one-off pipeline
compile or allocator growth lands in exactly ONE round and lifts it far above
the spread of its own cell. Verify width and repair state are structurally
confounded, so every cell is (M, repaired) and never M alone.

usage:
  research/e65_round_census.py TRACE [--seed N] [--json OUT] [--label TEXT]
"""
import argparse
import json
import re
import statistics
import sys

ROUND_RE = re.compile(
    r"mtp-trace: round=(?P<round>\d+) d=(?P<d>\d+) acc=(?P<acc>\d+) "
    r"draft_build_us=(?P<draft_build>\d+) verify_build_us=(?P<verify_build>\d+) "
    r"eval_wall_us=(?P<eval_wall>\d+) readout_us=(?P<readout>\d+) "
    r"commit_us=(?P<commit>\d+) upkeep_us=(?P<upkeep>\d+) "
    r"round_us=(?P<round_us>\d+)"
)
BEGIN_RE = re.compile(
    r"mtp-trace: begin seed=(?P<seed>\d+) build_us=(?P<build>\d+) "
    r"eval_wall_us=(?P<eval_wall>\d+)"
)
SEGMENTS = ("draft_build", "verify_build", "eval_wall", "readout", "commit",
            "upkeep")

# `sdpaWidthWallDepthCap` and `segmentedVerifyDepthCap` in
# Qwen36MTPBlockSession.swift. A round wider than the wall is fed to the target
# as <= 5-row sdpa segments.
SDPA_WIDTH_WALL = 5
SEGMENTED_DEPTH_CAP = 8
# scaled_dot_product_attention.cpp:746-748 selects sdpa_vector_2pass when the
# device architecture ends in 'd' or 's' and k.shape(2) >= 1024.
TWO_PASS_KL = 1024


def parse(path):
    """Split one trace file into sessions. A session starts at its `begin`."""
    sessions = []
    current = None
    for line in open(path, "r", errors="replace"):
        b = BEGIN_RE.match(line)
        if b:
            current = {"seed": int(b.group("seed")),
                       "begin_build_us": int(b.group("build")),
                       "begin_eval_wall_us": int(b.group("eval_wall")),
                       "rounds": []}
            sessions.append(current)
            continue
        m = ROUND_RE.match(line)
        if m and current is not None:
            r = {k: int(v) for k, v in m.groupdict().items()}
            current["rounds"].append(r)
    return sessions


def annotate(session, seed_override=None):
    """Reconstruct the structural state each round entered with.

    `base` is the trimmable cache offset at round top, which the session
    asserts as `seedTokenCount + committedTokenCount`. The verify forward
    appends M = d + 1 rows, so the full-attention key length inside that
    round's sdpa is `base + M`.
    """
    seed = seed_override if seed_override is not None else session["seed"]
    base = seed
    prev_rejected = False
    prev_d = None
    seen_widths = set()
    rows = []
    for r in session["rounds"]:
        d = r["d"]
        acc = r["acc"]
        m_width = d + 1
        kl = base + m_width
        row = dict(r)
        row["M"] = m_width
        row["kL_top"] = base
        row["kL_verify"] = kl
        row["repaired"] = prev_rejected
        row["rejected"] = acc < d
        row["cell"] = f"M{m_width}/{'rep' if prev_rejected else 'clean'}"
        row["first_of_width"] = m_width not in seen_widths
        row["crosses_two_pass"] = kl >= TWO_PASS_KL
        row["over_width_wall"] = m_width > SDPA_WIDTH_WALL
        row["at_segmented_cap"] = d >= SEGMENTED_DEPTH_CAP
        seen_widths.add(m_width)
        rows.append(row)
        base += 1 + acc
        prev_rejected = acc < d
        prev_d = d
    return rows


def cell_outliers(rows, k=3.0):
    """Round exceeding its own (M, repaired) median by more than k * IQR."""
    cells = {}
    for row in rows:
        cells.setdefault(row["cell"], []).append(row)
    stats = {}
    for cell, members in cells.items():
        vals = sorted(x["round_us"] for x in members)
        n = len(vals)
        med = statistics.median(vals)
        if n >= 4:
            q1 = statistics.median(vals[: n // 2])
            q3 = statistics.median(vals[(n + 1) // 2:])
            iqr = q3 - q1
        else:
            q1 = q3 = med
            iqr = 0.0
        stats[cell] = {"n": n, "median_us": med, "q1_us": q1, "q3_us": q3,
                       "iqr_us": iqr, "threshold_us": med + k * iqr,
                       "min_us": vals[0], "max_us": vals[-1]}
    outliers = []
    for row in rows:
        st = stats[row["cell"]]
        # A cell with fewer than 4 members has no usable IQR; it is reported
        # as structurally singular rather than silently given a zero bar.
        if st["n"] < 4:
            continue
        if row["round_us"] > st["threshold_us"] and st["iqr_us"] > 0:
            out = dict(row)
            out["cell_median_us"] = st["median_us"]
            out["cell_iqr_us"] = st["iqr_us"]
            out["excess_us"] = row["round_us"] - st["median_us"]
            out["excess_iqr"] = (row["round_us"] - st["median_us"]) / st["iqr_us"]
            outliers.append(out)
    singular = {c: s for c, s in stats.items() if s["n"] < 4}
    return stats, outliers, singular


def structural_events(rows):
    """First occurrence of every structural event the census can name."""
    events = {}

    def first(name, pred):
        for row in rows:
            if pred(row):
                events[name] = row["round"]
                return
        events[name] = None

    first("first_two_pass_kL_ge_1024", lambda r: r["crosses_two_pass"])
    first("first_reject", lambda r: r["rejected"])
    first("first_repair_round", lambda r: r["repaired"])
    first("first_over_width_wall", lambda r: r["over_width_wall"])
    first("first_at_segmented_cap", lambda r: r["at_segmented_cap"])
    first("first_zero_draft", lambda r: r["d"] == 0)
    seen = set()
    for row in rows:
        if row["M"] not in seen:
            events[f"first_width_M{row['M']}"] = row["round"]
            seen.add(row["M"])
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    ap.add_argument("--iqr-k", type=float, default=3.0)
    ap.add_argument("--min-rounds", type=int, default=20,
                    help="ignore sessions shorter than this (warm/reference)")
    args = ap.parse_args()

    sessions = parse(args.trace)
    report = {"trace": args.trace, "label": args.label,
              "iqr_k": args.iqr_k, "sessions": []}
    for index, session in enumerate(sessions):
        rows = annotate(session, args.seed)
        if len(rows) < args.min_rounds:
            continue
        stats, outliers, singular = cell_outliers(rows, args.iqr_k)
        leg_us = sum(r["round_us"] for r in rows)
        excess_us = sum(o["excess_us"] for o in outliers)
        tokens = sum(1 + r["acc"] for r in rows)
        entry = {
            "session_index": index,
            "seed": args.seed if args.seed is not None else session["seed"],
            "begin_build_us": session["begin_build_us"],
            "begin_eval_wall_us": session["begin_eval_wall_us"],
            "rounds": len(rows),
            "tokens": tokens,
            "leg_round_us": leg_us,
            "kL_first": rows[0]["kL_verify"],
            "kL_last": rows[-1]["kL_verify"],
            "kL_max": max(r["kL_verify"] for r in rows),
            "rounds_at_or_above_1024": sum(
                1 for r in rows if r["crosses_two_pass"]),
            "serial_like": all(r["d"] == 0 for r in rows),
            "cells": stats,
            "singular_cells": singular,
            "structural_events": structural_events(rows),
            "outliers": outliers,
            "outlier_excess_us": excess_us,
            "outlier_excess_pct_of_leg": 100.0 * excess_us / leg_us if leg_us else 0.0,
            "segment_totals_us": {
                s: sum(r[s] for r in rows) for s in SEGMENTS},
            "round_table": rows,
        }
        report["sessions"].append(entry)

    for entry in report["sessions"]:
        print(f"=== session {entry['session_index']} "
              f"({'serial' if entry['serial_like'] else 'mtp'}) "
              f"rounds={entry['rounds']} tokens={entry['tokens']} "
              f"leg={entry['leg_round_us'] / 1e6:.3f}s "
              f"kL {entry['kL_first']}..{entry['kL_max']} "
              f"rounds_kL>=1024={entry['rounds_at_or_above_1024']}")
        for cell, st in sorted(entry["cells"].items()):
            print(f"  cell {cell:14s} n={st['n']:4d} "
                  f"med={st['median_us'] / 1000:8.2f}ms "
                  f"iqr={st['iqr_us'] / 1000:7.2f}ms "
                  f"max={st['max_us'] / 1000:8.2f}ms")
        print(f"  outliers ({len(entry['outliers'])}):")
        for o in entry["outliers"]:
            tags = [k for k in ("crosses_two_pass", "first_of_width",
                                "repaired", "rejected", "over_width_wall",
                                "at_segmented_cap") if o[k]]
            print(f"    round={o['round']:4d} cell={o['cell']:12s} "
                  f"kL={o['kL_verify']:5d} "
                  f"t={o['round_us'] / 1000:8.2f}ms "
                  f"med={o['cell_median_us'] / 1000:8.2f}ms "
                  f"excess={o['excess_us'] / 1000:7.2f}ms "
                  f"({o['excess_iqr']:.1f} IQR) "
                  f"eval={o['eval_wall'] / 1000:.2f}ms "
                  f"vbuild={o['verify_build'] / 1000:.2f}ms "
                  f"tags={','.join(tags) or '-'}")
        print(f"  outlier excess = {entry['outlier_excess_us'] / 1000:.2f} ms "
              f"= {entry['outlier_excess_pct_of_leg']:.4f} % of the round series")
        print(f"  structural events: {json.dumps(entry['structural_events'])}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
