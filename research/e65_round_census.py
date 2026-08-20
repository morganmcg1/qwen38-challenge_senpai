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
                       "min_us": vals[0], "max_us": vals[-1],
                       "segment_median_us": {
                           s: statistics.median([x[s] for x in members])
                           for s in SEGMENTS}}
    # Conservative pooled bar for cells too small to carry their own IQR: the
    # largest same-cell spread measured anywhere in this session.
    resolved = [s["iqr_us"] for s in stats.values() if s["n"] >= 4]
    pooled_iqr = max(resolved) if resolved else 0.0

    outliers = []
    small_cell_outliers = []
    for row in rows:
        st = stats[row["cell"]]
        out = dict(row)
        out["cell_median_us"] = st["median_us"]
        out["excess_us"] = row["round_us"] - st["median_us"]
        out["segment_excess_us"] = {
            s: row[s] - st["segment_median_us"][s] for s in SEGMENTS}
        # A cell with fewer than 4 members has no usable IQR of its own. The
        # preregistered rule stays exactly as written; such rounds are scored
        # on a separate, clearly labelled pooled bar so they are never silently
        # dropped.
        if st["n"] < 4:
            if pooled_iqr > 0 and out["excess_us"] > k * pooled_iqr:
                out["cell_iqr_us"] = pooled_iqr
                out["excess_iqr"] = out["excess_us"] / pooled_iqr
                out["bar"] = "pooled"
                small_cell_outliers.append(out)
            continue
        if row["round_us"] > st["threshold_us"] and st["iqr_us"] > 0:
            out["cell_iqr_us"] = st["iqr_us"]
            out["excess_iqr"] = out["excess_us"] / st["iqr_us"]
            out["bar"] = "cell"
            outliers.append(out)
    singular = {c: s for c, s in stats.items() if s["n"] < 4}
    return stats, outliers, singular, small_cell_outliers, pooled_iqr


def crossing_probe(rows):
    """Compare every kL >= 1024 round with same-width rounds below the cross.

    At exactly 512 decode tokens the crossing round is also the last round and
    can be the only member of its own (M, repaired) cell, so the IQR rule has
    no bar to apply. This falls back to the widest legitimate comparison: all
    rounds of the SAME verify width that stayed below the threshold. Repair
    state is reported per comparator so the confound stays visible.
    """
    probes = []
    for row in rows:
        if not row["crosses_two_pass"]:
            continue
        peers = [r for r in rows
                 if r["M"] == row["M"] and not r["crosses_two_pass"]]
        if not peers:
            probes.append({"round": row["round"], "M": row["M"],
                           "kL_verify": row["kL_verify"],
                           "round_us": row["round_us"], "peers": 0})
            continue
        vals = sorted(p["round_us"] for p in peers)
        med = statistics.median(vals)
        spread = vals[-1] - vals[0]
        probes.append({
            "round": row["round"],
            "M": row["M"],
            "repaired": row["repaired"],
            "kL_verify": row["kL_verify"],
            "is_last_round": row is rows[-1],
            "round_us": row["round_us"],
            "peers": len(peers),
            "peer_repaired": sorted({p["repaired"] for p in peers}),
            "peer_median_us": med,
            "peer_min_us": vals[0],
            "peer_max_us": vals[-1],
            "peer_spread_us": spread,
            "excess_us": row["round_us"] - med,
            "excess_in_peer_spreads": (
                (row["round_us"] - med) / spread if spread else None),
            "segment_excess_us": {
                s: row[s] - statistics.median([p[s] for p in peers])
                for s in SEGMENTS},
        })
    return probes


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
        (stats, outliers, singular,
         small_cell_outliers, pooled_iqr) = cell_outliers(rows, args.iqr_k)
        leg_us = sum(r["round_us"] for r in rows)
        # The scored leg is seed processing plus decode, so the denominator for
        # a "% of the leg" claim includes `begin`, not the round series alone.
        timed_leg_us = (leg_us + session["begin_build_us"]
                        + session["begin_eval_wall_us"])
        excess_us = sum(o["excess_us"] for o in outliers)
        small_excess_us = sum(o["excess_us"] for o in small_cell_outliers)
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
            "crossing_probe": crossing_probe(rows),
            "outliers": outliers,
            "outlier_excess_us": excess_us,
            "pooled_iqr_us": pooled_iqr,
            "small_cell_outliers": small_cell_outliers,
            "small_cell_excess_us": small_excess_us,
            "timed_leg_us": timed_leg_us,
            "outlier_excess_pct_of_rounds": (
                100.0 * excess_us / leg_us if leg_us else 0.0),
            "outlier_excess_pct_of_leg": (
                100.0 * excess_us / timed_leg_us if timed_leg_us else 0.0),
            "combined_excess_pct_of_leg": (
                100.0 * (excess_us + small_excess_us) / timed_leg_us
                if timed_leg_us else 0.0),
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
            print("      segment excess ms: " + ", ".join(
                f"{s}{o['segment_excess_us'][s] / 1000:+.2f}"
                for s in SEGMENTS))
        print(f"  outlier excess = {entry['outlier_excess_us'] / 1000:.2f} ms "
              f"= {entry['outlier_excess_pct_of_rounds']:.4f} % of the round "
              f"series = {entry['outlier_excess_pct_of_leg']:.4f} % of the "
              f"timed leg ({entry['timed_leg_us'] / 1e6:.3f} s incl. begin)")
        print(f"  small-cell channel (pooled bar "
              f"{entry['pooled_iqr_us'] / 1000:.2f}ms, "
              f"{len(entry['small_cell_outliers'])}):")
        for o in entry["small_cell_outliers"]:
            seg = ", ".join(f"{s}{o['segment_excess_us'][s] / 1000:+.2f}"
                            for s in SEGMENTS)
            print(f"    round={o['round']:4d} cell={o['cell']:12s} "
                  f"kL={o['kL_verify']:5d} "
                  f"t={o['round_us'] / 1000:8.2f}ms "
                  f"med={o['cell_median_us'] / 1000:8.2f}ms "
                  f"excess={o['excess_us'] / 1000:7.2f}ms "
                  f"({o['excess_iqr']:.1f} pooled spreads) "
                  f"first_round={o['round'] == 1}")
            print(f"      segment excess ms: {seg}")
        print(f"  combined excess (cell + pooled bars) = "
              f"{(entry['outlier_excess_us'] + entry['small_cell_excess_us']) / 1000:.2f} ms "
              f"= {entry['combined_excess_pct_of_leg']:.4f} % of the timed leg")
        for p in entry["crossing_probe"]:
            if p["peers"] == 0:
                print(f"  kL>=1024 round={p['round']} M={p['M']}: no same-width "
                      f"peer below the threshold")
                continue
            seg = ", ".join(
                f"{k}{v / 1000:+.2f}" for k, v in p["segment_excess_us"].items())
            print(f"  kL>=1024 round={p['round']} M={p['M']} "
                  f"repaired={p['repaired']} last={p['is_last_round']} "
                  f"t={p['round_us'] / 1000:.2f}ms vs same-width peers "
                  f"n={p['peers']} med={p['peer_median_us'] / 1000:.2f}ms "
                  f"[{p['peer_min_us'] / 1000:.2f},{p['peer_max_us'] / 1000:.2f}] "
                  f"excess={p['excess_us'] / 1000:+.2f}ms "
                  f"= {p['excess_in_peer_spreads']:.1f} peer spreads "
                  f"= {100.0 * p['excess_us'] / entry['timed_leg_us']:.4f} % of leg")
            print(f"      segment excess ms: {seg}")
        print(f"  structural events: {json.dumps(entry['structural_events'])}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
