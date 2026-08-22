#!/usr/bin/env python3
"""E130 rung 0b: how often the wide-QMV entry point is actually dispatched.

Reads the E58/E80 dispatch-census JSONL a traced leg writes and answers the
three questions E130-F3 item 7 and E130-F4 ask, in the SHIPPED configuration
with Route B on:

  1. dispatches of `affine_qmv_fast_bfloat16_t_64_4_false` per decode round,
     split by `ntg.x`;
  2. the GPU-time share those dispatches hold, split by phase so target rounds
     and MTP head projections are separate;
  3. whether `ntg.x == 5` ever occurs with `n >= 4096`.

`quantized.cpp:254` sets `grid_dims(M, ceil(N / bn), B)` for the qmv/qmv_fast
family, so the census `grid=WxHxD` string carries `ntg.x == W == M` directly
and `H` recovers the output width `N` up to the `bn` tile size.

    python3 research/e130_rung0b.py CENSUS.jsonl --out research/e130-artifacts/rung0b.json

The census is a COUNTING instrument, not a timing instrument: it locks on every
dispatch, bind and barrier, so its host phases are inflated. Only the counts and
the GPU-side `gpu_ns` buckets are used here, and no leg time from this run is
ever reported as a candidate measurement.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys

KERNEL = "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0"
SHAPE_RE = re.compile(
    r"^(?P<kernel>\S+) grid=(?P<w>\d+)x(?P<h>\d+)x(?P<d>\d+)(?: tg=\S+)?$")
SIG_ITEM_RE = re.compile(r"^(?P<shape>.+?)\*(?P<count>\d+)$")

# quantized.cpp:252-254 sets `bn = 8` and `grid_dims(M, (N + bn - 1) / bn, B)`,
# and the fast path requires `N % 8 == 0`, so `N == grid.y * 8` exactly.
QMV_FAST_BN = 8

# Every affine-4/group-64 projection the decode round can reach, keyed by N.
# `n >= 4096` is the wide-branch test at quantized.h:1917.
KNOWN_N = {
    16480: "target gdn.in_proj",
    14336: "target fa.qkv",
    34816: "target mlp.gate_up",
    248320: "lm_head (target verify rows or head draft readout)",
    5120: "N=5120 projection (target gdn.out_proj / fa.o_proj / mlp.down, or MTP head fc / o_proj / down)",
}


def parse_shape(text: str):
    m = SHAPE_RE.match(text)
    if not m:
        return None
    return m.group("kernel"), int(m.group("w")), int(m.group("h")), int(m.group("d"))


def load(path: str):
    """Group census records by worker pid.

    One `--local-iterate` leg runs several workers against the same log: the
    reference-row generator and the serial control both decode at width 1, and
    only the MTP candidate worker reaches width > 1. Mixing them would dilute
    every share, so the caller must select one pid.
    """
    by_pid = collections.defaultdict(lambda: {"rounds": [], "gputime": []})
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("pid")
            if rec.get("event") in ("round", "gap"):
                by_pid[pid]["rounds"].append(rec)
            elif rec.get("event") == "gputime":
                by_pid[pid]["gputime"].append(rec)
    return by_pid


def select_pid(by_pid, requested):
    """The MTP candidate worker is the one that ever decodes at width > 1."""
    if requested is not None:
        if requested not in by_pid:
            sys.exit(f"pid {requested} not present in census")
        return requested
    best, best_key = None, None
    for pid, group in by_pid.items():
        widths = [r.get("width", 0) for r in group["rounds"]
                  if r.get("event") == "round"]
        if not widths:
            continue
        key = (max(widths), len(widths))
        if best_key is None or key > best_key:
            best, best_key = pid, key
    if best is None:
        sys.exit("no pid in the census has round records")
    return best


def census_counts(rounds, skip_rounds: int):
    """Exact dispatch counts of the entry point, per round, by ntg.x and phase."""
    by_ntg = collections.Counter()
    by_phase_ntg = collections.Counter()
    by_ntg_h = collections.Counter()
    by_phase_ntg_h = collections.Counter()
    per_round_total = []
    verify_width_hist = collections.Counter()
    accepted_hist = collections.Counter()
    kept = 0
    for rec in rounds:
        if rec.get("event") != "round":
            continue
        if rec.get("round", -1) < skip_rounds:
            continue
        kept += 1
        verify_width_hist[rec.get("width", -1)] += 1
        if rec.get("accepted", -1) >= 0:
            accepted_hist[rec["accepted"]] += 1
        total = 0
        for phase, entry in (rec.get("phases") or {}).items():
            for shape, count in (entry.get("shapes") or {}).items():
                parsed = parse_shape(shape)
                if not parsed or parsed[0] != KERNEL:
                    continue
                _, w, h, _ = parsed
                by_ntg[w] += count
                by_phase_ntg[(phase, w)] += count
                by_ntg_h[(w, h)] += count
                by_phase_ntg_h[(phase, w, h)] += count
                total += count
        per_round_total.append(total)
    return {
        "rounds_counted": kept,
        "by_ntg": dict(sorted(by_ntg.items())),
        "by_phase_ntg": {f"{p}|{w}": c for (p, w), c in sorted(by_phase_ntg.items())},
        "by_phase_ntg_n": {
            f"{p}|M={w}|N={h * QMV_FAST_BN}": c
            for (p, w, h), c in sorted(by_phase_ntg_h.items())
        },
        "by_ntg_tiles": {f"{w}|{h}": c for (w, h), c in sorted(by_ntg_h.items())},
        "per_round_total_mean": statistics.mean(per_round_total) if per_round_total else 0.0,
        "per_round_total_median": statistics.median(per_round_total) if per_round_total else 0.0,
        "verify_width_histogram": dict(sorted(verify_width_hist.items())),
        "accepted_histogram": dict(sorted(accepted_hist.items())),
    }


def exclusive_shape_cost(gputime):
    """Mean GPU ns per dispatch for every shape measured on a one-dispatch buffer."""
    acc = collections.defaultdict(lambda: {"buffers": 0, "gpu_ns": 0})
    for rec in gputime:
        for key, bucket in (rec.get("exclusive_kernels") or {}).items():
            parts = key.split("|", 2)
            if len(parts) != 3:
                continue
            shape = parts[2]
            acc[shape]["buffers"] += bucket.get("buffers", 0)
            acc[shape]["gpu_ns"] += bucket.get("gpu_ns", 0)
    return {
        shape: v["gpu_ns"] / v["buffers"]
        for shape, v in acc.items()
        if v["buffers"] > 0
    }


def signature_attribution(gputime, per_shape_ns):
    """GPU ns held by the entry point, per phase, from single-phase buffers.

    A buffer whose signature contains only entry-point dispatches contributes
    its whole `gpu_ns`. A mixed buffer contributes the part its entry-point
    dispatches account for under `per_shape_ns`, capped at the buffer total, and
    the uncovered remainder is reported so the attribution is auditable.
    """
    phase_entry_ns = collections.Counter()
    phase_total_ns = collections.Counter()
    mixed_uncovered_ns = 0
    for rec in gputime:
        for key, bucket in (rec.get("signatures") or {}).items():
            parts = key.split("|", 2)
            if len(parts) != 3:
                continue
            _, phase, sig = parts
            gpu_ns = bucket.get("gpu_ns", 0)
            phase_total_ns[phase] += gpu_ns
            entry_model, other_model = 0.0, 0.0
            for item in sig.split(","):
                m = SIG_ITEM_RE.match(item)
                if not m:
                    continue
                shape, count = m.group("shape"), int(m.group("count"))
                cost = per_shape_ns.get(shape, 0.0) * count
                parsed = parse_shape(shape)
                if parsed and parsed[0] == KERNEL:
                    entry_model += cost
                else:
                    other_model += cost
            model_total = entry_model + other_model
            if entry_model <= 0:
                continue
            if model_total <= 0:
                continue
            share = min(1.0, entry_model / model_total)
            phase_entry_ns[phase] += gpu_ns * share
            if other_model > 0:
                mixed_uncovered_ns += gpu_ns * (1.0 - share)
    return phase_entry_ns, phase_total_ns, mixed_uncovered_ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-rounds", type=int, default=8,
                    help="drop the cold ramp; the census keys rounds from 0")
    ap.add_argument("--label", default="")
    ap.add_argument("--pid", type=int, default=None,
                    help="worker pid; default is the one that reaches width > 1")
    args = ap.parse_args()

    by_pid = load(args.census)
    if not by_pid:
        sys.exit(f"{args.census}: no round records")
    pid = select_pid(by_pid, args.pid)
    rounds = by_pid[pid]["rounds"]
    gputime = by_pid[pid]["gputime"]
    pid_summary = {
        str(p): {
            "rounds": sum(1 for r in g["rounds"] if r.get("event") == "round"),
            "max_width": max((r.get("width", 0) for r in g["rounds"]
                              if r.get("event") == "round"), default=0),
            "gputime_snapshots": len(g["gputime"]),
        }
        for p, g in sorted(by_pid.items(), key=lambda kv: str(kv[0]))
    }

    counts = census_counts(rounds, args.skip_rounds)
    per_shape_ns = exclusive_shape_cost(gputime)
    phase_entry_ns, phase_total_ns, uncovered = signature_attribution(
        gputime, per_shape_ns)

    busy_ns = sum(r.get("gpu_busy_ns", 0) for r in gputime)
    idle_ns = sum(r.get("gpu_idle_ns", 0) for r in gputime)
    entry_ns = sum(phase_entry_ns.values())

    ntg5 = counts["by_ntg"].get(5, 0)
    by_ntg_n = collections.Counter()
    for key, c in counts["by_ntg_tiles"].items():
        w, h = (int(v) for v in key.split("|"))
        by_ntg_n[(w, h * QMV_FAST_BN)] += c
    ntg5_wide = sum(c for (w, n), c in by_ntg_n.items() if w == 5 and n >= 4096)

    out = {
        "harness": "local",
        "rung": "0b",
        "label": args.label,
        "kernel": KERNEL,
        "worker_pid": pid,
        "pids_in_census": pid_summary,
        "qmv_arm": "sumtable (shipped default, MLX_E120_QMV_ARM unset)",
        "instrument": "E58 dispatch census + E80 GPU-time ledger, applied from "
                      "research/e80-artifacts/gputime-census.patch and reverted",
        "instrument_is_timing_safe": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "counts": counts,
        "gpu_time": {
            "busy_ns": busy_ns,
            "idle_ns": idle_ns,
            "entry_point_ns": entry_ns,
            "entry_point_share_of_busy": entry_ns / busy_ns if busy_ns else None,
            "by_phase_entry_ns": dict(phase_entry_ns),
            "by_phase_total_ns": dict(phase_total_ns),
            "by_phase_entry_share": {
                p: (phase_entry_ns[p] / phase_total_ns[p])
                for p in phase_total_ns if phase_total_ns[p]
            },
            "mixed_buffer_uncovered_ns": uncovered,
            "shapes_with_exclusive_cost": len(per_shape_ns),
        },
        "question_3_ntg_x_equals_5": {
            "dispatches_any_n": ntg5,
            "dispatches_n_ge_4096": ntg5_wide,
            "reachable": ntg5_wide > 0,
        },
        "by_ntg_and_n": {
            f"M={w}|N={n}": {
                "dispatches": c,
                "wide_branch": n >= 4096,
                "projection": KNOWN_N.get(n, "unmapped"),
            }
            for (w, n), c in sorted(by_ntg_n.items())
        },
        "entry_point_shape_cost_ns": {
            s: v for s, v in sorted(per_shape_ns.items()) if s.startswith(KERNEL)
        },
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
