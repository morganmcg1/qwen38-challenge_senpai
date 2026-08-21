#!/usr/bin/env python3
"""E106 rung 0 -- mine every phase for N=5120 cells the target path cannot give.

    usage: research/e106_phase_sweep.py TAG [TAG ...] [--json OUT]

The working-set model says an N=5120 projection pays when its own `x` is large
and pays again when its predecessor evicted `x`. The target decode path only
supplies the diagonal of that 2x2: a 3.15 MB predecessor write in front of a
small `x` (`gdn.out_proj`), and a small write in front of a small `x`
(`fa.o_proj`).

But the target decode path is not the only phase in the trace. `draft_head`,
`seed_prefill` and `outside` run the same kernels at different row counts, so
they place the same shapes against different `x` sizes and different
predecessors. Any cell they supply is free evidence that needs no harness.

This reducer enumerates every `affine_qmv_fast` dispatch in every phase, tags it
with grid.x (the row count), grid.y (the output width), and the kernel that ran
immediately before it, then reports the cells found.

A census leg is never a timing leg. Only Metal's GPU clock is valid here.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys

SHAPE_RE = re.compile(
    r"^(?P<phase>[^|]+)\|(?P<kernel>\S+) grid=(?P<gx>\d+)x(?P<gy>\d+)x(?P<gz>\d+)"
    r" tg=(?P<tx>\d+)x(?P<ty>\d+)x(?P<tz>\d+)$")
NARROW_GY = 640


def short(kernel: str) -> str:
    """Collapse a mangled kernel name to something readable."""
    for probe in ("gated_delta_replay_state", "gated_delta_step",
                  "packed_gdn_prework", "sdpa_vector", "affine_qmv_fast",
                  "affine_qmm", "rms_norm", "layer_norm"):
        if probe in kernel:
            return probe
    return kernel.split("_")[0][:24]


def sweep(path):
    """(phase, gx, gy, predecessor) -> [us]."""
    cells = collections.defaultdict(list)
    seen_phase = collections.Counter()
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        rounds = collections.defaultdict(list)
        for rnd, ordinal, _w, shape_id, gpu_ns in rec["trace"]:
            rounds[rnd].append((ordinal, shape_id, gpu_ns))
        for _rnd, rows in rounds.items():
            prev = None
            for _ordinal, shape_id, gpu_ns in sorted(rows):
                match = parsed[shape_id]
                if match is None:
                    prev = None
                    continue
                phase = match.group("phase")
                kernel = match.group("kernel")
                seen_phase[phase] += 1
                if kernel.startswith("affine_qmv_fast"):
                    key = (phase, int(match.group("gx")),
                           int(match.group("gy")), prev or "?")
                    cells[key].append(gpu_ns / 1e3)
                prev = short(kernel)
    return cells, seen_phase


# Immediate predecessor of each N=5120 projection. Every one is a distinct
# elementwise op, so the string identifies the tensor without a marker scan.
PRED_TO_TENSOR = {
    "Cf4IAsTypeADf4ISigmoidCE": "gdn.out_proj",
    "CV2ISigmoidBDV2IBroadcas": "fa.o_proj",
    "CV2ISigmoidADV2IMultiply": "mlp.down",
}
K_OUT_PROJ = 6144


def effect_b_ladder(cells):
    """gdn.out_proj minus fa.o_proj at each (phase, rows) cell."""
    by_cell = collections.defaultdict(dict)
    for (phase, gx, gy, prev), vals in cells.items():
        if gy != NARROW_GY:
            continue
        tensor = PRED_TO_TENSOR.get(prev)
        if tensor in ("gdn.out_proj", "fa.o_proj"):
            by_cell[(phase, gx)][tensor] = vals
    out = []
    for (phase, gx), got in sorted(by_cell.items()):
        if len(got) != 2:
            continue
        gdn, fa = got["gdn.out_proj"], got["fa.o_proj"]
        if len(gdn) < 2 or len(fa) < 2:
            continue
        sem = (statistics.pstdev(gdn) ** 2 / len(gdn)
               + statistics.pstdev(fa) ** 2 / len(fa)) ** 0.5
        diff = statistics.fmean(gdn) - statistics.fmean(fa)
        out.append({
            "phase": phase, "rows": gx, "x_bytes": gx * K_OUT_PROJ * 2,
            "gdn_us": statistics.fmean(gdn), "fa_us": statistics.fmean(fa),
            "n_gdn": len(gdn), "n_fa": len(fa),
            "effect_b_us": diff, "sem_us": sem,
            "sigma": diff / sem if sem > 0 else float("nan"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    payload = {}

    for tag in args.tags:
        path = pathlib.Path("research/out") / tag / "census.jsonl"
        if not path.exists():
            sys.exit(f"e106_phase_sweep: no census at {path}")
        cells, seen_phase = sweep(path)
        print(f"=== {tag}")
        print("  dispatches seen per phase: " + ", ".join(
            f"{p}={n}" for p, n in seen_phase.most_common()))

        print(f"\n  every N=5120 cell, any phase, any row count")
        print(f"  {'phase':<15} {'rows':>5} {'predecessor':<26} {'n':>6} "
              f"{'us':>9} {'sd':>7}")
        rows_out = []
        for (phase, gx, gy, prev), vals in sorted(cells.items()):
            if gy != NARROW_GY:
                continue
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"  {phase:<15} {gx:5d} {prev:<26} {len(vals):6d} "
                  f"{statistics.fmean(vals):9.2f} {sd:7.2f}")
            rows_out.append({
                "phase": phase, "rows": gx, "predecessor": prev,
                "n": len(vals), "mean_us": statistics.fmean(vals),
                "sd_us": sd,
            })

        # The predecessor string identifies the tensor, because each N=5120
        # projection has a distinct elementwise op immediately before it. The
        # dispatch counts confirm it: 48 gdn, 16 fa and 64 mlp per round.
        ladder = effect_b_ladder(cells)
        print("\n  effect B = gdn.out_proj - fa.o_proj, byte-identical, by rows")
        print(f"  {'phase':<15} {'rows':>5} {'x bytes':>9} {'gdn us':>9} "
              f"{'fa us':>9} {'effect B':>9} {'sem':>6} {'sigma':>7}")
        for row in ladder:
            print(f"  {row['phase']:<15} {row['rows']:5d} {row['x_bytes']:9d} "
                  f"{row['gdn_us']:9.2f} {row['fa_us']:9.2f} "
                  f"{row['effect_b_us']:+9.2f} {row['sem_us']:6.2f} "
                  f"{row['sigma']:7.1f}")

        payload[tag] = {"n5120_cells": rows_out, "effect_b_ladder": ladder,
                        "phases": dict(seen_phase)}

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
