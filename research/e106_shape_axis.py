#!/usr/bin/env python3
"""E106 -- the N axis and the x-refetch readout, from the census trace alone.

    usage: research/e106_shape_axis.py TAG [TAG ...] [--json OUT]

The advisor asked for a 1x3 ladder in N at fixed K, to separate a shape term
from the neighbour term that effect B measures. The legal-shape warmup already
runs every scored projection at rows 1 through 9 inside one session, so the
ladder is observational rather than synthetic. This reducer prices each cell in
GB/s, because the N axis changes total bytes by design.

Byte model, affine 4-bit group 64: 0.5 bytes of packed weight per element,
plus a bfloat16 scale and a bfloat16 bias per 64 elements, so
`bytes = N * K * 0.5625`. K comes from the checkpoint config, not from a fit.

Each threadgroup of `qmv_fast` reads the whole of `x` from device memory, and
there are `N/8` threadgroups per m-slice. The cost above the clean-family law
is therefore reported as an implied number of `x` refetches, which is the
quantity the residency hypothesis predicts.

A census leg is never a timing leg. Only Metal's GPU clock is valid here.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

from e106_phase_sweep import PRED_TO_TENSOR, SHAPE_RE, short

BYTES_PER_ELEMENT = 0.5625

# grid.y -> (tensor, K). Every K is a checkpoint dimension: hidden 5120,
# intermediate 17408, 24 query heads x 256 = 6144 into `fa.o_proj`, and
# 48 Gated DeltaNet value heads x 128 = 6144 into `gdn.out_proj`.
BY_GRID_Y = {
    31040: ("lm_head", 5120),
    4352: ("mlp.gate_up", 5120),
    2060: ("gdn.in_proj", 5120),
    1792: ("fa.qkv", 5120),
}
# grid.y 640 holds three different tensors, separated by their predecessor.
N5120_K = {"gdn.out_proj": 6144, "fa.o_proj": 6144, "mlp.down": 17408}

# Clean-family refit on this base, both rung-0 legs agreeing to 1.1 us/GB.
LAW_F_US = 25.94
LAW_S_US_PER_GB = 5560.0
CLEAN = ("fa.qkv", "gdn.in_proj", "mlp.gate_up", "lm_head")


def fit(points):
    """Least squares `us = F + GB * S` over the clean streaming families."""
    n = len(points)
    if n < 2:
        return None
    sx = sum(gb for gb, _ in points)
    sy = sum(us for _, us in points)
    sxx = sum(gb * gb for gb, _ in points)
    sxy = sum(gb * us for gb, us in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    s = (n * sxy - sx * sy) / denom
    f = (sy - s * sx) / n
    mean = sy / n
    ss_tot = sum((us - mean) ** 2 for _, us in points)
    ss_res = sum((us - (f + gb * s)) ** 2 for gb, us in points)
    return {"F_us": f, "S_us_per_gb": s, "S_gb_per_s": 1e6 / s if s else 0.0,
            "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
            "n": n}


def cells(path):
    """(phase, rows, tensor) -> [us]."""
    out = collections.defaultdict(list)
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        rounds = collections.defaultdict(list)
        for rnd, ordinal, _w, shape_id, gpu_ns in rec["trace"]:
            rounds[rnd].append((ordinal, shape_id, gpu_ns))
        for rows in rounds.values():
            prev = None
            for _ordinal, shape_id, gpu_ns in sorted(rows):
                match = parsed[shape_id]
                if match is None:
                    prev = None
                    continue
                kernel = match.group("kernel")
                if kernel.startswith("affine_qmv_fast"):
                    grid_y = int(match.group("gy"))
                    named = BY_GRID_Y.get(grid_y)
                    tensor = named[0] if named else PRED_TO_TENSOR.get(prev)
                    if tensor is not None:
                        out[(match.group("phase"),
                             int(match.group("gx")), tensor)].append(
                                 gpu_ns / 1e3)
                prev = short(kernel)
    return out


def geometry(tensor):
    for grid_y, (name, k) in BY_GRID_Y.items():
        if name == tensor:
            return grid_y * 8, k, grid_y
    return 5120, N5120_K[tensor], 640


def table(found, phase):
    """One row per cell, with the law refitted independently at each width."""
    raw = collections.defaultdict(list)
    for (ph, m, tensor), vals in found.items():
        if ph == phase and len(vals) >= 2:
            n, k, tg = geometry(tensor)
            raw[m].append((tensor, n, k, tg, vals))

    rows, fits = [], {}
    for m, entries in raw.items():
        clean = [(n * k * BYTES_PER_ELEMENT / 1e9, statistics.fmean(vals))
                 for tensor, n, k, _tg, vals in entries if tensor in CLEAN]
        law = fit(clean)
        fits[m] = law
        for tensor, n, k, tg, vals in entries:
            weight_gb = n * k * BYTES_PER_ELEMENT / 1e9
            x_gb = m * k * 2 / 1e9
            us = statistics.fmean(vals)
            law_us = (law["F_us"] + weight_gb * law["S_us_per_gb"]
                      if law else float("nan"))
            extra_gb = ((us - law_us) / law["S_us_per_gb"]
                        if law else float("nan"))
            rows.append({
                "phase": phase, "rows": m, "tensor": tensor, "N": n, "K": k,
                "threadgroups": tg * m, "n": len(vals),
                "us": us, "sd_us": statistics.pstdev(vals),
                "weight_mb": weight_gb * 1e3,
                "gb_per_s": weight_gb / (us * 1e-6),
                "law_us": law_us, "excess_us": us - law_us,
                "excess_pct": 100.0 * (us - law_us) / law_us,
                "x_kb": x_gb * 1e6,
                "implied_x_refetches": extra_gb / x_gb if x_gb else
                float("nan"),
                # Every threadgroup in an m-slice reads the whole of x, and
                # there are N/8 of them, so N/8 refetches means no reuse at
                # all. The ratio is the share of threadgroups that missed.
                "implied_x_miss_pct": 100.0 * (extra_gb / x_gb) / (n / 8)
                if x_gb else float("nan"),
            })
    rows.sort(key=lambda r: (r["N"], r["K"], r["rows"]))
    return rows, fits


def show(rows, title):
    print(f"\n  {title}")
    print(f"  {'tensor':<13} {'N':>7} {'K':>6} {'rows':>4} {'TGs':>7} "
          f"{'wt MB':>7} {'us':>9} {'GB/s':>7} {'law us':>8} {'exc %':>7} "
          f"{'x KB':>7} {'x miss %':>9}")
    for r in rows:
        print(f"  {r['tensor']:<13} {r['N']:7d} {r['K']:6d} {r['rows']:4d} "
              f"{r['threadgroups']:7d} {r['weight_mb']:7.2f} {r['us']:9.2f} "
              f"{r['gb_per_s']:7.1f} {r['law_us']:8.2f} {r['excess_pct']:+7.1f} "
              f"{r['x_kb']:7.1f} {r['implied_x_miss_pct']:9.2f}")


def x_traffic_model(rows, fits, reference_rows=2):
    """Re-express the width penalty as an implied `x` cache-miss fraction.

    `qmv_fast` gives each threadgroup the whole of `x`, so an m-slice reads
    `N/8` copies of it, which is `N*K*M/4` bytes against `N*K*0.5625` bytes of
    weight. Those two are proportional at fixed M, so this is a change of
    units, not an independent test of the re-read mechanism: for the families
    the law was fitted on, a common miss fraction is guaranteed by the fit.

    What the column does carry is the held-out families. `fa.o_proj`,
    `gdn.out_proj` and `mlp.down` are never fitted, so whether they land on
    the common curve is a real result.
    """
    base = fits.get(reference_rows)
    if not base:
        return []
    out = []
    for r in rows:
        x_traffic_gb = (r["N"] / 8) * r["rows"] * r["K"] * 2 / 1e9
        weight_gb = r["weight_mb"] / 1e3
        surplus_us = r["us"] - base["F_us"] - weight_gb * base["S_us_per_gb"]
        out.append({
            **r,
            "x_traffic_gb": x_traffic_gb,
            "surplus_us": surplus_us,
            "implied_miss_pct": 100.0 * surplus_us
            / (base["S_us_per_gb"] * x_traffic_gb),
        })
    return out


def show_x_model(rows, reference_rows=2):
    families = sorted({r["tensor"] for r in rows})
    widths = sorted({r["rows"] for r in rows})
    print(f"\n  implied `x` cache-miss %, one rate for all cells "
          f"(the rows={reference_rows} fit)")
    print(f"  {'rows':>4} " + " ".join(f"{f:>13}" for f in families))
    for w in widths:
        cells_ = {r["tensor"]: r for r in rows if r["rows"] == w}
        line = f"  {w:4d} "
        for f in families:
            r = cells_.get(f)
            line += (f"{r['implied_miss_pct']:13.1f}" if r else f"{'--':>13}")
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--phase", action="append",
                    default=None, help="default: outside, target_forward")
    ap.add_argument("--json")
    args = ap.parse_args()
    phases = args.phase or ["target_forward", "outside", "target_verify"]
    payload = {}

    for tag in args.tags:
        path = pathlib.Path("research/out") / tag / "census.jsonl"
        if not path.exists():
            sys.exit(f"e106_shape_axis: no census at {path}")
        found = cells(path)
        print(f"=== {tag}")
        payload[tag] = {}
        for phase in phases:
            rows, fits = table(found, phase)
            if not rows:
                continue
            show(rows, f"phase {phase}")
            print(f"\n  law refitted on the four clean families at each width")
            print(f"  {'rows':>4} {'families':>9} {'F us':>9} {'S us/GB':>9} "
                  f"{'S GB/s':>8} {'R2':>10}")
            for m, law in sorted(fits.items()):
                if law:
                    print(f"  {m:4d} {law['n']:9d} {law['F_us']:9.2f} "
                          f"{law['S_us_per_gb']:9.1f} {law['S_gb_per_s']:8.1f} "
                          f"{law['r2']:10.6f}")
            modelled = x_traffic_model(rows, fits)
            if modelled:
                show_x_model(modelled)
            payload[tag][phase] = {"cells": modelled or rows,
                                   "fits": {str(m): law
                                            for m, law in fits.items()}}

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
