#!/usr/bin/env python3
"""E106 rung 0 -- price the redundant `x` reads of the qmv fast path.

    usage: research/e106_x_reread.py [--json research/out/e106/trace-d4.json]

`quantized.cpp:198` dispatches `grid_dims(M, ceil(N/8), B)` threadgroups and
`quantized.h:1171` idles every threadgroup column except `tid.x == 0`, so one
dispatch runs `N/8` live threadgroups. `qmv_fast_crossrow_affine4_g64_wide`
reads `x` straight from `device` memory inside its k-loop and never stages it
in threadgroup memory, so each live threadgroup re-reads the whole `M x K`
activation. This prints that redundant volume next to the measured excess so
the implied service bandwidth of the redundant path can be compared across
tensors.

A census leg is never a timing leg. Only Metal's GPU clock is valid here.
"""

from __future__ import annotations

import argparse
import json

BN = 8
ELEM_BYTES = 2

GEOMETRY = {
    "lm_head": (5120, 248_320),
    "mlp.gate_up": (5120, 34_816),
    "gdn.in_proj": (5120, 16_480),
    "fa.qkv": (5120, 14_336),
    "gdn.out_proj": (6144, 5120),
    "fa.o_proj": (6144, 5120),
    "mlp.down": (17_408, 5120),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research/out/e106/trace-d4.json")
    args = ap.parse_args()

    with open(args.json) as fh:
        payload = json.load(fh)
    leg = next(iter(payload.values()))
    width = leg["width"]
    tensors = leg["tensors"]

    print(f"M={width}   fit F={leg['fit']['F_us']:.2f}us  "
          f"S={leg['fit']['S_us_per_gb']:.1f}us/GB")
    print(f"{'tensor':14s} {'K':>6s} {'x KB':>7s} {'live tg':>8s} "
          f"{'x re-read MB':>13s} {'excess us':>10s} {'implied GB/s':>13s}")
    for name in sorted(tensors, key=lambda n: tensors[n]["excess_us"]):
        k, n = GEOMETRY[name]
        x_bytes = width * k * ELEM_BYTES
        live_tg = n // BN
        reread = live_tg * x_bytes
        excess = tensors[name]["excess_us"]
        implied = reread / 1e3 / excess if excess > 0.05 else float("inf")
        print(f"{name:14s} {k:6d} {x_bytes / 1024:7.1f} {live_tg:8d} "
              f"{reread / 1e6:13.1f} {excess:10.2f} {implied:13.0f}")


if __name__ == "__main__":
    main()
