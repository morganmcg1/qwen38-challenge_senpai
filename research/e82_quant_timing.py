#!/usr/bin/env python3
"""Estimate the wall-clock cost of a rung-6 head build before launching one.

A tier-0 or tier-1 build touches every quantizable element of the master head
several hundred times. Launching one blind risks burning an allocation on a job
that cannot finish, so measure the per-element rate on a realistic slab and
multiply by the real `TRUNK` element count.

  PYTHONPATH=research python3 research/e82_quant_timing.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from e82_build_head import MASTER, TRUNK
from e82_quantizers import quantize
from e82_st import SafeTensors

PROBE_GROUPS = 1 << 16


def main() -> None:
    src = SafeTensors(MASTER)
    total = sum(int(np.prod(src.entries[n].shape)) for n in TRUNK)
    print(f"master TRUNK: {len(TRUNK)} tensors, {total:,} elements")

    probe = mx.random.normal((PROBE_GROUPS, 64)).astype(mx.bfloat16).astype(mx.float32)
    mx.eval(probe)
    probe_n = probe.size

    for method in ("mlx", "ls", "hqq"):
        t0 = time.time()
        r = quantize(probe, methods=(method,))
        mx.eval(r["weight"], r["scales"], r["biases"])
        dt = time.time() - t0
        rate = probe_n / dt
        print(
            f"{method:5s} {dt:7.2f} s / {probe_n:,} elem"
            f"  -> {rate/1e6:8.2f} Me/s"
            f"  -> full head {total/rate/60:8.2f} min"
        )


if __name__ == "__main__":
    main()
