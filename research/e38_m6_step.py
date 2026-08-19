#!/usr/bin/env python3
"""E38 by-product: decompose the anomalous M=5 -> M=6 step in the shipped ladder.

The shipped selection table jumps from `_m<T,5,5,true>` (one weight stream, one
working x-block) to `_m<T,6,3,true>` (two weight streams, two working x-blocks).
Arm (a) and arm (b) price the weight stream directly at M=6, so the step can be
audited instead of assumed.
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CURVE = ROOT / ".mlxfast-private" / "qmv-curve"

SCORED = [
    "linear_attn.in_proj_fused_qkvzba",
    "linear_attn.out_proj",
    "full_attn.qkv_proj_fused",
    "full_attn.o_proj",
    "mlp.gate_up_fused",
    "mlp.down",
    "head.lm_head",
    "head.compact_draft_vocab",
]
CALLS: dict[str, int] = {}


def per_shape(tag: str) -> dict[str, dict[int, float]]:
    j = json.loads((CURVE / tag / "summary.json").read_text())
    out: dict[str, dict[int, float]] = {}
    for row in j["per_shape_curve"]:
        name = row["name"]
        if name not in SCORED:
            continue
        CALLS.setdefault(name, int(row["calls_per_verify"]))
        out.setdefault(name, {})[int(row["m"])] = row["seconds_per_call"] * 1e6
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="e38-base-r1")
    ap.add_argument("--arm-a", default="e38-arma-r1")
    ap.add_argument("--arm-b", default="e38-armb-r1")
    args = ap.parse_args()

    base, a, b = per_shape(args.base), per_shape(args.arm_a), per_shape(args.arm_b)

    print("M=5 -> M=6 step, per shape, against the measured weight-pass value")
    print("  step   = base(6)/base(5) - 1        (the shipped ladder's own jump)")
    print("  wpass  = a(6)/base(6) - b(6)/base(6) (R1: one weight stream at M=6)")
    print()
    hdr = f"{'shape':<34}{'b5 us':>10}{'b6 us':>10}{'step':>9}{'wpass':>9}{'resid':>9}"
    print(hdr)
    tot5 = tot6 = tot_w = 0.0
    for s in SCORED:
        n = CALLS[s]
        u5, u6 = base[s][5], base[s][6]
        w = (a[s][6] - b[s][6]) / u6
        step = u6 / u5 - 1.0
        # residual step after removing the priced weight stream
        resid = (u6 - a[s][6] + b[s][6]) / u5 - 1.0
        print(f"{s:<34}{u5:>10.2f}{u6:>10.2f}{step:>9.4f}{w:>9.4f}{resid:>9.4f}")
        tot5 += n * u5
        tot6 += n * u6
        tot_w += n * (a[s][6] - b[s][6])
    print()
    print(f"C_round(5) = {tot5/1000:.3f} ms   C_round(6) = {tot6/1000:.3f} ms")
    print(f"  step             = {tot6/tot5 - 1:+.4f}   ({(tot6-tot5)/1000:+.3f} ms)")
    print(f"  priced weight    = {tot_w/tot6:+.4f} of C(6)  ({tot_w/1000:+.3f} ms)")
    print(f"  residual step    = {(tot6-tot_w)/tot5 - 1:+.4f}   "
          f"({(tot6-tot_w-tot5)/1000:+.3f} ms)")
    print()
    print("neighbouring per-row increments of C_round, same session (ms):")
    widths = sorted({m for s in SCORED for m in base[s]})
    prev = None
    for m in widths:
        tot = sum(CALLS[s] * base[s][m] for s in SCORED) / 1000.0
        inc = "" if prev is None else f"{tot - prev:+8.2f}"
        print(f"  M={m}  C_round={tot:8.3f}{inc}")
        prev = tot
    print()
    print("Reading: the +34 % ladder step at M=6 is NOT an anomalous cell.  E38")
    print("prices the second weight stream at +15.4 ms; the remaining +17.4 ms is")
    print("one more input row, in line with the +13.6 ms (M=4->5) and +9.6 ms")
    print("(M=6->7) rows either side.  So the shipped M=6 cell carries no large")
    print("unexplained tax once the weight pass is priced -- which is a negative")
    print("answer to 'is our M=6 cell the width-confined defect?'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
