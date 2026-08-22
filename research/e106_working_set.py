#!/usr/bin/env python3
"""E106 rung 0 -- the two N=5120 components against one working-set model.

    usage: research/e106_working_set.py [--json OUT]

Rung 0 found two separate excesses on the N=5120 projections:

  * a size component, paid by a dispatch as a function of its own `x` and
    independent of the block that ran earlier in the layer; and
  * effect B, an extra cost paid only by the dispatch immediately after the
    Gated DeltaNet block, gone one dispatch later.

`research/e106_block_context.py` showed the two are additive, and its placebo
control showed effect B does not respond to elapsed distance. That leaves one
candidate mechanism for both: `qmv_fast_crossrow_affine4_g64_wide` reads the
whole of `x` from `device` memory in every one of its N/8 threadgroups, with no
threadgroup staging, so `x` must stay resident in a cache the 640 threadgroups
share. A dispatch pays when its own `x` is too big for that cache, and it pays
again when the predecessor evicted `x` before it ran.

This reducer puts the shipped shapes on that model's two axes so the confounds
are visible. It reads the measured excesses from the block-context payloads and
derives the byte columns from the source shapes, so nothing here is a new
measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib

# Source of every byte figure below.
#   Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift:171
#     outputShapes [[B, T, Hv, Dv], state.shape], outputDTypes [bf16, fp32]
#   Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:638
#     state = zeros([1, 48, 128, 128], fp32)   -- constant in T
GDN_STATE_BYTES = 1 * 48 * 128 * 128 * 4
GDN_Y_PER_ROW = 48 * 128 * 2          # bf16 [B, T, Hv, Dv]
SDPA_OUT_PER_ROW = 24 * 256 * 2       # bf16, 24 q heads x head dim 256
GATE_UP_PER_ROW = 34_816 * 2          # bf16

# victim -> (K, predecessor block, predecessor bytes written per row,
#            predecessor bytes written that do not depend on width)
VICTIMS = {
    "gdn.out_proj": (6144, "gated_delta_step", GDN_Y_PER_ROW,
                     GDN_STATE_BYTES),
    "fa.o_proj": (6144, "sdpa_vector", SDPA_OUT_PER_ROW, 0),
    "mlp.down": (17_408, "mlp.gate_up", GATE_UP_PER_ROW, 0),
}
PAYLOADS = {1: "block_context_m1.json", 3: "block_context_m3.json",
            5: "block_context_m5.json"}


def load(width):
    path = pathlib.Path("research/out/e106") / PAYLOADS[width]
    if not path.exists():
        return None
    data = json.load(path.open())
    groups = next(iter(data.values()))["groups"]
    out = {}
    for key, row in groups.items():
        tensor, _host = key.split("@")
        out.setdefault(tensor, []).append((row["excess_us"], row["count"]))
    # Pool the gdn-hosted and fa-hosted cells of a tensor by dispatch count.
    return {t: sum(e * n for e, n in v) / sum(n for _e, n in v)
            for t, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    print("working-set model: victim x bytes against predecessor write bytes\n")
    print(f"  {'M':>2} {'victim':<14} {'K':>6} {'x bytes':>9} "
          f"{'pred block':<18} {'pred write B':>13} {'excess us':>10}")
    payload = {}
    for width in sorted(PAYLOADS):
        excess = load(width)
        if excess is None:
            continue
        for tensor, (k, block, per_row, fixed) in VICTIMS.items():
            if tensor not in excess:
                continue
            x_bytes = width * k * 2
            pred = fixed + width * per_row
            print(f"  {width:2d} {tensor:<14} {k:6d} {x_bytes:9d} "
                  f"{block:<18} {pred:13d} {excess[tensor]:10.2f}")
            payload[f"m{width}/{tensor}"] = {
                "width": width, "k": k, "x_bytes": x_bytes,
                "predecessor": block, "predecessor_write_bytes": pred,
                "excess_us": excess[tensor],
            }
        print()

    print("  the design matrix at M=5, empty cells are what the shipped")
    print("  shapes cannot supply:\n")
    print(f"  {'pred write':>12} | {'x = 61440 B':>14} {'x = 174080 B':>14}")
    m5 = {t: v for t, v in (load(5) or {}).items()}
    cells = {
        "61 KB": ("fa.o_proj", None),
        "348 KB": (None, "mlp.down"),
        "3207 KB": ("gdn.out_proj", None),
    }
    for label, (narrow, wide) in cells.items():
        def cell(name):
            return f"{m5[name]:+.2f} us" if name and name in m5 else "--"
        print(f"  {label:>12} | {cell(narrow):>14} {cell(wide):>14}")

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
