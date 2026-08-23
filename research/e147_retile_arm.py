"""E147 rung E: verify the in-kernel origin re-derivation, and E-4's default.

The retile keeps the frozen host grid and re-derives each threadgroup's output
tile origin from the frozen block index. If that map is not a bijection onto
the retiled tile grid, the kernel silently computes wrong outputs. This
simulates the exact integer arithmetic the kernel runs, for every scored shape,
and proves three things:

  1. Under the gate the map covers every retiled tile exactly once, and the
     tiles partition the whole MxN output with no gap and no overlap.
  2. Outside the gate the kernel falls back to the shipped tiling, which is
     correct by construction. The M=511 proposal-head priming pass is the
     shape that needs this.
  3. E-4: the submitted default is off, read from the kernel source.

A wrong map must be detectable, so each check runs against deliberately broken
variants as a positive control (Rule 101).

  usage: research/e147_retile_arm.py [--json out.json]
"""
import argparse
import json
import re

QUANTIZED_H = (
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
)
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

HOST_BM = 32
HOST_BN = 32
ARM_BM = 64
ARM_BN = 16

# (label, M, N) reaching affine_qmm_t locally. N values are the scored
# projections; M=512 is the seed prefill, M=511 the proposal-head priming pass.
SHAPES = [
    ("gdn.in_proj", 512, 16480),
    ("gdn.out_proj", 512, 5120),
    ("fa.qkv", 512, 14336),
    ("fa.o_proj", 512, 5120),
    ("mlp.gate_up", 512, 34816),
    ("mlp.down", 512, 5120),
    ("lm_head", 512, 248320),
    ("head priming", 511, 5120),
    ("head priming qkv", 511, 14336),
]


def ceil_div(a, b):
    return (a + b - 1) // b


def gate(M, N, bm, bn):
    """The runtime gate in affine_qmm_t."""
    return M % bm == 0 and M % HOST_BM == 0 and N % bn == 0 and N % HOST_BN == 0


def origins(M, N, bm, bn, variant="correct"):
    """Replay the kernel's per-threadgroup origin computation over the frozen grid."""
    tiles_x_host = ceil_div(N, HOST_BN)
    tiles_y_host = ceil_div(M, HOST_BM)
    out = []
    if not gate(M, N, bm, bn):
        for ty in range(tiles_y_host):
            for tx in range(tiles_x_host):
                out.append((ty * HOST_BM, tx * HOST_BN, HOST_BM, HOST_BN))
        return out
    tiles_x = N // bn
    for ty in range(tiles_y_host):
        for tx in range(tiles_x_host):
            linear = ty * tiles_x_host + tx
            if variant == "correct":
                y_row = (linear // tiles_x) * bm
                y_col = (linear % tiles_x) * bn
            elif variant == "host_tiles_x":
                # the bug of reusing the frozen tile count
                y_row = (linear // tiles_x_host) * bm
                y_col = (linear % tiles_x_host) * bn
            elif variant == "swapped":
                y_row = (linear % tiles_x) * bm
                y_col = (linear // tiles_x) * bn
            elif variant == "no_rederive":
                # forgetting to re-derive at all
                y_row = ty * bm
                y_col = tx * bn
            out.append((y_row, y_col, bm, bn))
    return out


def coverage_ok(M, N, tiles):
    """Every output element covered exactly once by the emitted tiles."""
    if len(tiles) != len(set((r, c) for r, c, _, _ in tiles)):
        return False, "duplicate tile origin"
    covered = 0
    for r, c, bm, bn in tiles:
        if r < 0 or c < 0 or r >= M or c >= N:
            return False, f"origin ({r},{c}) outside {M}x{N}"
        covered += min(bm, M - r) * min(bn, N - c)
    if covered != M * N:
        return False, f"covered {covered} of {M * N}"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    src = open(QUANTIZED_H).read()
    twin = open(TWIN).read()
    out = {"host_tiling": [HOST_BM, HOST_BN], "arm_tiling": [ARM_BM, ARM_BN]}

    # E-4: the submitted default must be off, in the header and in the twin.
    m = re.search(r"constexpr bool kE147RetileOn = (true|false);", src)
    mt = re.search(r"constexpr bool kE147RetileOn = (true|false);", twin)
    out["retile_switch_found_in_header"] = bool(m)
    out["retile_switch_found_in_twin"] = bool(mt)
    out["retile_on_header"] = (m.group(1) == "true") if m else None
    out["retile_on_twin"] = (mt.group(1) == "true") if mt else None
    out["header_and_twin_agree"] = out["retile_on_header"] == out["retile_on_twin"]
    out["e4_submitted_default_is_off"] = out["retile_on_header"] is False
    print(f"retile switch in header : {out['retile_on_header']}")
    print(f"retile switch in twin   : {out['retile_on_twin']}")
    print(f"header and twin agree   : {out['header_and_twin_agree']}")
    print(f"E-4 default is off      : {out['e4_submitted_default_is_off']}")

    print(f"\n{'shape':18s} {'M':>4s} {'N':>7s}  {'gated':>5s}  {'blocks':>7s}  coverage")
    rows = []
    all_ok = True
    for label, M, N in SHAPES:
        g = gate(M, N, ARM_BM, ARM_BN)
        tiles = origins(M, N, ARM_BM, ARM_BN)
        ok, why = coverage_ok(M, N, tiles)
        all_ok &= ok
        print(f"{label:18s} {M:4d} {N:7d}  {str(g):>5s}  {len(tiles):7d}  {why}")
        rows.append(
            {"shape": label, "M": M, "N": N, "retiled": g, "blocks": len(tiles), "ok": ok, "why": why}
        )
    out["shapes"] = rows
    out["all_shapes_cover_exactly"] = bool(all_ok)

    # Rule 101 positive controls: each broken variant must be caught somewhere.
    print("\npositive controls (each must be CAUGHT on at least one scored shape)")
    controls = {}
    for variant in ("host_tiles_x", "swapped", "no_rederive"):
        caught = []
        for label, M, N in SHAPES:
            if not gate(M, N, ARM_BM, ARM_BN):
                continue
            ok, why = coverage_ok(M, N, origins(M, N, ARM_BM, ARM_BN, variant))
            if not ok:
                caught.append(label)
        controls[variant] = caught
        verdict = "CAUGHT" if caught else "NOT CAUGHT"
        print(f"  {variant:14s} {verdict:10s} on {len(caught)}/{sum(1 for l, M, N in SHAPES if gate(M, N, ARM_BM, ARM_BN))} gated shapes")
    out["positive_controls"] = controls
    out["all_controls_caught"] = all(bool(v) for v in controls.values())

    # The block count must be preserved, which is what makes the map a bijection.
    print("\nblock-count preservation on gated shapes")
    preserved = True
    for label, M, N in SHAPES:
        if not gate(M, N, ARM_BM, ARM_BN):
            continue
        host = ceil_div(N, HOST_BN) * ceil_div(M, HOST_BM)
        arm = ceil_div(N, ARM_BN) * ceil_div(M, ARM_BM)
        preserved &= host == arm
        print(f"  {label:18s} host {host:7d}  arm {arm:7d}  equal {host == arm}")
    out["block_count_preserved"] = bool(preserved)

    out["e1a_verified"] = bool(
        out["all_shapes_cover_exactly"]
        and out["all_controls_caught"]
        and out["block_count_preserved"]
        and out["header_and_twin_agree"]
    )
    print(f"\nE-1a index arithmetic verified : {out['e1a_verified']}")

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
