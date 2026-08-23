#!/usr/bin/env python3
"""E147 rung E-0. Price the NAX seed-prefill retile from source, zero GPU.

WHAT THIS SETTLES. Feedback 4 proposes retiling the NAX transposed quantized
GEMM from 64x64 to 128x32 because the weight bytes fall from 8*N*K to 4*N*K.
That is correct as far as it goes. The source shows it is not the whole cost.

`qmm_t_nax_tgp_impl` stages ONLY the weight tile in threadgroup memory. There
is no `Xs`. `Atile.load(xk + kk1, K)` reads the activation straight from device
memory inside the k-loop, so every threadgroup streams its own BM x K slice of
X. The activation is therefore re-read once per N-tile, and the number of
N-tiles is N/BN. Halving BN doubles that traffic.

The contrast is the non-NAX `qmm_t_impl` in `quantized.h`, which does stage
`Xs` and is what rung A measured on this host.

THE MODEL. For one GEMM of shape (M x K) @ (K x N) with tiles BM x BN:

    X bytes = (N / BN) * M * K * sizeof(T) * R
    W bytes = (M / BM) * N * K * w_bytes_per_element

`R` is the intra-threadgroup activation re-read factor. With no `Xs` staging
each of the WN simdgroup columns loads the same rows independently, so R = WN.
`w_bytes_per_element` for affine 4-bit group-64 is 0.5 for the nibble plus one
scale and one bias of type T per group of 64.

Total = M * N * K * (R * sizeof(T) / BN + w_bytes / BM), so the ratio between
two tilings depends ONLY on BM, BN, sizeof(T) and w_bytes. It is independent
of M, N and K, and therefore independent of which projection is being run.

Register-neutral retiles keep BM * BN fixed, because the accumulator is
BM * BN / (WM * WN * SIMD_SIZE) per lane.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess

NAX_H = ("Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
         "quantized_nax.h")
NAX_METAL = ("Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
             "quantized_nax.metal")
HOST_CPP = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"

SIMD_SIZE = 32
MAX_TGP_BYTES = 32768
GROUP_SIZE = 64
BITS = 4

# Host grid, frozen at quantized.cpp. Confirmed by `read_host_grid` below.
HOST_BM = HOST_BN = HOST_BK = 64
WM = WN = 2

SEED_ROWS = 512

# text_config of the pinned checkpoint.
HIDDEN = 5120
INTERMEDIATE = 17408
HEAD_DIM = 256
N_Q_HEADS = 24
N_KV_HEADS = 4
N_LAYERS = 64


def source(path: str) -> str:
    return pathlib.Path(path).read_text()


def read_host_grid(launcher: str = "qmm_nax") -> dict:
    """Confirm the launch geometry rather than trusting the constants above.

    `quantized.cpp` holds a dozen launchers and several of them declare `bn`
    and `bk`. Scope the search to the named function body, otherwise the
    first match is `qmv`, which launches a different grid entirely.
    """
    text = source(HOST_CPP)
    start = text.index("\nvoid %s(" % launcher)
    end = text.index("\nvoid ", start + 1)
    body = text[start:end]
    got = {}
    for name in ("wm", "wn", "bm", "bn", "bk"):
        # `qmm` declares no `bk` at the host; the kernel template supplies it.
        found = re.search(r"\bint %s = (\d+);" % name, body)
        got[name] = int(found.group(1)) if found else None
    return {
        "launcher": launcher,
        "constants": got,
        "grid_dims": re.search(r"MTL::Size grid_dims\((.+?)\);", body).group(1),
        "group_dims": re.search(r"MTL::Size group_dims\((.+?)\);", body).group(1),
    }


def read_instantiation() -> dict:
    """Which (BM, BK, BN, WM, WN) do the transposed NAX entry points get?"""
    text = source(NAX_METAL)
    rows = re.findall(
        r"instantiate_quantized_aligned(?:_batched)?\(\s*(affine_\w*qmm_t_nax)"
        r"[^)]*?,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)",
        text)
    shapes = sorted({(r[0], *(int(v) for v in r[1:])) for r in rows})
    default_bk = int(re.search(
        r"\[\[kernel\]\] void affine_qmm_t_nax", text[:0] + source(NAX_H)
    ) and re.search(
        r"const int BK = (\d+),\s*\n\s*const int BN = \d+,\s*\n\s*const int WM"
        r" = \d+,\s*\n\s*const int WN = \d+>\s*\n\[\[kernel\]\] void"
        r" affine_qmm_t_nax", source(NAX_H)).group(1))
    return {"instantiated": shapes, "template_default_BK": default_bk}


def stages_activation() -> dict:
    """The load-bearing fact: NAX has no `Xs`, the non-NAX path has one."""
    nax = source(NAX_H)
    plain = source(
        "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h")
    impl = nax[nax.index("METAL_FUNC void qmm_t_nax_tgp_impl"):]
    impl = impl[:impl.index("\ntemplate <")]
    return {
        "nax_qmm_t_declares_Xs": "Xs" in impl,
        "nax_qmm_t_reads_x_from_device_in_kloop": "Atile.load(xk" in impl,
        "non_nax_qmm_t_declares_Xs": "Xs_tile = BM * BK_padded" in plain,
    }


def tile_bytes(bm: int, bk: int, bn: int, elt: int) -> dict:
    bk_padded = bk + 16 // elt
    tile = bn * bk_padded
    doubled = 2 * tile * elt
    return {
        "BK_padded": bk_padded,
        "Ws_tile_elements": tile,
        "Ws_bytes_single": tile * elt,
        "Ws_bytes_doubled": doubled,
        "pipelined": doubled <= MAX_TGP_BYTES,
    }


def weight_bytes_per_element(elt: int) -> float:
    return BITS / 8 + 2 * elt / GROUP_SIZE


def traffic(bm: int, bn: int, elt: int, reread: int) -> float:
    """Device bytes per unit of M*N*K. Lower is better."""
    return reread * elt / bn + weight_bytes_per_element(elt) / bm


def accumulator_regs(bm: int, bn: int) -> int:
    return bm * bn // (WM * WN * SIMD_SIZE)


def operand_regs(bm: int, bn: int, sk: int = 32) -> dict:
    """Halves per lane held by the A and B operand tiles of one simdgroup."""
    sm, sn = bm // WM, bn // WN
    return {
        "Atile_halves_per_lane": sm * sk // SIMD_SIZE,
        "Btile_halves_per_lane": sn * sk // SIMD_SIZE,
    }


def threadgroups(m: int, n: int, bm: int, bn: int) -> int:
    return ((m + bm - 1) // bm) * ((n + bn - 1) // bn)


def scored_shapes() -> list:
    q = N_Q_HEADS * HEAD_DIM
    kv = N_KV_HEADS * HEAD_DIM
    return [
        ("attn q(+gate)", 2 * q, HIDDEN),
        ("attn k", kv, HIDDEN),
        ("attn v", kv, HIDDEN),
        ("attn o", HIDDEN, q),
        ("mlp gate", INTERMEDIATE, HIDDEN),
        ("mlp up", INTERMEDIATE, HIDDEN),
        ("mlp down", HIDDEN, INTERMEDIATE),
    ]


def main() -> None:
    elt = 2  # bfloat16, the checkpoint dtype
    base = (HOST_BM, HOST_BN)
    candidates = [(128, 32), (64, 64), (32, 128), (16, 256)]

    out: dict = {
        "host_grid": read_host_grid(),
        "instantiation": read_instantiation(),
        "staging": stages_activation(),
        "activation_reread_factor": WN,
        "weight_bytes_per_element_bf16": weight_bytes_per_element(elt),
    }

    print("host grid   %s" % out["host_grid"]["constants"])
    print("grid_dims   %s" % out["host_grid"]["grid_dims"])
    print("instantiated transposed NAX shapes (name, BM, BK, BN, WM, WN):")
    for row in out["instantiation"]["instantiated"]:
        print("   %s" % (row,))
    print("template default BK = %s, overridden by the metal instantiation"
          % out["instantiation"]["template_default_BK"])
    print()
    print("staging facts: %s" % json.dumps(out["staging"]))
    print()

    print("threadgroup bytes, Ws only (BK=64)")
    print("  %-9s %-6s %-10s %-12s %-12s %s"
          % ("tiling", "dtype", "BK_padded", "single", "doubled", "pipelined"))
    tg = {}
    for bm, bn in candidates:
        for name, e in (("bf16", 2), ("f32", 4)):
            t = tile_bytes(bm, HOST_BK, bn, e)
            tg["%dx%d/%s" % (bm, bn, name)] = t
            print("  %-9s %-6s %-10d %-12d %-12d %s"
                  % ("%dx%d" % (bm, bn), name, t["BK_padded"],
                     t["Ws_bytes_single"], t["Ws_bytes_doubled"],
                     t["pipelined"]))
    out["threadgroup_bytes"] = tg
    print()

    ref = traffic(*base, elt, WN)
    ref_w = weight_bytes_per_element(elt) / base[0]
    print("device traffic, bytes per M*N*K, bf16, activation reread %d" % WN)
    print("  two models. `cold` charges every activation re-read. `Xcached`")
    print("  charges none, which is the limit the advisor's rationale assumes.")
    print("  %-9s %-10s %-10s %-10s %-9s %-9s %-8s %s"
          % ("tiling", "X term", "W term", "cold", "vs 64x64", "Xcached",
             "acc regs", "A/B halves per lane"))
    rows = {}
    for bm, bn in candidates:
        x = WN * elt / bn
        w = weight_bytes_per_element(elt) / bm
        ops = operand_regs(bm, bn)
        rows["%dx%d" % (bm, bn)] = {
            "x_term": x, "w_term": w, "total_cold": x + w,
            "ratio_cold": (x + w) / ref, "ratio_x_cached": w / ref_w,
            "accumulator_floats_per_lane": accumulator_regs(bm, bn), **ops,
        }
        print("  %-9s %-10.6f %-10.6f %-10.6f %-9.4f %-9.4f %-8d %d / %d"
              % ("%dx%d" % (bm, bn), x, w, x + w, (x + w) / ref, w / ref_w,
                 accumulator_regs(bm, bn), ops["Atile_halves_per_lane"],
                 ops["Btile_halves_per_lane"]))
    out["traffic"] = rows
    out["models_disagree_in_direction"] = {
        t: (rows[t]["ratio_cold"] - 1.0) * (rows[t]["ratio_x_cached"] - 1.0) < 0
        for t in rows}
    print("  models disagree on the SIGN of the effect: %s"
          % json.dumps(out["models_disagree_in_direction"]))
    print()

    # The local proxy. This host has no `_nax`, so the NAX kernel above cannot
    # execute here at all. The measurable transposed path is the non-NAX
    # `qmm`, which stages Xs and launches a 32x32 grid, so its
    # register-neutral retiles are 64x16 and 16x64, not 128x32.
    local = read_host_grid("qmm")
    out["local_proxy_host_grid"] = local
    lbm, lbn = local["constants"]["bm"], local["constants"]["bn"]
    out["local_proxy_register_neutral_tilings"] = [
        [bm, lbm * lbn // bm] for bm in (16, 32, 64) if (lbm * lbn) % bm == 0]
    print("local proxy `qmm` (non-NAX, stages Xs): bm=%d bn=%d bk=%d"
          % (lbm, lbn, local["constants"]["bk"] or -1))
    print("  register-neutral retiles here are %s, not 128x32"
          % out["local_proxy_register_neutral_tilings"])
    print("  nax_available_on_this_host = False, so a NAX-only retile")
    print("  measures exactly zero in a local ABBA session.")
    print()

    print("threadgroup count at the 512-row seed, per projection")
    print("  %-16s %-8s %-8s %-10s %s"
          % ("projection", "N", "K", "64x64", "  ".join(
              "%dx%d" % c for c in candidates if c != base)))
    counts = {}
    for name, n, k in scored_shapes():
        got = {"%dx%d" % (bm, bn): threadgroups(SEED_ROWS, n, bm, bn)
               for bm, bn in candidates}
        counts[name] = {"N": n, "K": k, **got}
        print("  %-16s %-8d %-8d %-10d %s"
              % (name, n, k, got["64x64"],
                 "  ".join("%d" % got["%dx%d" % c]
                           for c in candidates if c != base)))
    out["threadgroup_counts"] = counts

    exact = {name: all(v == counts[name]["64x64"]
                       for k2, v in counts[name].items() if k2 not in ("N", "K"))
             for name in counts}
    out["threadgroup_count_preserved"] = exact
    print()
    print("threadgroup count preserved for every candidate tiling: %s"
          % all(exact.values()))

    div = {name: {"M mod 128": SEED_ROWS % 128,
                  "N mod 32": counts[name]["N"] % 32,
                  "N mod 128": counts[name]["N"] % 128}
           for name in counts}
    out["divisibility"] = div
    bad128 = [n for n in div if div[n]["N mod 128"]]
    print("N not divisible by 128 (blocks the 32x128 retile): %s"
          % (bad128 or "none"))
    bad32 = [n for n in div if div[n]["N mod 32"]]
    print("N not divisible by 32 (blocks the 128x32 retile): %s"
          % (bad32 or "none"))

    pathlib.Path("research/e147-rungE0.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("\nwrote research/e147-rungE0.json")


if __name__ == "__main__":
    main()
