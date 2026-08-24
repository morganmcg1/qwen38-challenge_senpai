#!/usr/bin/env python3
"""E176 task 1: the decode cell census, cell by cell (harness=source).

Every quantized projection in the scored decode path goes through exactly one
Swift entry point, ``Qwen35Ops.linear`` (`Sources/MLXFastModel/Qwen35Ops.swift`
line 36), which calls MLX ``quantizedMM(transpose: true)``.  There is no custom
quantized matvec replica in this tree.  MLX then decides qmv against qmm in
``QuantizedMatmul::eval_gpu``
(`Vendor/mlx-swift/.../backend/metal/quantized.cpp` line 1415):

    int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
    if (M >= vector_limit) { qmm_splitk or qmm; return; }
    dispatch_qmv(...);

Q (RULE 377) edits only ``qmm_t``, ``qmm_t_nax`` and ``qmm_t_splitk``, so a
cell consumes Q only when it reaches the ``M >= vector_limit`` branch with
``transpose_ == true``.

Run: ``python3 research/e176_cell_census.py``
"""

import json

# Shapes derived from fixtures/qwen3_6_27b_config.json (text_config):
#   hidden 5120, intermediate 17408, vocab 248320,
#   24 attention heads, 4 kv heads, head_dim 256,
#   GDN: 16 key heads x 128, 48 value heads x 128.
H, I, V = 5120, 17408, 248320
Q_OUT = 24 * 256          # 6144
KV_OUT = 4 * 256          # 1024
KEY = 16 * 128            # 2048
VAL = 48 * 128            # 6144
GDN_QKV = 2 * KEY + VAL   # 10240, == convolutionDimension
NV = 48                   # linear_num_value_heads

GDN_LAYERS, FA_LAYERS = 48, 16
ALL_LAYERS = GDN_LAYERS + FA_LAYERS

# (name, K, N, calls per decode round, source)
CELLS = [
    ("gdn.in_qkv", H, GDN_QKV, GDN_LAYERS, "Qwen35GatedDelta.swift:241"),
    ("gdn.in_z", H, VAL, GDN_LAYERS, "Qwen35GatedDelta.swift:245"),
    ("gdn.in_b", H, NV, GDN_LAYERS, "Qwen35GatedDelta.swift:254"),
    ("gdn.in_a", H, NV, GDN_LAYERS, "Qwen35GatedDelta.swift:255"),
    ("gdn.out", VAL, H, GDN_LAYERS, "Qwen35GatedDelta.swift:346"),
    ("attn.q", H, Q_OUT, FA_LAYERS, "Qwen35Attention.swift:143"),
    ("attn.k", H, KV_OUT, FA_LAYERS, "Qwen35Attention.swift:162"),
    ("attn.v", H, KV_OUT, FA_LAYERS, "Qwen35Attention.swift:163"),
    ("attn.o", Q_OUT, H, FA_LAYERS, "Qwen35Attention.swift:211"),
    ("mlp.gate", H, I, ALL_LAYERS, "Qwen35MLP.swift:28"),
    ("mlp.up", H, I, ALL_LAYERS, "Qwen35MLP.swift:29"),
    ("mlp.down", I, H, ALL_LAYERS, "Qwen35MLP.swift:30"),
    ("lm_head", H, V, 1, "Qwen35FastEngine.swift:263"),
]

MAX_DECODE_M = 9  # 1 committed primary + at most 8 drafts


def qmv_batch_limit(D, O, arch_gen, arch_size):
    """Verbatim port of get_qmv_batch_limit, quantized.cpp lines 84-125."""
    if arch_gen in (13, 14):
        if arch_size == "d":
            return 32 if (D <= 2048 and O <= 2048) else (
                18 if (D <= 4096 and O <= 4096) else 12)
        return 14 if (D <= 2048 and O <= 2048) else (
            10 if (D <= 4096 and O <= 4096) else 6)
    if arch_size == "d":
        return 32 if (D <= 2048 and O <= 2048) else (
            18 if (D <= 4096 and O <= 4096) else 12)
    return 18 if (D <= 2048 and O <= 2048) else (
        12 if (D <= 4096 and O <= 4096) else 10)


# (label, arch_gen, arch_size, provenance)
BRANCHES = [
    ("gen13/14 size s", 13, "s", "M1/M2 class, not the ranked runner"),
    ("gen13/14 size d", 13, "d", "Ultra, not the ranked runner"),
    ("gen16 size s", 16, "s", "first-hand probe, this host (applegpu_g16s)"),
    ("gen>=17 size s", 17, "s", "ranked M5, requires arch_gen>=17 for _nax"),
    ("gen>=17 size d", 17, "d", "Ultra, not the ranked runner"),
]


def main():
    out = {"harness": "source", "max_decode_M": MAX_DECODE_M, "cells": [],
           "branches": {}}
    total = sum(c[3] for c in CELLS)
    print("=" * 78)
    print("E176 TASK 1: decode cell census (harness=source)")
    print("=" * 78)
    print("  %d quantized-projection calls per decode round, %d distinct cells"
          % (total, len(CELLS)))
    print("  all reach MLX quantizedMM(transpose: true) via Qwen35Ops.linear")
    print()
    header = "  %-12s %7s %8s %7s  " % ("cell", "K", "N", "calls")
    header += " ".join("%-9s" % b[0] for b in BRANCHES)
    print(header)
    for name, k, n, calls, src in CELLS:
        limits = {b[0]: qmv_batch_limit(k, n, b[1], b[2]) for b in BRANCHES}
        row = "  %-12s %7d %8d %7d  " % (name, k, n, calls)
        row += " ".join("%-9d" % limits[b[0]] for b in BRANCHES)
        print(row)
        out["cells"].append({"cell": name, "K": k, "N": n,
                             "calls_per_round": calls, "source": src,
                             "transpose": True, "limits": limits})
    print()

    print("  CONSUMERS OF Q (cells reaching qmm_t / qmm_t_nax) BY WIDTH")
    print("    branch          " + " ".join("M=%d" % m for m in range(1, 10)))
    for label, gen, size, prov in BRANCHES:
        counts = []
        for m in range(1, MAX_DECODE_M + 1):
            counts.append(sum(c[3] for c in CELLS
                              if m >= qmv_batch_limit(c[1], c[2], gen, size)))
        print("    %-15s " % label + " ".join("%3d" % c for c in counts))
        out["branches"][label] = {
            "arch_gen": gen, "arch_size": size, "provenance": prov,
            "q_consuming_calls_by_M": {str(m + 1): counts[m]
                                       for m in range(MAX_DECODE_M)},
            "first_consuming_M": next(
                (m + 1 for m in range(MAX_DECODE_M) if counts[m]), None)}
    print()
    print("  Every cell has K >= %d > 4096, so no cell can take the"
          % min(c[1] for c in CELLS))
    print("  (D<=2048 && O<=2048) or (D<=4096 && O<=4096) branch.  The limit")
    print("  is therefore a single constant per host class, independent of N:")
    for label, gen, size, _ in BRANCHES:
        vals = {qmv_batch_limit(c[1], c[2], gen, size) for c in CELLS}
        print("    %-15s limit = %s for all %d cells"
              % (label, sorted(vals), len(CELLS)))
    print()
    print("  The small-N cells raised as candidate Q consumers (attn.k and")
    print("  attn.v at N=1024, gdn.in_b and gdn.in_a at N=48) do NOT get a")
    print("  lower limit: the guard is a conjunction on D AND O, and D=5120")
    print("  fails it.  They route to dispatch_qmv exactly like every other")
    print("  cell.")
    out["total_calls_per_round"] = total
    with open("research/e176-cell-census.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print()
    print("wrote research/e176-cell-census.json")


if __name__ == "__main__":
    main()
