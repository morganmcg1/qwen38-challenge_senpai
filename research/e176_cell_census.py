#!/usr/bin/env python3
"""E176 task 1 (revision r1): the decode cell census, cell by cell.

`harness=source`, with one `harness=local` device-metadata input (this host's
Metal architecture string).

TREE PROVENANCE. The scored decode path is the VENDORED tree.
``Sources/MLXFastModel/Qwen36MTPBlockSession.swift`` imports ``MLXLLM``
(line 4) and drives ``Qwen35TextModelInner.callAsFunction`` (line 775), which
lives in ``Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift``.  The
head lives in ``Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift``.
``Sources/MLXFastModel/Qwen35Ops.swift`` and ``Qwen35FastEngine`` are a
PARALLEL implementation referenced only from inside ``Sources/MLXFastModel/``;
the r0 census read that tree and its cell table is retained below only as a
labelled non-scored reference.

Two independent barriers keep Q (RULE 377: edits ``qmm_t``, ``qmm_t_nax`` and
the dead ``qmm_t_splitk`` only) out of every decode round.

Barrier 1, the candidate-owned replica.  ``Qwen35CustomQMV.matmul``
(Qwen35.swift:1841), reached through ``qwen35RoutedQuantizedMM`` (:1895) and
``qwen35RoutedLinear`` (:1918), consumes any cell accepted by ``routable()``
(:1767) and dispatches its own Metal kernel (:1598/:1881).  Those cells never
reach MLX.  ``routable()`` gates on dtype, affine 4-bit group-64, ``k % 512 ==
0``, ``n % 8 == 0``, ``n >= 4096`` (:1780), width ``m in 2...9`` (:1704) and
row contiguity.  It has NO ``arch_gen`` term, so barrier 1 holds on every GPU
generation.

Barrier 2, the MLX dispatcher.  Cells that reach MLX
(``QuantizedMatmul::eval_gpu``,
``Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp:1415``) take

    int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
    if (M >= vector_limit) { qmm_splitk or qmm; return; }
    dispatch_qmv(...);

``get_qmv_batch_limit`` (quantized.cpp:84-125) returns 6 only under
``arch_gen == 13 || arch_gen == 14``; every other generation takes the
18/12/10 table (32/18/12 for size ``d``).  Both narrow branches are a
conjunction on ``D`` AND ``O``, and every scored cell has ``D = K >= 5120 >
4096``, so the limit is 10 on this host and on any nax-capable ranked host.

Run: ``python3 research/e176_cell_census.py``
"""

import json

SCORED_TREE = "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models"
PARALLEL_TREE = "Sources/MLXFastModel"

# Shapes from fixtures/qwen3_6_27b_config.json (text_config): hidden 5120,
# intermediate 17408, vocab 248320, 64 layers = 48 linear_attention + 16
# full_attention, head_dim 256, 24 query heads, 4 kv heads,
# attn_output_gate true, GDN 16 key heads x 128 and 48 value heads x 128.
H, I, V = 5120, 17408, 248320
KEY, VAL, NV = 16 * 128, 48 * 128, 48        # 2048, 6144, 48
GDN_IN = 2 * KEY + VAL + VAL + NV + NV       # 16480 = [qkv | z | b | a]
GDN_OUT_K = VAL                              # 6144
FA_Q = 24 * 256 * 2                          # 12288, doubled by output gate
FA_KV = 4 * 256                              # 1024
FA_QKV = FA_Q + 2 * FA_KV                    # 14336
FA_O_K = 24 * 256                            # 6144
MLP_GATE_UP = 2 * I                          # 34816

GDN_LAYERS, FA_LAYERS, ALL_LAYERS = 48, 16, 64
MAX_DECODE_M = 9  # 1 committed primary + at most 8 drafts

# front: "replica" = reaches Qwen35CustomQMV.routable before MLX.
#        "mlx"     = direct QuantizedLinear / quantizedMM call, no replica.
# Fields: name, K, N, calls per decode round, bits, group size, front, site.
TARGET_CELLS = [
    ("mlp.gate_up (fused)", H, MLP_GATE_UP, ALL_LAYERS, 4, 64, "replica",
     "Qwen35.swift:1948 Qwen35FusedMLP.fusedGateUp"),
    ("mlp.down", I, H, ALL_LAYERS, 4, 64, "replica",
     "Qwen35.swift:1985 qwen35RoutedLinear(downProj)"),
    ("gdn.in_proj (fused)", H, GDN_IN, GDN_LAYERS, 4, 64, "replica",
     "Qwen35.swift:809 fusedInProjections"),
    ("gdn.out_proj", GDN_OUT_K, H, GDN_LAYERS, 4, 64, "replica",
     "Qwen35.swift:1322 qwen35RoutedLinear(outProj)"),
    ("fa.qkv (fused)", H, FA_QKV, FA_LAYERS, 4, 64, "replica",
     "Qwen35.swift:3098 Qwen35Attention.qkv"),
    ("fa.o_proj", FA_O_K, H, FA_LAYERS, 4, 64, "replica",
     "Qwen35.swift:3398 qwen35RoutedLinear(oProj)"),
    ("lm_head", H, V, 1, 4, 64, "replica",
     "Qwen35.swift:5602 routedLMHead"),
]

# The exact-KV island fast path splits fa.qkv instead of fusing all four
# outputs. Both forms are censused; neither adds a sub-4096 quantized cell.
TARGET_CELL_VARIANTS = [
    ("fa.q_gate (island path)", H, FA_Q, FA_LAYERS, 4, 64, "replica",
     "Qwen35.swift:3078 Qwen35Attention.qkv, _qOnlyW branch"),
    ("fa.kv (island path, BF16 dense)", H, 2 * FA_KV, FA_LAYERS, 16, 0, "mlx",
     "Qwen35.swift:3086 matmul(x, kvExact.T) - not quantized at all"),
]

# The MTP head runs autoregressively at M = 1 per draft step, so its cells are
# censused at every width 1..9 rather than by a fixed call count.
HEAD_CELLS = [
    ("mtp.fc", 2 * H, H, 4, 64, "mlx",
     "Qwen35MTP.swift:194 fc(...) - plain QuantizedLinear, no replica front"),
    ("mtp.layer fa.qkv (fused)", H, FA_QKV, 4, 64, "replica",
     "Qwen35MTP.swift:37 Qwen35Attention inside Qwen35MTPDecoderLayer"),
    ("mtp.layer fa.o_proj", FA_O_K, H, 4, 64, "replica",
     "Qwen35MTP.swift:37 Qwen35Attention inside Qwen35MTPDecoderLayer"),
    ("mtp.layer mlp.gate_up", H, MLP_GATE_UP, 4, 64, "replica",
     "Qwen35MTP.swift:50 Qwen35FusedMLP inside Qwen35MTPDecoderLayer"),
    ("mtp.layer mlp.down", I, H, 4, 64, "replica",
     "Qwen35MTP.swift:50 Qwen35FusedMLP inside Qwen35MTPDecoderLayer"),
    ("draft_lm_head (2-bit)", H, V, 2, 64, "mlx",
     "Qwen35.swift:5613 quantizedMM - bits 2, replica declines"),
    ("draft centroid (2-bit)", H, 1024, 2, 64, "mlx",
     "Qwen35.swift:5823 quantizedMM - bits 2, replica declines"),
]

# r0 census of the PARALLEL, NON-SCORED tree. Retained for the record only.
PARALLEL_CELLS = [
    ("gdn.in_qkv", H, 2 * KEY + VAL, GDN_LAYERS, "Qwen35GatedDelta.swift:241"),
    ("gdn.in_z", H, VAL, GDN_LAYERS, "Qwen35GatedDelta.swift:245"),
    ("gdn.in_b", H, NV, GDN_LAYERS, "Qwen35GatedDelta.swift:254"),
    ("gdn.in_a", H, NV, GDN_LAYERS, "Qwen35GatedDelta.swift:255"),
    ("gdn.out", VAL, H, GDN_LAYERS, "Qwen35GatedDelta.swift:346"),
    ("attn.q", H, 24 * 256, FA_LAYERS, "Qwen35Attention.swift:143"),
    ("attn.k", H, FA_KV, FA_LAYERS, "Qwen35Attention.swift:162"),
    ("attn.v", H, FA_KV, FA_LAYERS, "Qwen35Attention.swift:163"),
    ("attn.o", 24 * 256, H, FA_LAYERS, "Qwen35Attention.swift:211"),
    ("mlp.gate", H, I, ALL_LAYERS, "Qwen35MLP.swift:28"),
    ("mlp.up", H, I, ALL_LAYERS, "Qwen35MLP.swift:29"),
    ("mlp.down", I, H, ALL_LAYERS, "Qwen35MLP.swift:30"),
    ("lm_head", H, V, 1, "Qwen35FastEngine.swift:263"),
]

# (label, arch_gen, arch_size, provenance)
BRANCHES = [
    ("gen13/14 size s", 13, "s", "M1/M2 class, not the ranked runner"),
    ("gen13/14 size d", 13, "d", "Ultra, not the ranked runner"),
    ("gen16 size s", 16, "s", "first-hand probe, this host (applegpu_g16s)"),
    ("gen>=17 size s", 17, "s", "ranked M5, requires arch_gen>=17 for _nax"),
    ("gen>=17 size d", 17, "d", "Ultra, not the ranked runner"),
]


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


def replica_takes(k, n, bits, group_size, front, m):
    """Qwen35CustomQMV.routable, Qwen35.swift:1767. No arch_gen term."""
    if front != "replica":
        return False
    if bits != 4 or group_size != 64:
        return False
    if k % 512 or n % 8 or n < 4096:
        return False
    return 2 <= m <= 9


def kernel_for(k, n, bits, group_size, front, m, gen, size):
    """The kernel family one cell reaches at width m, after both barriers."""
    if bits == 16:
        return "matmul (BF16 dense, not quantized)"
    if replica_takes(k, n, bits, group_size, front, m):
        return "qwen35CustomAffine4QMV (replica)"
    return ("qmm_t/qmm_t_nax" if m >= qmv_batch_limit(k, n, gen, size)
            else "dispatch_qmv")


def consumes_q(k, n, bits, group_size, front, m, gen, size):
    return kernel_for(
        k, n, bits, group_size, front, m, gen, size).startswith("qmm_t")


def main():
    out = {
        "harness": "source",
        "revision": "r1",
        "max_decode_M": MAX_DECODE_M,
        "scored_tree": SCORED_TREE,
        "scored_entry_point":
            "Qwen36MTPBlockSession.swift:775 -> Qwen35TextModelInner",
        "barriers": {
            "1_replica": "Qwen35.swift:1767 routable(), no arch_gen term, "
                         "widths 2...9, n>=4096; consumed before MLX",
            "2_mlx_limit": "quantized.cpp:1415 with get_qmv_batch_limit "
                           ":84-125; minimum 10 outside arch_gen 13/14",
        },
        "target_cells": [],
        "target_cell_variants": [],
        "head_cells": [],
        "branches": {},
        "parallel_tree_cells": {
            "tree": PARALLEL_TREE,
            "scored": False,
            "note": "r0 census of Sources/MLXFastModel (Qwen35Ops.linear, "
                    "Qwen35FastEngine). NOT the scored path: referenced only "
                    "from inside Sources/MLXFastModel/. Retained for the "
                    "record; withdrawn as a scored-path census.",
            "cells": [{"cell": c[0], "K": c[1], "N": c[2],
                       "calls_per_round": c[3], "source": c[4]}
                      for c in PARALLEL_CELLS],
            "total_calls_per_round": sum(c[3] for c in PARALLEL_CELLS),
        },
    }

    total = sum(c[3] for c in TARGET_CELLS)
    print("=" * 78)
    print("E176 TASK 1 r1: scored-tree decode cell census (harness=source)")
    print("=" * 78)
    print("  scored tree: %s/Qwen35.swift + Qwen35MTP.swift" % SCORED_TREE)
    print("  %d quantized target calls per decode round, %d fused cells"
          % (total, len(TARGET_CELLS)))
    print("  source agrees: 'all 257 wide QMV calls of one decode round'"
          " (Qwen35.swift:1737)")
    print()

    print("  TARGET CELLS")
    print("  %-24s %7s %8s %7s %6s %10s" %
          ("cell", "K", "N", "calls", "bits", "routable"))
    for name, k, n, calls, bits, gs, front, site in TARGET_CELLS:
        routable = replica_takes(k, n, bits, gs, front, 2)
        print("  %-24s %7d %8d %7d %6d %10s"
              % (name, k, n, calls, bits, "yes" if routable else "no"))
        out["target_cells"].append({
            "cell": name, "K": k, "N": n, "calls_per_round": calls,
            "bits": bits, "group_size": gs, "front": front, "source": site,
            "tree": SCORED_TREE, "scored": True, "transpose": True,
            "replica_routable_at_M2_9": routable,
            "limits": {b[0]: qmv_batch_limit(k, n, b[1], b[2])
                       for b in BRANCHES},
            "kernel_by_M": {
                str(m): kernel_for(k, n, bits, gs, front, m, 16, "s")
                for m in range(1, MAX_DECODE_M + 1)},
        })
    print()

    print("  TARGET CELL VARIANT: exact-KV island fast path replaces fa.qkv")
    for name, k, n, calls, bits, gs, front, site in TARGET_CELL_VARIANTS:
        routable = replica_takes(k, n, bits, gs, front, 2)
        label = ("n/a (BF16 dense)" if bits == 16
                 else ("yes" if routable else "no"))
        print("  %-34s K=%-6d N=%-6d routable=%s" % (name, k, n, label))
        out["target_cell_variants"].append({
            "cell": name, "K": k, "N": n, "calls_per_round": calls,
            "bits": bits, "group_size": gs, "front": front, "source": site,
            "tree": SCORED_TREE, "scored": True,
            "replica_routable_at_M2_9": routable,
            "kernel_by_M": {
                str(m): kernel_for(k, n, bits, gs, front, m, 16, "s")
                for m in range(1, MAX_DECODE_M + 1)},
        })
    print()

    print("  MTP HEAD CELLS (autoregressive drafting runs them at M=1)")
    print("  %-28s %7s %8s %6s  %s" %
          ("cell", "K", "N", "bits", "kernels over M=1..9, gen16/gen>=17"))
    for name, k, n, bits, gs, front, site in HEAD_CELLS:
        fams = {kernel_for(k, n, bits, gs, front, m, 16, "s")
                for m in range(1, MAX_DECODE_M + 1)}
        print("  %-28s %7d %8d %6d  %s"
              % (name, k, n, bits, " / ".join(sorted(fams))))
        out["head_cells"].append({
            "cell": name, "K": k, "N": n, "bits": bits, "group_size": gs,
            "front": front, "source": site, "tree": SCORED_TREE,
            "scored": True, "widths_in_decode": "1 per draft step",
            "limits": {b[0]: qmv_batch_limit(k, n, b[1], b[2])
                       for b in BRANCHES},
            "kernel_by_M": {
                str(m): kernel_for(k, n, bits, gs, front, m, 16, "s")
                for m in range(1, MAX_DECODE_M + 1)},
        })
    print()

    print("  Q-CONSUMING TARGET CALLS PER ROUND, BY WIDTH AND HOST CLASS")
    print("    branch          " + " ".join("M=%d" % m for m in range(1, 10)))
    for label, gen, size, prov in BRANCHES:
        counts = []
        for m in range(1, MAX_DECODE_M + 1):
            counts.append(sum(
                c[3] for c in TARGET_CELLS
                if consumes_q(c[1], c[2], c[4], c[5], c[6], m, gen, size)))
        head_hits = sum(
            1 for c in HEAD_CELLS
            for m in range(1, MAX_DECODE_M + 1)
            if consumes_q(c[1], c[2], c[3], c[4], c[5], m, gen, size))
        print("    %-15s " % label + " ".join("%3d" % c for c in counts)
              + "   head cell-widths consuming Q: %d" % head_hits)
        out["branches"][label] = {
            "arch_gen": gen, "arch_size": size, "provenance": prov,
            "limit_all_scored_cells": sorted(
                {qmv_batch_limit(c[1], c[2], gen, size)
                 for c in TARGET_CELLS}),
            "q_consuming_target_calls_by_M": {
                str(m + 1): counts[m] for m in range(MAX_DECODE_M)},
            "q_consuming_head_cell_widths": head_hits,
            "first_consuming_M": next(
                (m + 1 for m in range(MAX_DECODE_M) if counts[m]), None)}
    print()

    print("  Barrier 1 alone removes every target call at M=2..9 on EVERY")
    print("  generation: routable() has no arch_gen term, and all seven fused")
    print("  cells satisfy n>=4096, n%8==0, k%512==0, affine 4-bit g64.")
    print("  Barrier 2 removes the rest: every scored cell has K>=5120>4096,")
    print("  so neither narrow branch of get_qmv_batch_limit applies and the")
    print("  limit is a single constant per host class:")
    scored_shapes = [(c[1], c[2]) for c in TARGET_CELLS]
    scored_shapes += [(c[1], c[2]) for c in HEAD_CELLS]
    for label, gen, size, _ in BRANCHES:
        vals = sorted({qmv_batch_limit(k, n, gen, size)
                       for k, n in scored_shapes})
        print("    %-15s limit = %s for every scored cell" % (label, vals))
    print()
    print("  Fusion removes the small-N cells entirely: k and v (N=1024) live")
    print("  inside fa.qkv, b and a (N=48) inside gdn.in_proj. They are not")
    print("  dispatched, so they cannot be Q's decode consumers.")

    out["total_target_calls_per_round"] = total
    out["q_consuming_calls_at_M_le_5"] = 0
    out["q_consuming_calls_at_M_le_9"] = 0
    with open("research/e176-cell-census.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print()
    print("wrote research/e176-cell-census.json")


if __name__ == "__main__":
    main()
