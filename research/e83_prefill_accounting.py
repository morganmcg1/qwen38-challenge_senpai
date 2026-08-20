#!/usr/bin/env python3
"""E83 rung 0 - static FLOP and byte accounting for the charged 512-token seed.

Every shape here is read from the live source, not from a prior campaign note:

  Qwen36MTPBlockSession.begin()                     the charged phase list
  Qwen35GatedDeltaNet.init / fusedInProjections     GDN projection widths
  Qwen35Attention.init / qkv                        FA projection widths
  Qwen35FusedMLP.callAsFunction                     the `x.dim(-2) <= 16` gate
  weights/config.json                               the model dimensions

Research-only: never packaged into a submission.
"""

import json

SEED = 512
H = 5120
INTERMEDIATE = 17408
VOCAB = 248320

N_GDN = 48
N_FA = 16
N_LAYERS = N_GDN + N_FA

# GDN (Qwen35GatedDeltaNet.init)
GDN_KEY_HEADS, GDN_KEY_DIM = 16, 128
GDN_VAL_HEADS, GDN_VAL_DIM = 48, 128
GDN_CONV_K = 4
GDN_KEYDIM = GDN_KEY_HEADS * GDN_KEY_DIM          # 2048
GDN_VALDIM = GDN_VAL_HEADS * GDN_VAL_DIM          # 6144
GDN_CONVDIM = GDN_KEYDIM * 2 + GDN_VALDIM         # 10240

# Full attention (Qwen35Attention.init); q_proj carries the output gate, so its
# N is attentionHeads * headDim * 2.
FA_Q_HEADS, FA_KV_HEADS, FA_HEAD_DIM = 24, 4, 256
FA_Q_OUT = FA_Q_HEADS * FA_HEAD_DIM * 2           # 12288
FA_KV_OUT = FA_KV_HEADS * FA_HEAD_DIM             # 1024
FA_O_IN = FA_Q_HEADS * FA_HEAD_DIM                # 6144

QBITS, QGROUP = 4, 64


def gemm(m, k, n):
    return 2 * m * k * n


def qbytes(k, n):
    """Resident bytes of one affine-4 group-64 weight: packed 4-bit plus a
    bf16 scale and a bf16 bias per group."""
    return n * k * QBITS // 8 + 2 * (2 * n * k // QGROUP)


def main():
    rows = []

    def add(family, count, m, k, n, kind, note=""):
        rows.append(
            {
                "family": family,
                "layers": count,
                "m": m,
                "k": k,
                "n": n,
                "kind": kind,
                "flop_per_layer": gemm(m, k, n),
                "flop_total": count * gemm(m, k, n),
                "weight_bytes_total": count * qbytes(k, n) if kind == "gemm" else 0,
                "note": note,
            }
        )

    # --- GDN layers. `fusedInProjections` is decode-only: prefill (S > 2)
    # keeps the four separate launches so qmm reduction order is unchanged.
    add("gdn.in_proj_qkv", N_GDN, SEED, H, GDN_CONVDIM, "gemm", "separate at S>2")
    add("gdn.in_proj_z", N_GDN, SEED, H, GDN_VALDIM, "gemm", "separate at S>2")
    add("gdn.in_proj_b", N_GDN, SEED, H, GDN_VAL_HEADS, "gemm", "separate at S>2")
    add("gdn.in_proj_a", N_GDN, SEED, H, GDN_VAL_HEADS, "gemm", "separate at S>2")
    add("gdn.out_proj", N_GDN, SEED, GDN_VALDIM, H, "gemm", "")

    # --- Full-attention layers. `qkv()` packs Q+gate, K and V on N with no
    # width guard, so prefill takes the packed GEMM.
    add("fa.qkv_packed", N_FA, SEED, H, FA_Q_OUT + 2 * FA_KV_OUT, "gemm", "packed on N")
    add("fa.o_proj", N_FA, SEED, FA_O_IN, H, "gemm", "")

    # --- MLP. The fused gate_up GEMM is gated on `x.dim(-2) <= 16`, so the
    # 512-row seed runs two separate 5120 -> 17408 GEMMs and an uncompiled
    # silu/multiply, NOT the 5120 -> 34816 shape the decode path uses.
    add("mlp.gate_proj", N_LAYERS, SEED, H, INTERMEDIATE, "gemm", "unfused at S>16")
    add("mlp.up_proj", N_LAYERS, SEED, H, INTERMEDIATE, "gemm", "unfused at S>16")
    add("mlp.down_proj", N_LAYERS, SEED, INTERMEDIATE, H, "gemm", "")

    # --- readout. `seedLogits` over all 512 rows is built and never evaluated
    # (dead lazy graph, receipt b5130678). Only the tail row is projected.
    add("lm_head.tail_row", 1, 1, H, VOCAB, "gemm", "1 of 512 rows; 511 are dead")

    gemm_flop = sum(r["flop_total"] for r in rows)
    gemm_bytes = sum(r["weight_bytes_total"] for r in rows)

    # --- non-GEMM work, priced separately because it does not ride the
    # quantized matmul roofline.
    sdpa_flop = N_FA * (2 * 2 * FA_Q_HEADS * SEED * SEED * FA_HEAD_DIM) // 2
    # gatedDeltaKernel: per token per value head, read k.state, apply the
    # rank-1 delta update and read q.state -> ~6 * Dk * Dv fused ops.
    gdn_scan_flop = N_GDN * SEED * GDN_VAL_HEADS * 6 * GDN_KEY_DIM * GDN_VAL_DIM
    conv_flop = N_GDN * 2 * SEED * GDN_CONVDIM * GDN_CONV_K
    dead_lm_head_flop = gemm(SEED - 1, H, VOCAB)

    # bf16 activation traffic that the seed forward materialises per layer.
    gdn_state_bytes = N_GDN * GDN_VAL_HEADS * GDN_VAL_DIM * GDN_KEY_DIM * 4
    fa_kv_bytes = N_FA * 2 * SEED * FA_KV_OUT * 2

    payload = {
        "seed_tokens": SEED,
        "rows": rows,
        "gemm_flop_total": gemm_flop,
        "gemm_flop_total_tflop": gemm_flop / 1e12,
        "gemm_weight_bytes_total": gemm_bytes,
        "gemm_weight_gib": gemm_bytes / 2**30,
        "non_gemm": {
            "sdpa_causal_flop": sdpa_flop,
            "gdn_scan_flop": gdn_scan_flop,
            "gdn_conv1d_flop": conv_flop,
            "total_flop": sdpa_flop + gdn_scan_flop + conv_flop,
            "share_of_gemm_flop": (sdpa_flop + gdn_scan_flop + conv_flop) / gemm_flop,
        },
        "not_executed": {
            "dead_lm_head_flop": dead_lm_head_flop,
            "dead_lm_head_tflop": dead_lm_head_flop / 1e12,
        },
        "state_bytes": {
            "gdn_recurrent_fp32": gdn_state_bytes,
            "fa_kv_bf16": fa_kv_bytes,
        },
    }
    print(json.dumps(payload, indent=2))

    print("\n# family table (FLOP at a 512-token seed)")
    print(f"{'family':22s} {'x':>3s} {'M':>5s} {'K':>6s} {'N':>7s} "
          f"{'TFLOP':>8s} {'share':>7s}")
    for r in sorted(rows, key=lambda r: -r["flop_total"]):
        print(
            f"{r['family']:22s} {r['layers']:3d} {r['m']:5d} {r['k']:6d} "
            f"{r['n']:7d} {r['flop_total']/1e12:8.4f} "
            f"{100*r['flop_total']/gemm_flop:6.2f}%"
        )
    print(f"{'GEMM TOTAL':22s} {'':3s} {'':5s} {'':6s} {'':7s} "
          f"{gemm_flop/1e12:8.4f} {100.0:6.2f}%")
    nz = payload["non_gemm"]
    print(f"\nnon-GEMM FLOP  sdpa={sdpa_flop/1e9:.1f}G "
          f"gdn_scan={gdn_scan_flop/1e9:.1f}G conv={conv_flop/1e9:.1f}G "
          f"total={nz['total_flop']/1e9:.1f}G "
          f"({100*nz['share_of_gemm_flop']:.2f}% of GEMM FLOP)")
    print(f"quantized weight bytes read once = {gemm_bytes/2**30:.3f} GiB")
    print(f"dead 511-row lm_head not executed = {dead_lm_head_flop/1e12:.3f} TFLOP")


if __name__ == "__main__":
    main()
