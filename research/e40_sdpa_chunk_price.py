#!/usr/bin/env python3
"""E40 addendum: price the inherited WIDE-DECODE SDPA chunk (advisor ledger 156).

Zero GPU. Every input is either a source constant (cited inline) or an
authoritative fixture value. Answers the advisor's question -- "how much does
the chunk cost us at widths 6-9, and is that cost the size of the deficit?" --
and adjudicates the inverse hypothesis from the dispatch predicate.

Sources
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp
      :591-639  ScaledDotProductAttention::use_fallback   (NOT in editablePaths)
  Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift:104-142
      the chunk                                            (editablePaths[7])
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:1918
      the sole scored attention entry point
  fixtures/qwen3_8_27b_mtp_track.json:108-110
      64 layers, full_attention_interval 4, 24/4 heads, head_dim 256
"""

# ---------------------------------------------------------------- geometry
N_LAYERS = 64
FULL_ATTN_INTERVAL = 4
N_FULL_ATTN = N_LAYERS // FULL_ATTN_INTERVAL          # 16
N_Q_HEADS, N_KV_HEADS = 24, 4
GQA = N_Q_HEADS // N_KV_HEADS                          # 6
HEAD_DIM = 256
BYTES_PER_ELEM = 2                                     # bf16 KV cache

# --------------------------------------------- use_fallback, verbatim
VECTOR_OK_HEAD_DIMS = (64, 96, 128, 256)               # :621-624
FULL_OK_HEAD_DIMS = (64, 80, 128)                      # :625-626


def use_fallback(qL, kL, head_dim=HEAD_DIM, gqa=GQA, do_causal=True):
    """Verbatim port of scaled_dot_product_attention.cpp:621-639."""
    vec_hd = head_dim in VECTOR_OK_HEAD_DIMS
    full_hd = head_dim in FULL_OK_HEAD_DIMS
    full_mask = qL <= kL and do_causal                              # :628-629
    supports_full = qL > 8 and full_mask and full_hd                # :631-632
    supports_vector = (qL <= 8) and (qL <= kL) and vec_hd \
        and (qL * gqa) <= 32                                        # :634-637
    return not (supports_full or supports_vector), supports_full, supports_vector


def route(qL, kL):
    fb, sf, sv = use_fallback(qL, kL)
    return "EAGER FALLBACK" if fb else ("sdpa_full" if sf else "sdpa_vector")


# ------------------------------------------- exactness: window algebra
def windows_full(qL, kL):
    """Bottom-right-aligned causal: row i sees keys 0..kL-qL+i."""
    return [set(range(0, kL - qL + i + 1)) for i in range(qL)]


def windows_chunked(qL, kL, split=5):
    kSplit = kL - (qL - split)
    a = [set(range(0, kSplit - split + i + 1)) for i in range(split)]
    b = [set(range(0, kL - (qL - split) + j + 1)) for j in range(qL - split)]
    return a + b


def verify_window_equivalence():
    bad = []
    for qL in range(6, 10):
        for kL in range(qL, 1200):
            if windows_full(qL, kL) != windows_chunked(qL, kL):
                bad.append((qL, kL))
    return bad


# ---------------------------------------------------------- KV economics
def kv_bytes_per_layer(rows):
    """K and V, both [1, n_kv, rows, head_dim]."""
    return 2 * N_KV_HEADS * rows * HEAD_DIM * BYTES_PER_ELEM


def extra_bytes_per_round(qL, kL, split=5):
    """Chunk reads kSplit + kL rows; a single fused call would read kL."""
    kSplit = kL - (qL - split)
    return N_FULL_ATTN * kv_bytes_per_layer(kSplit)


MB = 1024 ** 2
BW_M4PRO = 273e9    # bytes/s, this box

# beagle, exactly recovered by askeladd and reproduced in E40 section 4
BEAGLE_R = 107
BEAGLE_DEFICIT_PCT = 0.3631
BEAGLE_DEFICIT_MS = 22.63
BEAGLE_LEG_MS = BEAGLE_DEFICIT_MS / (BEAGLE_DEFICIT_PCT / 100.0)

# MDE for the paired local_microbench instrument, from research/e39_mde.py
MDE_NORMAL_PCT = 0.3758
MDE_EXACT_PCT = 0.5040

# thorfinn's M=6 step split, per advisor correction
THORFINN_STEP_MS = 32.850
THORFINN_WEIGHT_MS = 15.401
THORFINN_RESIDUAL_MS = 17.448


def main():
    print("=" * 72)
    print("1. DISPATCH TRUTH TABLE  (head_dim=%d, gqa=%d)" % (HEAD_DIM, GQA))
    print("=" * 72)
    print("  qL*gqa<=32  <=>  qL <= %d" % (32 // GQA))
    print("  sdpa_full needs head_dim in %s; ours is %d -> NEVER"
          % (FULL_OK_HEAD_DIMS, HEAD_DIM))
    print()
    print("  %-6s %-8s %-14s %s" % ("qL", "qL*gqa", "route", "chunk fires?"))
    for qL in (1, 2, 3, 4, 5, 6, 7, 8, 9, 512):
        kL = max(768, qL)
        fires = "YES -> 5 + %d" % (qL - 5) if 6 <= qL <= 9 else "no"
        print("  %-6d %-8d %-14s %s" % (qL, qL * GQA, route(qL, kL), fires))
    print()
    print("  Chunk halves: A qL=5 -> 5*%d=%d <=32 OK ; B qL=1..4 -> <=%d OK"
          % (GQA, 5 * GQA, 4 * GQA))
    print("  => the chunk converts 4 fallback widths into fused vector calls.")

    print()
    print("=" * 72)
    print("2. EXACTNESS: window equivalence, all qL 6..9 x kL qL..1199")
    print("=" * 72)
    bad = verify_window_equivalence()
    assert not bad, bad
    print("  mismatches: %d  -> chunked windows == unchunked windows EXACTLY"
          % len(bad))
    kL = 9
    print("  advisor's flagged corner kL==qL: qL=9,kL=9 -> kSplit=%d"
          % (kL - (9 - 5)))
    print("  chunk A sees a square 5x5 window; alignment holds (checked above).")
    print("  NOTE: this proves the MASK/window algebra only. Reduction order")
    print("        inside the kernel is a separate question -- see section 5.")

    print()
    print("=" * 72)
    print("3. PRICE OF THE CHUNK  (vs a hypothetical single fused call)")
    print("=" * 72)
    print("  full-attn layers %d ; K+V bytes/layer/row = %d"
          % (N_FULL_ATTN, kv_bytes_per_layer(1)))
    print()
    print("  %-6s %-8s %-14s %-12s" % ("kL", "qL", "extra MB", "us @273GB/s"))
    for kL in (512, 768, 1024):
        for qL in (6, 9):
            eb = extra_bytes_per_round(qL, kL)
            print("  %-6d %-8d %-14.1f %-12.1f"
                  % (kL, qL, eb / MB, eb / BW_M4PRO * 1e6))
    mid = extra_bytes_per_round(7, 768)
    us_round = mid / BW_M4PRO * 1e6
    print()
    print("  representative (kL=768 mean, qL=7): %.1f MB -> %.1f us/wide round"
          % (mid / MB, us_round))

    print()
    print("=" * 72)
    print("4. IS IT THE SIZE OF THE DEFICIT?")
    print("=" * 72)
    print("  beagle leg = %.2f ms (%.2f ms deficit / %.4f %%)"
          % (BEAGLE_LEG_MS, BEAGLE_DEFICIT_MS, BEAGLE_DEFICIT_PCT))
    print("  R = %d rounds" % BEAGLE_R)
    print()
    print("  %-28s %-12s %-12s" % ("fraction of rounds M>=6", "total ms", "% of leg"))
    for f in (0.25, 0.5, 0.75, 1.0):
        tot = BEAGLE_R * f * us_round / 1000.0
        print("  %-28.2f %-12.2f %-12.4f"
              % (f, tot, tot / BEAGLE_LEG_MS * 100))
    ceiling = BEAGLE_R * 1.0 * us_round / 1000.0
    ceiling_pct = ceiling / BEAGLE_LEG_MS * 100
    print()
    print("  CEILING (f=1, zero cache reuse): %.4f %% of the beagle leg"
          % ceiling_pct)
    print("  instrument MDE  normal %.4f %% | exact(df=4) %.4f %%"
          % (MDE_NORMAL_PCT, MDE_EXACT_PCT))
    print("  ceiling / MDE_exact = %.2f  -> %s"
          % (ceiling_pct / MDE_EXACT_PCT,
             "UNDETECTABLE even at the ceiling"
             if ceiling_pct < MDE_EXACT_PCT else "detectable"))
    print()
    kv_layer_mb = kv_bytes_per_layer(768) / MB
    print("  lower bound: KV for ONE layer at kL=768 is %.2f MB; chunk A and B"
          % kv_layer_mb)
    print("  run back-to-back on the same layer, so an SLC-resident second read")
    print("  makes the true price << the bandwidth bound. Unmeasured.")

    print()
    print("=" * 72)
    print("5. CROSS-CONNECTION: thorfinn's unexplained M=6 residual")
    print("=" * 72)
    print("  step %.3f ms = weight stream %.3f + residual %.3f"
          % (THORFINN_STEP_MS, THORFINN_WEIGHT_MS, THORFINN_RESIDUAL_MS))
    implied = THORFINN_RESIDUAL_MS * 1000.0 / BEAGLE_R
    print("  residual spread over R=%d rounds = %.1f us/round" % (BEAGLE_R, implied))
    print("  source-derived chunk bound            = %.1f us/round" % us_round)
    print("  ratio implied/bound = %.3f" % (implied / us_round))
    print("  -> the residual is CONSISTENT with the chunk's second KV pass,")
    print("     which is structural and not removable in the editable surface.")
    print("     Depends on thorfinn's step definition, which I do not have.")

    print()
    print("=" * 72)
    print("6. VERDICT ON THE INVERSE HYPOTHESIS")
    print("=" * 72)
    print("  H5: 'rivals who do not chunk are FASTER at widths 6-9, same ids'")
    print("  The non-chunking alternative at qL 6..8 is not another fused")
    print("  kernel -- use_fallback is TRUE, so it is the unfused eager graph")
    print("  (materialised [1,%d,qL,kL] scores + separate matmul/softmax/matmul)." % N_Q_HEADS)
    print("  At qL=9, sdpa_full is ALSO unreachable (head_dim %d not in %s)."
          % (HEAD_DIM, FULL_OK_HEAD_DIMS))
    print("  => a non-chunking rival is SLOWER at every width 6..9, not faster.")
    print("  => H5 predicts the WRONG SIGN. REFUTED from source, zero GPU.")


if __name__ == "__main__":
    main()
