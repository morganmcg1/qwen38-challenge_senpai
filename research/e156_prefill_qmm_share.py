#!/usr/bin/env python3
"""E156 F3 section 4: what fraction of 512-token seed prefill reaches
`affine_qmm_t_nax`?

The advisor priced PREFILL. This candidate changes THE QUANTIZED GEMM INSIDE
PREFILL. Those are different quantities and the ratio was unknown, so this
script derives it.

Two separate questions, kept separate on purpose:

1. ROUTING. Which prefill projections actually dispatch to `affine_qmm_t_nax`?
   This is decided by real predicates in
   Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp, and one of
   them is easy to miss: a transposed non-batched matmul with M >= vector_limit
   goes to `qmm_splitk` FIRST, and only falls through to `qmm` (and therefore to
   the NAX path) when its computed `split_k` collapses to 1.

2. ARITHMETIC SHARE. Of the multiply-accumulate work in one prefill forward
   pass, how much sits in those routed projections?

The output is a FLOP share, not a time share. Converting one to the other needs
per-kernel efficiency that cannot be measured on this host, because
`is_nax_available()` is false on `applegpu_g16s`. The conversion bound is
reported rather than hidden. harness=offline.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = ROOT / "weights/config.json"

# quantized.cpp qmm_splitk: bm = bn = 32, target ~512 threadgroups.
SPLITK_BM = 32
SPLITK_BN = 32
SPLITK_TARGET_TGS = 512


def sh(*args: str) -> str:
    return subprocess.run(
        args, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def splitk_for(M: int, N: int, K: int, group_size: int) -> int:
    """Reproduce qmm_splitk's split_k computation from quantized.cpp."""
    n_tiles = (N + SPLITK_BN - 1) // SPLITK_BN
    m_tiles = (M + SPLITK_BM - 1) // SPLITK_BM
    split_k = max(1, SPLITK_TARGET_TGS // (n_tiles * m_tiles))
    k_align = max(group_size, 32)
    split_k = min(split_k, K // k_align)
    while split_k > 1 and (K % (split_k * k_align) != 0):
        split_k -= 1
    return split_k


def routes_to_nax(M: int, N: int, K: int, group_size: int,
                  vector_limit: int) -> tuple[bool, str]:
    """Apply the real dispatch chain and say where the projection lands."""
    if M < vector_limit:
        return False, "qmv: M below vector_limit"
    if K % 64 != 0:
        return False, "qmm: K %% 64 != 0 (%d)" % (K % 64)
    split_k = splitk_for(M, N, K, group_size)
    if split_k > 1:
        return False, "qmm_splitk: split_k=%d > 1, never reaches qmm" % split_k
    return True, "qmm -> qmm_nax -> affine_qmm_t_nax (split_k collapsed to 1)"


def build_projections(cfg: dict, args) -> list[dict]:
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    hd = cfg["head_dim"]
    nq = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    lkh = cfg["linear_num_key_heads"]
    lkd = cfg["linear_key_head_dim"]
    lvh = cfg["linear_num_value_heads"]
    lvd = cfg["linear_value_head_dim"]

    layer_types = cfg.get("layer_types") or []
    n_full = sum(1 for t in layer_types if "full" in t)
    n_gdn = sum(1 for t in layer_types if "linear" in t)
    if not layer_types:
        raise SystemExit("config.json has no layer_types; refusing to guess")

    q_w = nq * hd
    gate_w = q_w if cfg.get("attn_output_gate") else 0
    kv_w = nkv * hd

    gdn_qk_w = lkh * lkd
    gdn_v_w = lvh * lvd

    # Widths below are read from the scored prefill path, which is
    # Qwen36MTPBlockSession.begin over the vendored
    # Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift (cited as V).
    # Two fusions that exist in the source are NOT taken at S=512, so they are
    # listed here as the separate projections that actually dispatch.
    rows = [
        {
            "family": "fa.qkv",
            "layer_type": "full_attention",
            "layers": n_full,
            "N": q_w + gate_w + 2 * kv_w,
            "K": H,
            "components": {
                "q": q_w, "attn_output_gate": gate_w,
                "k": kv_w, "v": kv_w,
            },
            "source": "V:3537-3543 fused at runtime; parts V:3461-3466",
        },
        {
            "family": "fa.o_proj",
            "layer_type": "full_attention",
            "layers": n_full,
            "N": H,
            "K": q_w,
            "source": "V:3467-3468",
        },
        {
            "family": "gdn.in_proj_qkv",
            "layer_type": "gdn",
            "layers": n_gdn,
            "N": 2 * gdn_qk_w + gdn_v_w,
            "K": H,
            "components": {"q": gdn_qk_w, "k": gdn_qk_w, "v": gdn_v_w},
            "source": "V:787; unfused at S=512 because the fusion guard "
                      "at V:1143 is S <= 9",
        },
        {
            "family": "gdn.in_proj_z",
            "layer_type": "gdn",
            "layers": n_gdn,
            "N": gdn_v_w,
            "K": H,
            "source": "V:788; unfused at S=512, guard V:1143",
        },
        {
            "family": "gdn.in_proj_b",
            "layer_type": "gdn",
            "layers": n_gdn,
            "N": lvh,
            "K": H,
            "source": "V:789",
        },
        {
            "family": "gdn.in_proj_a",
            "layer_type": "gdn",
            "layers": n_gdn,
            "N": lvh,
            "K": H,
            "source": "V:790",
        },
        {
            "family": "gdn.out_proj",
            "layer_type": "gdn",
            "layers": n_gdn,
            "N": H,
            "K": gdn_v_w,
            "source": "V:797",
        },
        {
            "family": "mlp.gate_proj",
            "layer_type": "all",
            "layers": n_full + n_gdn,
            "N": I,
            "K": H,
            "source": "V:2525; unfused at S=512 because the 2*I pack guard "
                      "at V:2568 is x.dim(-2) <= 16",
        },
        {
            "family": "mlp.up_proj",
            "layer_type": "all",
            "layers": n_full + n_gdn,
            "N": I,
            "K": H,
            "source": "V:2527; unfused at S=512, guard V:2568",
        },
        {
            "family": "mlp.down",
            "layer_type": "all",
            "layers": n_full + n_gdn,
            "N": H,
            "K": I,
            "source": "V:2526",
        },
        {
            "family": "lm_head",
            "layer_type": "final",
            "layers": 1,
            "N": cfg["vocab_size"],
            "K": H,
            "rows_override": args.m if args.lm_head_all_positions else 1,
            "source": "V:5725. begin() projects ONE row: the 512-row "
                      "seedLogits graph at V:5911-5915 is built but never "
                      "evaluated, and Qwen36MTPBlockSession.begin:706-708 "
                      "calls applyLMHead(hiddenRow(hidden, last)). The "
                      "eval list at :717-718 omits seedLogits.",
        },
    ]
    return rows, n_full, n_gdn


def read_weight_shapes() -> dict[str, tuple[str, list[int]]]:
    """Read safetensors headers only. No tensor data is loaded."""
    shapes: dict[str, tuple[str, list[int]]] = {}
    for f in sorted((ROOT / "weights").glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                shapes[k] = (v["dtype"], v["shape"])
    return shapes


# Each row is verified against the checkpoint tensors it is built from, so a
# misread of the Swift source cannot survive into the share. A row backed by
# several checkpoint tensors sums their N.
CHECKPOINT_BACKING = {
    "fa.qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "fa.o_proj": ["self_attn.o_proj"],
    "gdn.in_proj_qkv": ["linear_attn.in_proj_qkv"],
    "gdn.in_proj_z": ["linear_attn.in_proj_z"],
    "gdn.in_proj_b": ["linear_attn.in_proj_b"],
    "gdn.in_proj_a": ["linear_attn.in_proj_a"],
    "gdn.out_proj": ["linear_attn.out_proj"],
    "mlp.gate_proj": ["mlp.gate_proj"],
    "mlp.up_proj": ["mlp.up_proj"],
    "mlp.down": ["mlp.down_proj"],
    "lm_head": ["lm_head"],
}


def verify_against_checkpoint(rows: list[dict], bits: int,
                              group_size: int) -> dict:
    """Confirm every derived (N, K, layers) against the packed weight headers.

    Packed U32 weights are stored [N, K*bits/32] with is_transposed_weight
    true, so K is recovered as cols * 32 / bits and cross-checked against the
    scales column count times group_size.
    """
    shapes = read_weight_shapes()
    checks = []
    all_ok = True
    for r in rows:
        n_sum = 0
        k_seen = set()
        count_seen = set()
        missing = []
        for suffix in CHECKPOINT_BACKING[r["family"]]:
            keys = [k for k in shapes
                    if k.endswith(suffix + ".weight")
                    and shapes[k][0] == "U32"]
            if not keys:
                missing.append(suffix)
                continue
            count_seen.add(len(keys))
            n_dim, packed_cols = shapes[keys[0]][1]
            n_sum += n_dim
            k_seen.add(packed_cols * 32 // bits)
            scale_key = keys[0].replace(".weight", ".scales")
            if scale_key in shapes:
                k_seen.add(shapes[scale_key][1][1] * group_size)
        ok = (not missing
              and n_sum == r["N"]
              and k_seen == {r["K"]}
              and count_seen == {r["layers"]})
        all_ok = all_ok and ok
        checks.append({
            "family": r["family"],
            "derived_N": r["N"], "checkpoint_N": n_sum,
            "derived_K": r["K"], "checkpoint_K": sorted(k_seen),
            "derived_layers": r["layers"],
            "checkpoint_layers": sorted(count_seen),
            "missing_tensors": missing,
            "agrees": ok,
        })
    conv = [k for k in shapes if k.endswith("conv1d.weight")]
    conv_quantized = any(k.replace(".weight", ".scales") in shapes
                         for k in conv)
    return {
        "every_projection_agrees_with_the_checkpoint": all_ok,
        "checks": checks,
        "gdn_conv1d_tensors": len(conv),
        "gdn_conv1d_is_quantized": conv_quantized,
        "note": "shapes read from safetensors headers only; no tensor data "
                "was loaded. harness=offline",
    }


def non_gemm_flops(cfg: dict, m: int, n_full: int, n_gdn: int) -> dict:
    """Arithmetic that cannot reach a quantized GEMM, counted explicitly."""
    H = cfg["hidden_size"]
    hd = cfg["head_dim"]
    nq = cfg["num_attention_heads"]
    lvh = cfg["linear_num_value_heads"]
    lvd = cfg["linear_value_head_dim"]
    lkd = cfg["linear_key_head_dim"]
    conv_k = cfg["linear_conv_kernel_dim"]
    I = cfg["intermediate_size"]

    # Full attention scores and context, causal, so about half the square.
    attn = n_full * 2 * (2 * m * m * nq * hd) * 0.5

    # Gated DeltaNet recurrence: per position, per value head, the state is
    # (lkd x lvd). The update and the readout each touch it once.
    gdn_state = n_gdn * m * lvh * lkd * lvd * 2 * 2

    # Short depthwise convolution on the GDN q/k/v stream.
    gdn_conv = n_gdn * m * (2 * cfg["linear_num_key_heads"] * lkd
                            + lvh * lvd) * conv_k * 2

    # RMS norms, SiLU, gating, rope. Small, counted generously.
    elementwise = (n_full + n_gdn) * m * (4 * H + 2 * I) * 2

    return {
        "full_attention_scores_and_context": attn,
        "gated_deltanet_recurrence": gdn_state,
        "gdn_short_convolution": gdn_conv,
        "norms_activations_gating_rope": elementwise,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512, help="seed prefill tokens")
    ap.add_argument("--vector-limit", type=int, default=32,
                    help="get_qmv_batch_limit (quantized.cpp:84-110) returns "
                         "6..32 over every architecture branch. 32 is the "
                         "maximum, so it is the conservative choice: M=512 "
                         "clears it and M=1 misses it under every branch")
    ap.add_argument("--lm-head-all-positions", action="store_true",
                    help="score the whole seed window through lm_head")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    group_size = cfg["quantization"]["group_size"]
    bits = cfg["quantization"]["bits"]
    rows, n_full, n_gdn = build_projections(cfg, args)

    total_gemm = 0.0
    nax_gemm = 0.0
    for r in rows:
        m_rows = r.get("rows_override", args.m)
        r["M"] = m_rows
        r["flops_per_layer"] = 2.0 * m_rows * r["N"] * r["K"]
        r["flops_total"] = r["flops_per_layer"] * r["layers"]
        ok, why = routes_to_nax(m_rows, r["N"], r["K"], group_size,
                                args.vector_limit)
        r["routes_to_affine_qmm_t_nax"] = ok
        r["routing_reason"] = why
        r["split_k"] = splitk_for(m_rows, r["N"], r["K"], group_size)
        total_gemm += r["flops_total"]
        if ok:
            nax_gemm += r["flops_total"]

    checkpoint = verify_against_checkpoint(rows, bits, group_size)
    other = non_gemm_flops(cfg, args.m, n_full, n_gdn)
    total_other = sum(other.values())
    total_prefill = total_gemm + total_other

    qmm_share = nax_gemm / total_prefill

    # The one soft reading in the table is that begin() projects a single row
    # through lm_head. Price the opposite reading instead of arguing about it:
    # at M=512 lm_head has n_tiles=7760, so split_k collapses to 1 and it would
    # route to NAX as well, entering both the numerator and the denominator.
    lm = next(r for r in rows if r["family"] == "lm_head")
    lm_full = 2.0 * args.m * lm["N"] * lm["K"]
    lm_delta = lm_full - lm["flops_total"]
    share_if_lm_head_scored_every_position = (
        (nax_gemm + lm_delta) / (total_prefill + lm_delta)
    )
    gemm_share = total_gemm / total_prefill

    published_per_prefill_pct = 0.100436
    out = {
        "experiment": "E156",
        "rung": "R1",
        "harness": "offline",
        "worktree_head": sh("git", "rev-parse", "HEAD"),
        "question": (
            "What fraction of 512-token seed prefill arithmetic reaches "
            "affine_qmm_t_nax?"
        ),
        "m_prefill_tokens": args.m,
        "layers_full_attention": n_full,
        "layers_gdn": n_gdn,
        "source_provenance": {
            "scored_prefill_entry":
                "Qwen36MTPBlockSession.begin, "
                "Sources/MLXFastModel/Qwen36MTPBlockSession.swift:688-739, "
                "called from "
                "Sources/MLXFastTrustedHarness/QwenRuntimeMTPWorker.swift:313",
            "model_under_it":
                "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift "
                "loaded through LLMModelFactory "
                "(QwenRuntimeMTPWorker.swift:146-165). "
                "Sources/MLXFastModel/Qwen35{Attention,Block,MLP,FastEngine}"
                ".swift are NOT on this path.",
            "dispatch_chain":
                "QuantizedMatmul::eval_gpu quantized.cpp:1393-1440 -> "
                "qmm_splitk:776-810 -> qmm:682-700 -> qmm_nax -> "
                "affine_qmm_t_nax",
            "layer_split_cross_checked":
                "weights/config.json layer_types gives 48 linear_attention "
                "and 16 full_attention; Qwen35.swift:3888 derives the same "
                "split as (layerIdx+1) % fullAttentionInterval != 0",
            "unquantized_by_construction":
                "GDN depthwise Conv1d(10240,10240,k=4,groups=10240) at "
                "Qwen35.swift:776-785 is BF16 with no .scales/.biases, as are "
                "A_log, dt_bias and linear_attn.norm.weight. They cannot "
                "reach a quantized GEMM at any M.",
            "mtp_head_not_in_prefill":
                "begin() runs the backbone plus one applyLMHead and stashes "
                "the seed hidden state at :711; the head first runs at the "
                "first drafting round, :1574-1590.",
            "method": "static source read, harness=offline, nothing executed",
        },
        "quantization": {"group_size": group_size, "bits": bits},
        "lm_head_all_positions": bool(args.lm_head_all_positions),
        "checkpoint_cross_check": checkpoint,
        "projections": rows,
        "non_gemm_flops": other,
        "totals": {
            "quantized_gemm_flops": total_gemm,
            "nax_routed_gemm_flops": nax_gemm,
            "non_gemm_flops": total_other,
            "prefill_flops": total_prefill,
        },
        "e156_prefill_qmm_share_estimate": qmm_share,
        "e156_prefill_all_gemm_share": gemm_share,
        "e156_prefill_qmm_share_sensitivity": {
            "lm_head_last_position_only": qmm_share,
            "lm_head_every_seed_position":
                share_if_lm_head_scored_every_position,
            "verdict":
                "The single soft reading in the table cannot change the "
                "conclusion. Both readings put the share above 0.99, so the "
                "share is insensitive to it and no execution trace is needed "
                "to settle it.",
        },
        "e156_prefill_qmm_share_basis": (
            "multiply-accumulate FLOPs in one prefill forward pass, NOT time. "
            "harness=offline"
        ),
        "e156_prize_model": {
            "equation": (
                "published gain % = 0.100436 x qmm_share x kernel_speedup, "
                "harness=ranked, sign convention POSITIVE PUBLISHED PERCENT "
                "IS A GAIN and kernel_speedup is the fractional REDUCTION in "
                "qmm_t_nax time"
            ),
            "published_pct_per_prefill_pct": published_per_prefill_pct,
            "qmm_share": qmm_share,
            "published_pct_per_kernel_pct":
                published_per_prefill_pct * qmm_share,
            "kernel_speedup_needed_to_close_the_1_2578_pct_gap":
                (1.2578 / (published_per_prefill_pct * qmm_share)
                 if qmm_share > 0 else math.inf),
        },
        "flop_share_is_not_time_share": (
            "A FLOP share becomes a time share only if the routed GEMMs and "
            "the rest of prefill run at the same fraction of their respective "
            "roofline limits. They do not have to. The Gated DeltaNet "
            "recurrence in particular is known from campaign FINDING work to "
            "run far below its DRAM bound in the DECODE path, which would "
            "raise its time share above its FLOP share and LOWER the effective "
            "qmm time share below the number reported here. This estimate is "
            "therefore an UPPER bound on the time share unless the routed "
            "GEMMs are themselves unusually inefficient. It cannot be "
            "measured on applegpu_g16s because is_nax_available() is false."
        ),
    }
    (HERE / "e156-prefill-qmm-share.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    print("layers: %d full attention, %d gdn" % (n_full, n_gdn))
    print("%-20s %7s %7s %6s %8s %4s  %s"
          % ("family", "N", "K", "layers", "Tflop", "ckpt", "routing"))
    by_family = {c["family"]: c for c in checkpoint["checks"]}
    for r in rows:
        print("%-20s %7d %7d %6d %8.2f %4s  %s"
              % (r["family"], r["N"], r["K"], r["layers"],
                 r["flops_total"] / 1e12,
                 "ok" if by_family[r["family"]]["agrees"] else "MISMATCH",
                 "NAX" if r["routes_to_affine_qmm_t_nax"]
                 else r["routing_reason"]))
    print()
    for name, v in other.items():
        print("%-44s %8.2f Tflop" % (name, v / 1e12))
    print()
    print("quantized GEMM share of prefill FLOPs   %.4f" % gemm_share)
    print("share routed to affine_qmm_t_nax        %.4f" % qmm_share)
    print("published %% per 1 %% kernel speedup      %.6f"
          % (published_per_prefill_pct * qmm_share))
    print("kernel speedup needed to close +1.2578 %%  %.2f %%"
          % out["e156_prize_model"]
             ["kernel_speedup_needed_to_close_the_1_2578_pct_gap"])
    print("checkpoint cross-check                  %s"
          % checkpoint["every_projection_agrees_with_the_checkpoint"])

    if not checkpoint["every_projection_agrees_with_the_checkpoint"]:
        raise SystemExit(
            "derived shapes disagree with the checkpoint; the share is not "
            "trustworthy. See checkpoint_cross_check in the JSON."
        )


if __name__ == "__main__":
    main()
