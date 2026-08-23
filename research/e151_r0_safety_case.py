#!/usr/bin/env python3
"""E151 rung R0: the offline safety case for the 128x32 NAX seed-prefill retile.

harness=offline. Nothing here is a timing measurement. Every number is either
read from source in this checkout or derived from a stated model. The ranked
NAX prefill GEMM cannot execute on any host this campaign owns (Finding 250:
`is_nax_available()` is false on every `applegpu_g16s` Mac we have), so the
purpose of this rung is to decide whether the arm is SAFE to put in a
submission slot, not whether it is fast.

Sections map one-to-one onto the assignment rungs R0.1 - R0.7.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
READABLE = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h"
TWIN = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp"
NAX_H = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/gemm/nax.h"
HOST = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp"
SESSION = ROOT / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
JIT = ROOT / "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/jit_kernels.cpp"

# ---------------------------------------------------------------------------
# Host launch geometry. Read from `qmm_nax` in quantized.cpp, asserted below.
# ---------------------------------------------------------------------------
HOST_BM = 64
HOST_BN = 64
HOST_BK = 64
WM = 2
WN = 2
SIMD_SIZE = 32
TGP_SIZE = WM * WN * SIMD_SIZE  # 128 threads

# Arm geometry.
ARM_BM = 128
ARM_BN = 32

# Model constants. Qwen 3.8 27B, affine 4-bit group-64, bf16 activations.
GROUP_SIZE = 64
BITS = 4
SIZEOF_T = 2  # bfloat16
BYTES_PER_WEIGHT_ELEM = BITS / 8 + (2 * SIZEOF_T) / GROUP_SIZE  # + scale, bias

# The seven linear shapes the assignment names. `evaluated_at_M` records
# whether the SCORED seed prefill actually evaluates that GEMM at M = seed.
SCORED_SHAPES = [
    # name, K, N, layers, evaluated_at_seed_M
    ("gdn.in_proj", 5120, 16480, 48, True),
    ("fa.qkv", 5120, 14336, 16, True),
    ("mlp.gate_up", 5120, 34816, 64, True),
    ("lm_head", 5120, 248320, 1, False),
    ("gdn.out_proj", 6144, 5120, 48, True),
    ("fa.o_proj", 6144, 5120, 16, True),
    ("mlp.down", 17408, 5120, 64, True),
]

# harness=ranked, from the E151 assignment section 6 and Finding 251.
RANKED_PREFILL_SECONDS = 0.52643
RANKED_ESSAYS_SERIAL_SPT_MEDIAN = 0.037984678


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Source facts. Fail closed: if the tree moves under us, this rung stops.
# ---------------------------------------------------------------------------
def find_line(path: pathlib.Path, needle: str) -> int | None:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if needle in line:
            return i
    return None


def source_facts() -> dict:
    host_src = HOST.read_text()
    qmm_nax = host_src[host_src.index("void qmm_nax(") :][:2000]
    readable = READABLE.read_text()
    twin = TWIN.read_text()

    facts = {
        "host_bm_64": bool(re.search(r"\bint bm = 64;", qmm_nax)),
        "host_bn_64": bool(re.search(r"\bint bn = 64;", qmm_nax)),
        "host_bk_64": bool(re.search(r"\bint bk = 64;", qmm_nax)),
        "host_wm_2": bool(re.search(r"\bint wm = 2;", qmm_nax)),
        "host_wn_2": bool(re.search(r"\bint wn = 2;", qmm_nax)),
        "host_grid_dims_verbatim": (
            "MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);" in qmm_nax
        ),
        "host_group_dims_verbatim": "MTL::Size group_dims(32, wn, wm);" in qmm_nax,
        "host_aligned_predicate": "bool aligned = N % 64 == 0;" in qmm_nax,
        "sk_is_32": "constexpr short SK = 32;" in readable,
        "bk_padded_formula": "constexpr int BK_padded = (BK + 16 / sizeof(T));" in readable,
        "tk_from_sk": "constexpr short TK = SK / 16;" in readable,
        "ws_alloc_uses_ktgbn": "threadgroup T Ws[kTgBN * BK_padded];" in readable,
        "ktgbn_formula": (
            "constexpr int kTgBN = kE147NaxRetileBN > BN ? kE147NaxRetileBN : BN;"
            in readable
        ),
        "arm_flag_present": "constexpr bool kE147NaxRetileOn" in readable,
        "arm_bm_128": "kE147NaxRetileArmed ? 128 : BM" in readable,
        "arm_bn_32": "kE147NaxRetileArmed ? kE147ArmBN : BN" in readable,
        "arm_bn_constant_is_32": "kE147ArmBN = 32;" in readable,
        "arm_gated_on_loader_legality": (
            "kE147NaxRetileOn && kE147NaxRetileLoaderLegal" in readable
        ),
        "scored_cell_arm_static_assert": (
            "the scored affine group-64 4-bit tile must take the retile arm"
            in readable
        ),
        "retile_area_static_assert": (
            'BM * BN == kHostBM * kHostBN, "a retiled tile must cover the host tile"'
            in readable
        ),
        "retile_split_static_assert": (
            'static_assert(kHostBN % BN == 0, "the host tile must split along N");'
            in readable
        ),
        "rule145_guard_sites_readable": readable.count(
            "tile shape matches no tile_matmad_nax branch"
        ),
        "rule145_guard_sites_twin": twin.count(
            "tile shape matches no tile_matmad_nax branch"
        ),
        "nax_branch_one_verbatim": "if constexpr (TN == 1 && TM % 2 == 0) {" in NAX_H.read_text(),
        "nax_branch_two_verbatim": "} else if constexpr (TN % 2 == 0) {" in NAX_H.read_text(),
        "nax_has_no_else": NAX_H.read_text().count("} else {\n    STEEL_PRAGMA_UNROLL") == 0,
        "mma_descriptor_k_extent_16": NAX_H.read_text().count(
            "matmul2d_descriptor(\n        16,\n        32,\n        16,"
        ),
        "seed_lm_head_dead_comment": (
            "It is deliberately\n        // NEVER evaluated" in SESSION.read_text()
        ),
    }
    facts["lines"] = {
        "host_qmm_nax": find_line(HOST, "void qmm_nax("),
        "host_bm": find_line(HOST, "  int bm = 64;"),
        "host_grid_dims": find_line(HOST, "MTL::Size grid_dims((N + bn - 1) / bn"),
        "readable_sk": find_line(READABLE, "constexpr short SK = 32;"),
        "readable_bk_padded": find_line(READABLE, "constexpr int BK_padded"),
        "readable_arm_flag": find_line(READABLE, "constexpr bool kE147NaxRetileOn"),
        "readable_ws_alloc": find_line(READABLE, "threadgroup T Ws[kTgBN * BK_padded];"),
        "readable_gridstride": find_line(READABLE, "for (int t = int(tid.y) * tiles_x_host"),
        "twin_arm_flag": find_line(TWIN, "constexpr bool kE147NaxRetileOn"),
        "nax_branch_one": find_line(NAX_H, "if constexpr (TN == 1 && TM % 2 == 0) {"),
        "nax_branch_two": find_line(NAX_H, "} else if constexpr (TN % 2 == 0) {"),
        "session_begin": find_line(SESSION, "public func begin(seedTokens: [Int])"),
    }
    return facts


# ---------------------------------------------------------------------------
# R0.1  scored dispatch shape table
# ---------------------------------------------------------------------------
def r0_1_shape_table(M_values=(512, 511)) -> list[dict]:
    rows = []
    for M in M_values:
        for name, K, N, layers, evaluated in SCORED_SHAPES:
            tiles_x_64 = ceil_div(N, 64)
            tiles_x_32 = ceil_div(N, 32)
            tiles_y_64 = ceil_div(M, 64)
            tiles_y_128 = ceil_div(M, 128)
            launched = tiles_y_64 * tiles_x_64
            required = tiles_y_128 * tiles_x_32
            rows.append(
                {
                    "M": M,
                    "projection": name,
                    "K": K,
                    "N": N,
                    "layers": layers,
                    "evaluated_in_scored_seed_prefill": evaluated,
                    "N_mod_64": N % 64,
                    "N_mod_32": N % 32,
                    "aligned_N_template_flag": (N % 64 == 0),
                    "tiles_x_at_BN64": tiles_x_64,
                    "tiles_x_at_BN32": tiles_x_32,
                    "tiles_y_at_BM64": tiles_y_64,
                    "tiles_y_at_BM128": tiles_y_128,
                    "launched_threadgroups": launched,
                    "required_tiles_under_retile": required,
                    "idle_threadgroups": launched - required,
                    "tiles_per_threadgroup_max": ceil_div(required, launched)
                    if launched
                    else 0,
                    "m_passes_over_weights_at_BM64": tiles_y_64,
                    "m_passes_over_weights_at_BM128": tiles_y_128,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# R0.2  exhaustive index-coverage proof
# ---------------------------------------------------------------------------
def coverage(M: int, N: int, variant: str = "shipped") -> dict:
    """Host-language model of the grid-stride loop in qmm_t_nax_tgp_impl.

    `shipped` reproduces the arm exactly. The other variants are deliberate
    defects that a real proof must catch (Rule 101).
    """
    tiles_x_frozen = ceil_div(N, HOST_BN)
    tiles_x_new = ceil_div(N, ARM_BN)
    tiles_y_new = ceil_div(M, ARM_BM)
    required = tiles_y_new * tiles_x_new

    if variant == "control_launched_from_BM":
        launched = ceil_div(M, ARM_BM) * tiles_x_frozen
    else:
        launched = ceil_div(M, HOST_BM) * tiles_x_frozen

    visits: dict[tuple[int, int], int] = {}
    out_of_range = 0
    for tid_y in range(ceil_div(M, HOST_BM)):
        for tid_x in range(tiles_x_frozen):
            t = tid_y * tiles_x_frozen + tid_x
            while t < required:
                if variant == "control_frozen_tiles_x":
                    bx, by = t % tiles_x_frozen, t // tiles_x_frozen
                elif variant == "control_by_from_frozen":
                    bx, by = t % tiles_x_new, t // tiles_x_frozen
                else:
                    bx, by = t % tiles_x_new, t // tiles_x_new
                if 0 <= by < tiles_y_new and 0 <= bx < tiles_x_new:
                    visits[(by, bx)] = visits.get((by, bx), 0) + 1
                else:
                    out_of_range += 1
                t += launched

    counts = [visits.get((by, bx), 0) for by in range(tiles_y_new) for bx in range(tiles_x_new)]
    return {
        "M": M,
        "N": N,
        "variant": variant,
        "tiles_x_frozen": tiles_x_frozen,
        "tiles_x_new": tiles_x_new,
        "tiles_y_new": tiles_y_new,
        "launched": launched,
        "required": required,
        "tiles_visited_zero_times": sum(1 for c in counts if c == 0),
        "tiles_visited_once": sum(1 for c in counts if c == 1),
        "tiles_visited_more_than_once": sum(1 for c in counts if c > 1),
        "max_visits_to_one_tile": max(counts) if counts else 0,
        "visits_outside_required": out_of_range,
        "exact": (
            out_of_range == 0
            and bool(counts)
            and all(c == 1 for c in counts)
        ),
    }


def r0_2_coverage(shape_rows: list[dict]) -> dict:
    pairs = sorted(
        {(r["M"], r["N"]) for r in shape_rows if r["evaluated_in_scored_seed_prefill"]}
    )
    # Keep the dead lm_head pair in the enumeration anyway: if a later change
    # ever evaluates it, the coverage claim must already hold.
    pairs += sorted({(r["M"], r["N"]) for r in shape_rows} - set(pairs))

    shipped = [coverage(M, N, "shipped") for M, N in pairs]
    controls = {}
    for variant in (
        "control_frozen_tiles_x",
        "control_by_from_frozen",
        "control_launched_from_BM",
    ):
        runs = [coverage(M, N, variant) for M, N in pairs]
        controls[variant] = {
            "pairs_evaluated": len(runs),
            "pairs_where_control_is_exact": sum(1 for r in runs if r["exact"]),
            "caught": all(not r["exact"] for r in runs),
            "example": next(r for r in runs if not r["exact"]),
        }

    return {
        "pairs_enumerated": [{"M": m, "N": n} for m, n in pairs],
        "shipped": shipped,
        "e151_coverage_exact": all(r["exact"] for r in shipped),
        "controls": controls,
        "e151_coverage_failing_controls_caught": sum(
            1 for c in controls.values() if c["caught"]
        ),
    }


# ---------------------------------------------------------------------------
# R0.3  K-order preservation
# ---------------------------------------------------------------------------
def r0_3_k_order(facts: dict) -> dict:
    """The retile changes BM and BN. It changes neither BK, SK, nor TK, so the
    K reduction sequence seen by every output element is unchanged."""
    TK = 32 // 16
    per_bk_step_mma_calls = (HOST_BK // 32) * TK  # kk1 loop x kk loop
    return {
        "e151_k_order_preserved": bool(
            facts["sk_is_32"]
            and facts["tk_from_sk"]
            and facts["host_bk_64"]
            and facts["mma_descriptor_k_extent_16"] >= 2
        ),
        "BK_host": HOST_BK,
        "SK": 32,
        "TK": TK,
        "mma_k_extent_per_call": 16,
        "mma_calls_per_output_element_per_BK_step": per_bk_step_mma_calls,
        "loop_nest": "for k in 0..K step BK { for kk1 in 0..BK step SK { for kk in 0..TK } }",
        "arm_off_branch": "TM=2 TN=2 -> nax.h branch two (dual-N mma)",
        "arm_on_branch": "TM=4 TN=1 -> nax.h branch one (dual-M mma)",
        "citation": {
            "SK": f"quantized_nax.h:{facts['lines']['readable_sk']}",
            "host_bk": f"quantized.cpp:{facts['lines']['host_bm']} block (int bk = 64)",
            "branch_one": f"steel/gemm/nax.h:{facts['lines']['nax_branch_one']}",
            "branch_two": f"steel/gemm/nax.h:{facts['lines']['nax_branch_two']}",
        },
        "argument": (
            "Both tile_matmad_nax branches call BaseNAXFrag::mma with the identical "
            "matmul2d_descriptor(16, 32, 16, ..., multiply_accumulate). The two "
            "branches differ only in which pair of accumulator fragments is fused "
            "into one hardware call: branch two pairs Cn0/Cn1 across N, branch one "
            "pairs Cm0/Cm1 across M. Each accumulator occupies its own destination "
            "cooperative-tensor lane range, so per-element accumulation is "
            "independent of the pairing. K extent per call is 16 in both branches, "
            "and the enclosing k / kk1 / kk nest is byte-identical, so every output "
            "element sums the same K partial products in the same increasing order. "
            "The retile is therefore bit-identical arithmetic."
        ),
        "residual_risk": (
            "Unverifiable offline: the internal reduction tree of the vendor "
            "mpp::tensor_ops::matmul2d instruction. It is the same instruction with "
            "the same descriptor in both branches, so this risk is shared with the "
            "shipped arm-off path rather than introduced by the retile."
        ),
    }


# ---------------------------------------------------------------------------
# R0.4  Rule 145 shape legality
# ---------------------------------------------------------------------------
def r0_4_rule145(facts: dict) -> dict:
    def tile_shape(BM: int, BN: int) -> dict:
        SM, SN = BM // WM, BN // WN
        TM, TN = SM // 16, SN // 16
        return {
            "BM": BM,
            "BN": BN,
            "SM": SM,
            "SN": SN,
            "TM": TM,
            "TN": TN,
            "branch_one_TN1_TMeven": (TN == 1 and TM % 2 == 0),
            "branch_two_TNeven": (TN % 2 == 0),
            "guard_passes": (TN == 1 and TM % 2 == 0) or (TN % 2 == 0),
        }

    off = tile_shape(HOST_BM, HOST_BN)
    on = tile_shape(ARM_BM, ARM_BN)
    return {
        "arm_off": off,
        "arm_on": on,
        "e151_rule145_arm_on_legal": on["guard_passes"],
        "e151_rule145_guard_sites_readable": facts["rule145_guard_sites_readable"],
        "e151_rule145_guard_sites_twin": facts["rule145_guard_sites_twin"],
        "guard_reached_by_arm_on": True,
        "note": (
            "The arm-on tile matches branch one only. The guard is the disjunction "
            "of both branch predicates copied verbatim, so a tile that matches "
            "neither stops the build instead of silently multiplying nothing."
        ),
    }


# ---------------------------------------------------------------------------
# R0.5  threadgroup memory budget
# ---------------------------------------------------------------------------
def r0_5_threadgroup() -> dict:
    bk_padded = HOST_BK + 16 // SIZEOF_T
    # `threadgroup T Ws[kTgBN * BK_padded]` with
    # kTgBN = (arm_BN > host_BN) ? arm_BN : host_BN.
    ktgbn_off = HOST_BN
    ktgbn_on = ARM_BN if ARM_BN > HOST_BN else HOST_BN
    bytes_off = ktgbn_off * bk_padded * SIZEOF_T
    bytes_on = ktgbn_on * bk_padded * SIZEOF_T
    bytes_arm_on_actually_written = ARM_BN * bk_padded * SIZEOF_T

    def frag_regs(TM: int, TN: int, TK: int) -> dict:
        elems_per_frag = 8  # (16*16)/32
        a = TM * TK * elems_per_frag * SIZEOF_T
        b = TN * TK * elems_per_frag * SIZEOF_T
        d = TM * TN * elems_per_frag * 4  # AccumType = float
        return {"A_bytes": a, "B_bytes": b, "D_bytes": d, "total_bytes": a + b + d}

    return {
        "BK": HOST_BK,
        "BK_padded": bk_padded,
        "sizeof_T": SIZEOF_T,
        "e151_tgp_bytes_64x64": bytes_off,
        "e151_tgp_bytes_128x32": bytes_on,
        "tgp_bytes_delta": bytes_on - bytes_off,
        "tgp_bytes_touched_by_arm_on": bytes_arm_on_actually_written,
        "threadgroup_limit_bytes": 32768,
        "e151_tgp_fits": bytes_on <= 32768,
        "why_no_growth": (
            "Only BN drives the staging allocation, and the arm narrows BN from 64 "
            "to 32. kTgBN takes the maximum of the two, so the allocation is "
            "unchanged at the host value and the arm writes half of it. The arm "
            "cannot increase threadgroup pressure."
        ),
        "per_thread_register_tiles_bytes": {
            "arm_off_TM2_TN2_TK2": frag_regs(2, 2, 2),
            "arm_on_TM4_TN1_TK2": frag_regs(4, 1, 2),
        },
        "loader_reads_per_thread_per_k_step": {
            "arm_off_BROWS64": (HOST_BN * (HOST_BK // (8 // BITS))) // TGP_SIZE,
            "arm_on_BROWS32": (ARM_BN * (HOST_BK // (8 // BITS))) // TGP_SIZE,
        },
    }


# ---------------------------------------------------------------------------
# R0.6  loader traffic model  (MODEL, not a measurement)
# ---------------------------------------------------------------------------
def r0_6_traffic(shape_rows: list[dict], M: int = 512) -> dict:
    rows = []
    tot_flop = 0.0
    tot_w_bytes_once = 0.0
    tot_w_bytes_off = 0.0
    tot_w_bytes_on = 0.0
    for r in shape_rows:
        if r["M"] != M or not r["evaluated_in_scored_seed_prefill"]:
            continue
        K, N, layers = r["K"], r["N"], r["layers"]
        elems = K * N * layers
        flop = 2.0 * M * elems
        w_once = elems * BYTES_PER_WEIGHT_ELEM
        off = w_once * r["m_passes_over_weights_at_BM64"]
        on = w_once * r["m_passes_over_weights_at_BM128"]
        tot_flop += flop
        tot_w_bytes_once += w_once
        tot_w_bytes_off += off
        tot_w_bytes_on += on
        rows.append(
            {
                "projection": r["projection"],
                "K": K,
                "N": N,
                "layers": layers,
                "flop": flop,
                "weight_bytes_single_pass": w_once,
                "weight_bytes_arm_off": off,
                "weight_bytes_arm_on": on,
                "m_passes_off": r["m_passes_over_weights_at_BM64"],
                "m_passes_on": r["m_passes_over_weights_at_BM128"],
            }
        )
    for row in rows:
        row["share_of_prefill_flop"] = row["flop"] / tot_flop
        row["share_of_weight_traffic_saved"] = (
            row["weight_bytes_arm_off"] - row["weight_bytes_arm_on"]
        ) / (tot_w_bytes_off - tot_w_bytes_on)

    return {
        "harness": "model",
        "M": M,
        "per_gemm": rows,
        "total_prefill_flop": tot_flop,
        "weight_bytes_single_pass": tot_w_bytes_once,
        "weight_bytes_arm_off": tot_w_bytes_off,
        "weight_bytes_arm_on": tot_w_bytes_on,
        "e151_predicted_loader_traffic_reduction": (
            tot_w_bytes_on / tot_w_bytes_off - 1.0
        ),
        "dequantize_work_ratio_on_over_off": 0.5,
        "activation_reads_ratio_on_over_off": 2.0,
        "activation_unique_bytes": M * 5120 * SIZEOF_T,
        "bitwonka_claim": "each weight region loaded 4 times instead of 8",
        "claim_reproduced": all(
            r["m_passes_off"] == 8 and r["m_passes_on"] == 4 for r in rows
        ),
        "caveats": (
            "MODEL, not a measurement. It assumes (a) weight tiles miss cache "
            "between M passes, which holds because the smallest scored weight "
            "matrix is 5120x5120 at 4 bits = 14.4 MiB and the dispatch orders "
            "tid.x fastest, so an M pass completes before the next begins; and "
            "(b) activation re-reads stay resident, which holds because the whole "
            "seed activation block is 512x5120 bf16 = 5.24 MB and is shared by "
            "every concurrent threadgroup. The arm doubles activation re-reads and "
            "halves both weight traffic and weight dequantization ALU work."
        ),
    }


# ---------------------------------------------------------------------------
# R0.7  roofline sanity bound
# ---------------------------------------------------------------------------
def r0_7_roofline(traffic: dict) -> dict:
    flop = traffic["total_prefill_flop"]
    achieved = flop / RANKED_PREFILL_SECONDS
    # Lower bound on sustained read bandwidth demonstrated by the ranked serial
    # decode leg: batch-1 decode must stream every resident weight byte per
    # token. harness=ranked input, model output.
    decode_bw_lower_bound = (
        traffic["weight_bytes_single_pass"] / RANKED_ESSAYS_SERIAL_SPT_MEDIAN
    )
    bw_off = traffic["weight_bytes_arm_off"] / RANKED_PREFILL_SECONDS
    bw_on = traffic["weight_bytes_arm_on"] / RANKED_PREFILL_SECONDS
    return {
        "harness": "model",
        "e151_prefill_tflops_achieved": achieved / 1e12,
        "prefill_seconds": RANKED_PREFILL_SECONDS,
        "prefill_flop": flop,
        "implied_weight_read_bandwidth_arm_off_GBs": bw_off / 1e9,
        "implied_weight_read_bandwidth_arm_on_GBs": bw_on / 1e9,
        "decode_demonstrated_read_bandwidth_lower_bound_GBs": decode_bw_lower_bound / 1e9,
        "prefill_bandwidth_as_fraction_of_decode_demonstrated": bw_off / decode_bw_lower_bound,
        "arithmetic_intensity_arm_off_flop_per_byte": flop / traffic["weight_bytes_arm_off"],
        "arithmetic_intensity_arm_on_flop_per_byte": flop / traffic["weight_bytes_arm_on"],
        "reasoning": (
            "The serial decode leg is batch-1 and must read every resident weight "
            "byte per token, so 0.037985 s/token over a single weight pass "
            "demonstrates a sustained read rate that bounds the device from below. "
            "Prefill arm-off demands a large fraction of that same rate while also "
            "sustaining tens of TFLOP/s, so the kernel is co-limited rather than "
            "purely compute bound. The arm halves the traffic term and leaves the "
            "compute term untouched, so it can only help through traffic and "
            "dequantization ALU relief. A 5 percent effect is the right order for "
            "partial relief of a co-limited kernel; a 50 percent effect would imply "
            "a fully bandwidth-bound kernel and is NOT predicted. The published "
            "5cdc9c17 receipt at -4.97 percent is consistent with this model."
        ),
        "falsifiable_prediction": (
            "If the mechanism is traffic, the ranked receipt moves the prefill "
            "channel and leaves the decode channel unchanged, because decode runs "
            "qmv and never enters qmm_t_nax."
        ),
    }


# ---------------------------------------------------------------------------
# R0.8  Rule 153 JIT library partition
#
# get_quantized_kernel (decode QMV) and get_qmm_nax_kernel (prefill NAX GEMM)
# each build a private JIT translation unit by concatenating a fixed list of
# metal::<name>() source blobs. A source file changes the decode QMV library if
# and only if it backs a name in the decode list. Enumerating the two lists
# from the enforcing C++ answers the question statically.
# ---------------------------------------------------------------------------
GENERATED_DIR = "Vendor/mlx-swift/Source/Cmlx/mlx-generated"
KERNELS_DIR = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"


def _concatenated_names(func_name: str) -> list[str]:
    text = JIT.read_text()
    start = text.index(f"MTL::ComputePipelineState* {func_name}(")
    end = text.index("\nMTL::ComputePipelineState* ", start + 1)
    body = text[start:end]
    call = body[body.index("concatenate(") : body.index("return kernel_source;")]
    return re.findall(r"metal::(\w+)\(\)", call)


def _backing_paths(names: list[str]) -> list[str]:
    paths: list[str] = []
    for name in names:
        for cand in (f"{GENERATED_DIR}/{name}.cpp", f"{KERNELS_DIR}/{name}.h"):
            if (ROOT / cand).exists():
                paths.append(cand)
    return paths


def r0_8_rule153(base: str) -> dict:
    decode_names = _concatenated_names("get_quantized_kernel")
    prefill_names = _concatenated_names("get_qmm_nax_kernel")
    decode_paths = _backing_paths(decode_names)
    prefill_paths = _backing_paths(prefill_names)

    tracked = subprocess.run(
        ["git", "diff", "--name-only", base, "--"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    changed = sorted(set(tracked) | set(untracked))

    hits_decode = sorted(set(changed) & set(decode_paths))
    hits_prefill = sorted(set(changed) & set(prefill_paths))
    return {
        "harness": "offline",
        "base": base,
        "enforcing_source": "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/jit_kernels.cpp",
        "decode_qmv_entry_point": "get_quantized_kernel",
        "decode_qmv_concatenated_names": decode_names,
        "decode_qmv_backing_paths": decode_paths,
        "prefill_nax_entry_point": "get_qmm_nax_kernel",
        "prefill_nax_concatenated_names": prefill_names,
        "prefill_nax_backing_paths": prefill_paths,
        "shared_backing_paths": sorted(set(decode_paths) & set(prefill_paths)),
        "changed_paths": changed,
        "changed_paths_in_decode_qmv_library": hits_decode,
        "changed_paths_in_prefill_nax_library": hits_prefill,
        "e151_r1_touches_decode_qmv_library": bool(hits_decode),
        "e151_r1_touches_prefill_nax_library": bool(hits_prefill),
    }


# ---------------------------------------------------------------------------
# R0.9  weight-loader legality across every ahead-of-time instantiation
#
# FINDING. The arm was shipped OFF by E147, so the metallib never instantiated
# it and a latent incompatibility stayed invisible. `quantized_nax.metal`
# instantiates `affine_qmm_t_nax` at (BM, BK, BN, WM, WN) = (64, 64, 64, 2, 2)
# for group sizes 128, 64 and 32 and for bits 2, 3, 4, 5, 6 and 8. The loader's
# `group_size == 32` specialization asserts `(BCOLS_PACKED / n_reads) ==
# n_groups`, and `n_reads` scales with BROWS, which is the tile's BN. The arm
# halves BN from 64 to 32, which halves `n_reads` and doubles the left side, so
# every group-32 instantiation fails to compile. The scored path is affine
# group-64, which uses the primary loader template and does not constrain
# BROWS at all.
#
# This census reproduces the loader arithmetic offline for the whole
# instantiation matrix and reports which cells the arm may take.
# ---------------------------------------------------------------------------
AOT_GROUP_SIZES = (128, 64, 32)
AOT_BITS = (2, 3, 4, 5, 6, 8)


def pack_factor(bits: int, wsize: int = 8) -> int:
    if bits in (3, 5):
        return 8
    if bits == 6:
        return 4
    return wsize // bits


def loader_legal(group_size: int, bits: int, brows: int) -> bool:
    """Reproduce QuantizedBlockLoader's own admission rule for BK = HOST_BK.

    The primary template requires BCOLS <= group_size and group_size % BCOLS
    == 0 and never mentions BROWS. Only the group_size == 32 specialization
    carries the n_reads split assert, which is the one the arm can break.
    """
    cols_packed = HOST_BK // pack_factor(bits)
    tgp = WM * WN * 32
    if group_size != 32:
        return HOST_BK <= group_size and group_size % HOST_BK == 0
    reads = 1 if cols_packed * brows < tgp else (cols_packed * brows) // tgp
    return (cols_packed // reads) == (HOST_BK // 32)


def r0_9_loader_legality() -> dict:
    rows = []
    for group_size in AOT_GROUP_SIZES:
        for bits in AOT_BITS:
            off = loader_legal(group_size, bits, HOST_BN)
            on = loader_legal(group_size, bits, ARM_BN)
            rows.append(
                {
                    "group_size": group_size,
                    "bits": bits,
                    "cols_packed": HOST_BK // pack_factor(bits),
                    "arm_off_brows": HOST_BN,
                    "arm_on_brows": ARM_BN,
                    "arm_off_loader_legal": off,
                    "arm_on_loader_legal": on,
                    "arm_may_engage": on,
                    "is_scored_cell": group_size == 64 and bits == 4,
                }
            )
    scored = [r for r in rows if r["is_scored_cell"]]
    return {
        "harness": "offline",
        "instantiation_source": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
            "quantized_nax.metal instantiate_quantized_all()"
        ),
        "instantiated_tile": {
            "BM": HOST_BM,
            "BK": HOST_BK,
            "BN": HOST_BN,
            "WM": WM,
            "WN": WN,
        },
        "rows": rows,
        "e151_aot_cells_total": len(rows),
        "e151_aot_cells_arm_engages": sum(1 for r in rows if r["arm_may_engage"]),
        "e151_aot_cells_arm_disarmed": sum(
            1 for r in rows if not r["arm_may_engage"]
        ),
        # Every arm-off cell must stay legal, otherwise the failure predates
        # the arm and this census is measuring the wrong thing.
        "e151_arm_off_legal_everywhere": all(
            r["arm_off_loader_legal"] for r in rows
        ),
        "e151_scored_cell_arm_engages": all(r["arm_may_engage"] for r in scored),
        "e151_disarmed_group_sizes": sorted(
            {r["group_size"] for r in rows if not r["arm_may_engage"]}
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e151-r0-safety-case.json")
    ap.add_argument("--base", default="de8ce44c7bc133c3c6c079957240782664afd287")
    args = ap.parse_args()

    facts = source_facts()
    shape_rows = r0_1_shape_table()
    cov = r0_2_coverage(shape_rows)
    kord = r0_3_k_order(facts)
    r145 = r0_4_rule145(facts)
    tgp = r0_5_threadgroup()
    traffic = r0_6_traffic(shape_rows)
    roof = r0_7_roofline(traffic)
    rule153 = r0_8_rule153(args.base)
    loaders = r0_9_loader_legality()

    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()

    missing = [k for k, v in facts.items() if v is False]
    verdict = {
        "e151_source_facts_all_true": not missing,
        "e151_source_facts_missing": missing,
        "e151_coverage_exact": cov["e151_coverage_exact"],
        "e151_coverage_failing_controls_caught": cov[
            "e151_coverage_failing_controls_caught"
        ],
        "e151_k_order_preserved": kord["e151_k_order_preserved"],
        "e151_rule145_arm_on_legal": r145["e151_rule145_arm_on_legal"],
        "e151_tgp_fits": tgp["e151_tgp_fits"],
        "e151_r1_touches_decode_qmv_library": rule153[
            "e151_r1_touches_decode_qmv_library"
        ],
        "e151_arm_off_legal_everywhere": loaders["e151_arm_off_legal_everywhere"],
        "e151_scored_cell_arm_engages": loaders["e151_scored_cell_arm_engages"],
        "e151_aot_cells_arm_disarmed": loaders["e151_aot_cells_arm_disarmed"],
    }
    verdict["e151_r0_pass"] = all(
        [
            verdict["e151_source_facts_all_true"],
            verdict["e151_coverage_exact"],
            verdict["e151_coverage_failing_controls_caught"] == 3,
            verdict["e151_k_order_preserved"],
            verdict["e151_rule145_arm_on_legal"],
            verdict["e151_tgp_fits"],
            not verdict["e151_r1_touches_decode_qmv_library"],
            verdict["e151_arm_off_legal_everywhere"],
            verdict["e151_scored_cell_arm_engages"],
        ]
    )

    doc = {
        "experiment": "E151",
        "rung": "R0",
        "harness": "offline",
        "commit": base_sha,
        "host_launch_geometry": {
            "BM": HOST_BM,
            "BN": HOST_BN,
            "BK": HOST_BK,
            "WM": WM,
            "WN": WN,
            "tgp_size": TGP_SIZE,
        },
        "arm_geometry": {"BM": ARM_BM, "BN": ARM_BN},
        "source_facts": facts,
        "e151_retile_shape_table": shape_rows,
        "r0_2_coverage": cov,
        "r0_3_k_order": kord,
        "r0_4_rule145": r145,
        "r0_5_threadgroup": tgp,
        "r0_6_traffic_model": traffic,
        "r0_7_roofline": roof,
        "r0_8_rule153_jit_library_partition": rule153,
        "r0_9_loader_legality": loaders,
        "verdict": verdict,
    }

    out = ROOT / args.out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    print(f"wrote {out}")
    return 0 if verdict["e151_r0_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
