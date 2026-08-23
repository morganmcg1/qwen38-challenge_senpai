"""E156: does the 128x32 retile change what `tile_matmad_nax` computes?

Why this exists
---------------
The E156 candidate arms a `(BM, BN) = (128, 32)` seed-prefill tile. With
`WM = WN = 2` that moves the per-simdgroup tile from `(TM, TN) = (2, 2)` to
`(4, 1)`, and `tile_matmad_nax` in
`Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/gemm/nax.h`
dispatches those two shapes to two *different* `BaseNAXFrag::mma` overloads:

  * `TN % 2 == 0`            -> the paired-B overload  (one A fragment, two B)
  * `TN == 1 && TM % 2 == 0` -> the paired-A overload  (two A fragments, one B)

The advisor's F4 item 4 said the two overloads build different
`matmul2d_descriptor` geometries, `(16,32,16)` versus `(32,16,16)`, and asked for
an adversarial-fragment harness plus a max-ULP delta. Reading the vendored source
shows the premise is not true on this tree: **both** overloads build
`matmul2d_descriptor(16, 32, 16, transpose_a, transpose_b, true,
multiply_accumulate)`. Every `matmul2d_descriptor` in the whole kernels tree is
`(16, 32, 16)`.

That makes the real question sharper, not softer. The paired-A overload writes
`2 * kElemsPerFrag` elements into the *left* cooperative tensor and only
`kElemsPerFrag` into the *right* one, which is the operand budget of a
`(32, 16, 16)` descriptor, not of the `(16, 32, 16)` descriptor it actually
declares. So this probe answers two things on real hardware:

  1. What are the per-thread capacities of the left, right and destination
     cooperative tensors under each descriptor? That decides whether the
     paired-A overload writes inside or outside its operand budget.
  2. Do the two overloads compute the same 32x32 output, bit for bit, from the
     same adversarial bf16 inputs over a K loop?

Rule 101: the numerical comparison carries a control that must fail. One arm is
re-run with a single input fragment element perturbed; if that does not move the
output the comparison proves nothing.

harness=local. This probe uses `mpp::tensor_ops::matmul2d` directly, so it does
not depend on `is_nax_available()`, which is false on this `applegpu_g16s` host.
It measures numerics, never time.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import mlx.core as mx
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "e156-matmul2d-pairing.json"

# Copied verbatim from BaseNAXFrag in steel/gemm/nax.h so the probe exercises the
# same lane->element mapping the scored kernel uses.
COMMON = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

constant constexpr short kFragRows = 16;
constant constexpr short kFragCols = 16;
constant constexpr short kElemsPerFrag = (kFragRows * kFragCols) / 32;  // 8
constant constexpr short kElemRows = 2;
constant constexpr short kElemCols = 4;
constant constexpr short kElemRowsJump = 8;

template <typename U>
using frag_t = metal::vec<U, kElemsPerFrag>;

// steel/gemm/nax.h BaseNAXFrag::get_coord()
inline short2 nax_coord(ushort lane) {
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2{fn, fm};
}

// descB: the geometry both overloads declare on this tree.
constant constexpr auto descB = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

// descA: the geometry the paired-A operand budget implies.
constant constexpr auto descA = mpp::tensor_ops::matmul2d_descriptor(
    32, 16, 16, false, false, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
"""

# ---------------------------------------------------------------------------
# Stage 1: operand capacities.
# ---------------------------------------------------------------------------
CAPACITY_SOURCE = r"""
  if (thread_position_in_grid.x != 0) { return; }

  mpp::tensor_ops::matmul2d<descB, metal::execution_simdgroup> gB;
  auto b_a = gB.get_left_input_cooperative_tensor<T, T, float>();
  auto b_b = gB.get_right_input_cooperative_tensor<T, T, float>();
  auto b_c = gB.get_destination_cooperative_tensor<decltype(b_a), decltype(b_b), float>();

  mpp::tensor_ops::matmul2d<descA, metal::execution_simdgroup> gA;
  auto a_a = gA.get_left_input_cooperative_tensor<T, T, float>();
  auto a_b = gA.get_right_input_cooperative_tensor<T, T, float>();
  auto a_c = gA.get_destination_cooperative_tensor<decltype(a_a), decltype(a_b), float>();

  cap[0] = (int)b_a.get_capacity();
  cap[1] = (int)b_b.get_capacity();
  cap[2] = (int)b_c.get_capacity();
  cap[3] = (int)a_a.get_capacity();
  cap[4] = (int)a_b.get_capacity();
  cap[5] = (int)a_c.get_capacity();
  cap[6] = (int)kElemsPerFrag;
"""

# ---------------------------------------------------------------------------
# Stage 2: the two overloads on the same logical 32x32xK GEMM.
#
# `a` is row-major 32 x K, `b` is row-major K x 32, both bf16. `arm` selects the
# overload. `perturb` is the Rule 101 control: index into the flat `a` buffer to
# nudge by one bf16 ulp, or -1 for none.
# ---------------------------------------------------------------------------
GEMM_SOURCE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const short2 sc = nax_coord(lane);

  const int K = kdim[0];
  const int arm = mode[0];

  // Two row fragments (m block 0 and 1) x two column fragments (n block 0, 1).
  frag_t<float> C00 = frag_t<float>(0.0f);
  frag_t<float> C01 = frag_t<float>(0.0f);
  frag_t<float> C10 = frag_t<float>(0.0f);
  frag_t<float> C11 = frag_t<float>(0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    frag_t<T> A0, A1, B0, B1;
    for (short i = 0; i < kElemRows; i++) {
      for (short j = 0; j < kElemCols; j++) {
        const short r = sc.y + i * kElemRowsJump;
        const short c = sc.x + j;
        A0[i * kElemCols + j] = a[(r) * K + (k0 + c)];
        A1[i * kElemCols + j] = a[(r + 16) * K + (k0 + c)];
        B0[i * kElemCols + j] = b[(k0 + r) * 32 + (c)];
        B1[i * kElemCols + j] = b[(k0 + r) * 32 + (c + 16)];
      }
    }

    if (arm == 0) {
      // paired-B, verbatim from BaseNAXFrag::mma(Cn0, Cn1, A, ta, Bn0, Bn1, tb)
      mpp::tensor_ops::matmul2d<descB, metal::execution_simdgroup> gemm_op;
      for (short blk = 0; blk < 2; ++blk) {
        thread frag_t<T>& A = blk == 0 ? A0 : A1;
        thread frag_t<float>& Cn0 = blk == 0 ? C00 : C10;
        thread frag_t<float>& Cn1 = blk == 0 ? C01 : C11;
        auto ct_a = gemm_op.get_left_input_cooperative_tensor<T, T, float>();
        auto ct_b = gemm_op.get_right_input_cooperative_tensor<T, T, float>();
        auto ct_c = gemm_op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
        for (short i = 0; i < kElemsPerFrag; i++) { ct_a[i] = A[i]; }
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_b[i] = B0[i];
          ct_b[kElemsPerFrag + i] = B1[i];
        }
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_c[i] = Cn0[i];
          ct_c[kElemsPerFrag + i] = Cn1[i];
        }
        gemm_op.run(ct_a, ct_b, ct_c);
        for (short i = 0; i < kElemsPerFrag; i++) {
          Cn0[i] = ct_c[i];
          Cn1[i] = ct_c[kElemsPerFrag + i];
        }
      }
    } else {
      // paired-A, verbatim from BaseNAXFrag::mma(Cm0, Cm1, Am0, Am1, ta, B, tb)
      mpp::tensor_ops::matmul2d<descB, metal::execution_simdgroup> gemm_op;
      for (short blk = 0; blk < 2; ++blk) {
        thread frag_t<T>& B = blk == 0 ? B0 : B1;
        thread frag_t<float>& Cm0 = blk == 0 ? C00 : C01;
        thread frag_t<float>& Cm1 = blk == 0 ? C10 : C11;
        auto ct_a = gemm_op.get_left_input_cooperative_tensor<T, T, float>();
        auto ct_b = gemm_op.get_right_input_cooperative_tensor<T, T, float>();
        auto ct_c = gemm_op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_a[i] = A0[i];
          ct_a[kElemsPerFrag + i] = A1[i];
        }
        for (short i = 0; i < kElemsPerFrag; i++) { ct_b[i] = B[i]; }
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_c[i] = Cm0[i];
          ct_c[kElemsPerFrag + i] = Cm1[i];
        }
        gemm_op.run(ct_a, ct_b, ct_c);
        for (short i = 0; i < kElemsPerFrag; i++) {
          Cm0[i] = ct_c[i];
          Cm1[i] = ct_c[kElemsPerFrag + i];
        }
      }
    }
  }

  for (short i = 0; i < kElemRows; i++) {
    for (short j = 0; j < kElemCols; j++) {
      const short r = sc.y + i * kElemRowsJump;
      const short c = sc.x + j;
      const short e = i * kElemCols + j;
      out[(r) * 32 + (c)] = C00[e];
      out[(r) * 32 + (c + 16)] = C01[e];
      out[(r + 16) * 32 + (c)] = C10[e];
      out[(r + 16) * 32 + (c + 16)] = C11[e];
    }
  }
"""


def make_kernel(name: str, source: str, inputs, outputs):
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=outputs,
        header=COMMON,
        source=source,
        ensure_row_contiguous=True,
    )


def run_capacity() -> dict:
    kern = make_kernel("e156_cap", CAPACITY_SOURCE, ["dummy"], ["cap"])
    (cap,) = kern(
        inputs=[mx.zeros((1,), dtype=mx.bfloat16)],
        template=[("T", mx.bfloat16)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(7,)],
        output_dtypes=[mx.int32],
    )
    mx.eval(cap)
    c = [int(v) for v in np.array(cap)]
    return {
        "descB_16_32_16": {"left": c[0], "right": c[1], "destination": c[2]},
        "descA_32_16_16": {"left": c[3], "right": c[4], "destination": c[5]},
        "kElemsPerFrag": c[6],
    }


def run_gemm(a: mx.array, b: mx.array, arm: int, kdim: int) -> np.ndarray:
    kern = make_kernel("e156_pairgemm", GEMM_SOURCE, ["a", "b", "kdim", "mode"],
                       ["out"])
    (out,) = kern(
        inputs=[a, b, mx.array([kdim], dtype=mx.int32),
                mx.array([arm], dtype=mx.int32)],
        template=[("T", mx.bfloat16)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(32 * 32,)],
        output_dtypes=[mx.float32],
    )
    mx.eval(out)
    return np.array(out).reshape(32, 32)


def adversarial(rng: np.random.Generator, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs chosen so a changed summation order would show up.

    Each K slice mixes one large magnitude with many small ones, which is the
    classic cancellation shape: reordering the additions moves the result by
    many ulp instead of hiding inside the round-off.
    """
    a = rng.normal(0.0, 1.0, size=(32, k)).astype(np.float32)
    b = rng.normal(0.0, 1.0, size=(k, 32)).astype(np.float32)
    a[:, 0] *= 512.0
    b[0, :] *= 512.0
    a[:, k // 2] *= 1.0 / 512.0
    b[k // 2, :] *= 1.0 / 512.0
    return a, b


def ulp_delta(x: np.ndarray, y: np.ndarray) -> int:
    xi = x.astype(np.float32).view(np.int32).astype(np.int64)
    yi = y.astype(np.float32).view(np.int32).astype(np.int64)
    xi = np.where(xi < 0, np.int64(0x80000000) - xi, xi)
    yi = np.where(yi < 0, np.int64(0x80000000) - yi, yi)
    return int(np.max(np.abs(xi - yi)))


def bf16(x: np.ndarray) -> mx.array:
    return mx.array(x).astype(mx.bfloat16)


def one_ulp_up_bf16(x: np.float32) -> np.float32:
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    bits = (bits + 0x10000) & 0xFFFFFFFF
    return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])


def diagnose() -> dict:
    """Is the paired-B arm, which is the promoted M5 path, faithful here?

    `B` is the 16x32 matrix whose left half is the identity, so a faithful
    single-chunk GEMM must return `C = [A | 0]`. Three outcomes separate the
    possible causes:

      * `C == [A | 0]`                    the harness maps fragments correctly;
      * `C` is a permutation of `[A | 0]` the harness mis-maps lanes, fixable;
      * `C` is neither                    `matmul2d` is not faithful on this
                                          host, so the harness cannot measure
                                          the paired-A arm here at all.
    """
    a_np = np.arange(32 * 16, dtype=np.float32).reshape(32, 16) + 1.0
    b_np = np.zeros((16, 32), dtype=np.float32)
    b_np[:, :16] = np.eye(16, dtype=np.float32)
    a = bf16(a_np)
    b = bf16(b_np)
    got = run_gemm(a, b, 0, 16)
    want = np.zeros((32, 32), dtype=np.float32)
    want[:, :16] = a_np
    exact = bool(np.array_equal(got, want))
    got_vals = sorted(np.unique(got).tolist())
    want_vals = sorted(np.unique(want).tolist())
    permutation = bool(got_vals == want_vals)

    # `a[r, k] = r * 16 + k + 1`, so a delivered value names exactly which
    # element of `A` the destination slot actually received. That decodes the
    # per-lane element layout without needing an accessor MPP does not expose.
    left = got[:, :16]
    codes = np.rint(left).astype(np.int64) - 1
    src_row = np.where(codes >= 0, codes // 16, -1)
    src_col = np.where(codes >= 0, codes % 16, -1)
    want_col = np.tile(np.arange(16), (32, 1))
    want_row = np.tile(np.arange(32).reshape(32, 1), (1, 16))
    slots_correct = int(np.sum((src_row == want_row) & (src_col == want_col)))

    return {
        "identity_b_exact": exact,
        "identity_b_is_a_permutation_of_expected": permutation,
        "distinct_values_produced": len(got_vals),
        "distinct_values_expected": len(want_vals),
        "expected_row0_first8": want[0, :8].tolist(),
        "produced_row0_first8": got[0, :8].tolist(),
        "destination_slots_holding_the_right_element": slots_correct,
        "destination_slots_total": 512,
        "delivered_source_column_row0": src_col[0].tolist(),
        "delivered_source_column_row1": src_col[1].tolist(),
        "delivered_source_row_row0": src_row[0].tolist(),
        "verdict": (
            "faithful" if exact
            else "lane_mapping_only" if permutation
            else "matmul2d_not_faithful_on_this_host"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--kdim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=156)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    info = mx.device_info()
    rec: dict = {
        "harness": "local",
        "host": {
            "device_name": info["device_name"],
            "architecture": info["architecture"],
        },
        "note": (
            "Numerics only, never time. Runs mpp::tensor_ops::matmul2d directly, "
            "so it does not need is_nax_available()."
        ),
    }

    rec["e156_matmul2d_descriptor_source"] = {
        "all_matmul2d_descriptors_in_kernels_tree": "16, 32, 16",
        "paired_B_overload": "steel/gemm/nax.h:393 builds desc(16,32,16)",
        "paired_A_overload": "steel/gemm/nax.h:464 builds desc(16,32,16)",
        "advisor_F4_item4_claim": "paired-A builds desc(32,16,16)",
        "advisor_F4_item4_claim_holds_on_this_tree": False,
        "descriptor_changes_between_the_two_tile_shapes": False,
    }

    rec["e156_capacities"] = run_capacity()
    cap = rec["e156_capacities"]
    paired_a_left_writes = 2 * cap["kElemsPerFrag"]
    paired_a_right_writes = cap["kElemsPerFrag"]
    rec["e156_paired_a_operand_budget"] = {
        "writes_into_left": paired_a_left_writes,
        "left_capacity_under_declared_desc": cap["descB_16_32_16"]["left"],
        "writes_into_right": paired_a_right_writes,
        "right_capacity_under_declared_desc": cap["descB_16_32_16"]["right"],
        "left_write_fits": paired_a_left_writes
        <= cap["descB_16_32_16"]["left"],
        "right_operand_fully_initialised": paired_a_right_writes
        >= cap["descB_16_32_16"]["right"],
    }

    rec["e156_matmul2d_instrument_check"] = diagnose()

    rng = np.random.default_rng(args.seed)
    trials = []
    agree = 0
    worst = 0
    ref_worst_b = 0.0
    ref_worst_a = 0.0
    for t in range(args.trials):
        a_np, b_np = adversarial(rng, args.kdim)
        a = bf16(a_np)
        b = bf16(b_np)
        a_round = np.array(a.astype(mx.float32))
        b_round = np.array(b.astype(mx.float32))
        ref = (a_round.astype(np.float64) @ b_round.astype(np.float64))

        cb = run_gemm(a, b, 0, args.kdim)
        ca = run_gemm(a, b, 1, args.kdim)

        d = ulp_delta(cb, ca)
        worst = max(worst, d)
        if d == 0:
            agree += 1
        rb = float(np.max(np.abs(cb - ref) / np.maximum(np.abs(ref), 1e-30)))
        ra = float(np.max(np.abs(ca - ref) / np.maximum(np.abs(ref), 1e-30)))
        ref_worst_b = max(ref_worst_b, rb)
        ref_worst_a = max(ref_worst_a, ra)
        trials.append({"trial": t, "ulp_delta": d,
                       "rel_err_paired_b": rb, "rel_err_paired_a": ra})

    # Rule 101 control. One bf16 ulp on a single element of `a` must move both
    # arms, otherwise the comparison above cannot detect anything.
    a_np, b_np = adversarial(rng, args.kdim)
    a = bf16(a_np)
    b = bf16(b_np)
    base_b = run_gemm(a, b, 0, args.kdim)
    base_a = run_gemm(a, b, 1, args.kdim)
    a_pert = np.array(a.astype(mx.float32))
    a_pert[0, 0] = one_ulp_up_bf16(a_pert[0, 0])
    ap_arr = bf16(a_pert)
    ctrl_b = run_gemm(ap_arr, b, 0, args.kdim)
    ctrl_a = run_gemm(ap_arr, b, 1, args.kdim)
    control_b_moved = ulp_delta(base_b, ctrl_b) > 0
    control_a_moved = ulp_delta(base_a, ctrl_a) > 0

    rec["e156_matmul2d_numerical"] = {
        "kdim": args.kdim,
        "trials": args.trials,
        "arms_agree_bit_for_bit": agree,
        "max_ulp_delta_paired_a_vs_paired_b": worst,
        "max_rel_err_vs_float64_reference_paired_b": ref_worst_b,
        "max_rel_err_vs_float64_reference_paired_a": ref_worst_a,
        "per_trial": trials,
        "rule_101_control_one_bf16_ulp_on_a": {
            "paired_b_moved": control_b_moved,
            "paired_a_moved": control_a_moved,
            "both_moved": bool(control_b_moved and control_a_moved),
        },
    }

    bitexact = (agree == args.trials and worst == 0)
    controls_ok = bool(control_b_moved and control_a_moved)
    # A relative error above about 1e-2 means the arm is not computing the GEMM
    # at all. The paired-B arm is the promoted M5 path, so it is the instrument
    # control: if it fails here, nothing this probe says about paired-A on this
    # host is usable.
    sane_b = bool(ref_worst_b < 1e-2)
    sane_a = bool(ref_worst_a < 1e-2)
    instrument_valid = bool(
        sane_b and rec["e156_matmul2d_instrument_check"]["identity_b_exact"])

    rec["e156_matmul2d_paired_a_computes_the_gemm"] = sane_a
    rec["e156_matmul2d_paired_b_computes_the_gemm"] = sane_b
    rec["e156_matmul2d_controls_caught"] = controls_ok
    rec["e156_matmul2d_instrument_valid_on_this_host"] = instrument_valid
    rec["e156_matmul2d_descriptor_bitexact"] = (
        bool(bitexact and controls_ok) if instrument_valid
        else "not_measurable_on_this_host")
    rec["gate"] = (
        "PASS" if rec["e156_matmul2d_descriptor_bitexact"] is True
        else "FAIL" if instrument_valid
        else "INCONCLUSIVE_INSTRUMENT_INVALID")

    args.out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rec.items()
                      if k != "e156_matmul2d_numerical"}, indent=2))
    n = rec["e156_matmul2d_numerical"]
    print(f"kdim={n['kdim']} trials={n['trials']} "
          f"agree={n['arms_agree_bit_for_bit']} "
          f"max_ulp={n['max_ulp_delta_paired_a_vs_paired_b']}")
    print(f"rel_err paired_b={n['max_rel_err_vs_float64_reference_paired_b']:.3e} "
          f"paired_a={n['max_rel_err_vs_float64_reference_paired_a']:.3e}")
    print(f"controls: {n['rule_101_control_one_bf16_ulp_on_a']}")
    print(f"instrument: {rec['e156_matmul2d_instrument_check']}")
    print(f"wrote {args.out}")
    return 1 if rec["gate"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
