// E46: AIR census of the register allocation a dispatch table implies.
//
// Two readouts, because they answer different questions and disagree by ~55
// registers:
//
//   E46_CELL_M / E46_CELL_IPG   one width case, `..._m<T, M, IPG, true>`.
//                               The max over the seven width cases is the
//                               campaign's "kernel-wide max" (108 at the tip,
//                               129 on the NA<=5 table), and it is the number
//                               stop rule 3 is written against.
//   default                     the real `affine_qmv_fast` entry template, into
//                               which every width case, `qmv_fast_impl` and the
//                               2-bit draft readout all inline. Strictly larger,
//                               and reported as a secondary check that an arm
//                               did not perturb the shared allocation.
//
// The include path decides which dispatch table is measured, so an arm is a
// patched copy of quantized.h in a shadow include dir, never an edit here.
//
//   python3 research/e46_reg_census.py

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#if defined(E46_CELL_M)

[[kernel]] void e46_cell(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, E46_CELL_M, E46_CELL_IPG, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}

#else

#define e46_instantiate(name, type, group_size, bits, batched)             \
  instantiate_kernel(                                                      \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched,   \
      name, type, group_size, bits, batched)

// bfloat16 is the scored dtype: the offline transform writes bf16 scales and
// biases, and batched=0 is the shape the width table dispatches through.
e46_instantiate(affine_qmv_fast, bfloat16_t, 64, 4, 0)
e46_instantiate(affine_qmv_fast, bfloat16_t, 64, 4, 1)

#endif
