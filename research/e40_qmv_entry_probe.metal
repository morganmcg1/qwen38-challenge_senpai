// E40: the PRODUCTION entry point, not an isolated NA cell.
//
// research/crossrow_na_probe.metal says in its own header that "the production
// entry point inlines every NA at once, so its AIR cannot be attributed to one
// packing factor", and every register number in the campaign (E13/E27/E32/E36)
// was therefore taken from an isolated cell. That is the right instrument for
// "what does one NA cost" and the WRONG instrument for the E40 question.
//
// `affine_qmv_fast` switches on `ntg.x`, a RUNTIME value, inside a single
// [[kernel]] (quantized.h:1869, switch at :1925). Every width cell 2..9 for
// both the >=4096 and <4096 output branches is inlined into one function, so
// the compiled kernel carries ONE register/private-memory footprint equal to
// the worst branch. E27 raised only the M=5 and M=9 cells, but if it raised the
// kernel-wide peak, every width pays -- which is exactly H1's mechanism.
//
// This probe instantiates the shipped entry point at the scored template
// arguments so the whole-kernel footprint can be compared across two commits:
//
//   research/e40_entry_air_diff.sh
//
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

// The scored decode cell: bfloat16 activations, affine 4-bit group-64
// backbone, non-batched. This is the instantiation the `ntg.x` switch and the
// wide cross-row family live in (quantized.h:1918-1968).
template [[host_name("e40_affine_qmv_fast_bf16_gs64_b4_batch0")]] [[kernel]]
decltype(affine_qmv_fast<bfloat16_t, 64, 4, false>)
    affine_qmv_fast<bfloat16_t, 64, 4, false>;

// The batched twin, for contrast: `batched == true` still reaches the same
// switch, so it must move identically or the comparison is unsound.
template [[host_name("e40_affine_qmv_fast_bf16_gs64_b4_batch1")]] [[kernel]]
decltype(affine_qmv_fast<bfloat16_t, 64, 4, true>)
    affine_qmv_fast<bfloat16_t, 64, 4, true>;

// The 2-bit draft-readout cell, which E27 does not touch: a negative control.
template [[host_name("e40_affine_qmv_fast_bf16_gs64_b2_batch0")]] [[kernel]]
decltype(affine_qmv_fast<bfloat16_t, 64, 2, false>)
    affine_qmv_fast<bfloat16_t, 64, 2, false>;
