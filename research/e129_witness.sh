#!/usr/bin/env bash
# E129 — the one witness set for the candidate worker.
#
#   source research/e129_witness.sh      defines WITNESS
#   research/e129_witness.sh [ARGS...]   rebuilds and asserts the worker
#
# WHY THIS IS ITS OWN FILE. The witness set is the only thing that proves the
# timed binary is the tree the receipt names. Two callers need it: the full
# pre-submit chain, and the plain rebuild that must happen before the 512-token
# digest. A copied array would let those two drift, and the copy that drifted
# would be the one that certified a submission.
#
# `qwen_e120_xsums` is NOT used. It is the synthetic entry-point name that
# `research/e120_census.py` wraps around the fill body for the register census;
# it appears in no shipped source and in no built worker, so requiring it would
# fail a correct build. The shipped fill pipeline is
# `qwen35_custom_affine4_g64_xsums_v1`, and that name is required below.
readonly WITNESS=(
  --require 'qwen35_dual_rms_norm_concat_bf16_v1'
  --forbid  'qwen35_dual_rms_norm_bf16_v1'
  --require 'qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>'
  --forbid  'qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>'
  --require 'qwen_mtp_draft_selected_affine4_rerank_g64_v1'
  --require 'qwen_mtp_row_top32_partial'
  --forbid  'MLX_E85_GATHER_QMM'
  --require-symbol 'snapshotScheduleSignal'
  --forbid  'constexpr bool SHARE_SUMS = NA <= 4;'
  --forbid  'threadgroup float sums_xchg[1 * 4 * 32];'
  --require 'sums[m] += load_vector'
  --require 'qwen35_custom_affine4_g64_qmv_wide_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_sums_v2'
  --forbid  'qwen35_custom_affine4_g64_qmv_wide_v1'
  --forbid  'qwen35_custom_affine4_g64_qmv_wide_sums_v1'
  --require 'qwen35_custom_affine4_g64_xsums_v1'
  --require 'inline void qwen_e120_qmv_m('
  --require 'template <int NA, int RPS, bool USE_TABLE>'
  # Every tiered entry point, as a whole literal. Concatenated names never
  # reach the string table, so these are only assertable because
  # `qwen35E120QMVName` is a total switch over whole literals.
  --require 'qwen35_custom_affine4_g64_qmv_wide_na3_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_na4_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_na5_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_na6_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_na7_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_sums_na3_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_sums_na4_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_sums_na5_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_sums_na6_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_sums_na7_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_na8_v2'
  --require 'qwen35_custom_affine4_g64_qmv_wide_sums_na8_v2'
  # D_S, askeladd's E132 code motion. The metadata loads must sit inside the
  # accumulate loop and the table read must sit below the product loop, or the
  # timed binary is the pre-D_S body and the g17s spill at NA=7 is back.
  --require 'const float scale_local_r = scales[group_index];'
  --forbid  'thread float scale_local[rows_per_simd];'
  # THE WITNESS THAT DECIDES THE RECEIPT. The ranked runner sets no
  # environment, so the shipped route is whatever `entry` and `table` fall back
  # to. This literal is built only for the selected pair, so it can fail.
  --require 'e120_default_route/tiered_switch/onepass67'
  --forbid  'e120_default_route/shared_switch/shipped'
  --forbid  'e120_default_route/tiered_switch/shipped'
  --forbid  'e120_default_route/tiered_switch/onepass678'
  # The dispatch table itself is interpolated into the Metal source, so no
  # `qwen_e120_qmv_m<...>` instantiation can ever be witnessed here. The plan
  # witness is the literal that carries it. Every table's literal is compiled
  # in whichever table is the default, so these four prove which tables exist
  # and CANNOT prove which one a run selects. The `e120_default_route` literal
  # above is the gate; these are inventory. `renderPlan` equality with each
  # literal is asserted by `planWitnessMatchesWidthPlan`.
  --require 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:3:4,7:4:4,8:4:4,9:3:4'
  --require 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:6:4,7:4:4,8:4:4,9:3:4'
  --require 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:6:4,7:7:4,8:4:4,9:3:4'
  --require 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:6:4,7:7:4,8:8:4,9:3:4'
  # The refuted `rps = 2` plan is deleted, not parked behind a flag.
  --forbid  'e120_width_plan/3:3:4,4:4:4,5:5:4,6:6:2,7:7:2,8:4:4,9:3:4'
)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -uo pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  senpai/rebuild-and-assert-worker.sh "$@" "${WITNESS[@]}"
  exit $?
fi
