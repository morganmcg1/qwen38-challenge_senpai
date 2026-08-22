#!/usr/bin/env bash
# E129 rung 0: the pre-submit chain for the reverted Route B candidate.
#
#   usage: research/e129_presubmit.sh [TOKENS]
#
# WHY THIS IS NOT `research/e121_presubmit.sh`. That script reverse-applies
# `research/e121-artifacts/e121-share.patch` to build its `swift test` control
# and asserts the three E121 witnesses in their pre-revert polarity. Alphonse's
# revert removed that arm from the base, so the patch no longer applies and all
# three witnesses now read the other way. This chain keeps the same order and
# the same gates, and changes only those two things.
#
# THE `swift test` CONTROL. This branch changes no submitted byte against the
# advisor head, so there is no candidate-versus-base tree to compare. The gate
# is `senpai/known-test-failures.md`: the nine organizer names must sum to
# exactly 40 issues with unchanged per-name counts, and no other failing name
# may appear that is not a listed campaign-added test.
#
# THE WORKER WITNESS. `benchmark-qwen-mtp.sh` does not rebuild the worker
# (finding 28), so the worker is witnessed before and after the submit leg and
# both digests are recorded. Equal digests are what prove the timed binary is
# the witnessed binary.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
out="research/out/e129-presubmit"
mkdir -p "${out}"

# The standing set, with the three E121 witnesses inverted by the revert, plus
# the Route B pipelines.
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
  # The dispatch table itself is interpolated into the Metal source, so no
  # `qwen_e120_qmv_m<...>` instantiation can ever be witnessed here. The plan
  # witness is the literal that carries it. Both tables are compiled in, so
  # `strings` proves which tables exist, not which one a run selects; the
  # pipeline log proves the selection at run time. `renderPlan` equality with
  # each literal is asserted by `planWitnessMatchesWidthPlan`.
  --require 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:3:4,7:4:4,8:4:4,9:3:4'
  --require 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:6:2,7:7:2,8:4:4,9:3:4'
)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e129_presubmit: worktree is dirty; refusing to certify uncommitted work" >&2
  git status --porcelain >&2
  exit 1
fi

echo "=== swift test on the candidate tree ==="
swift test --force-resolved-versions > "${out}/swift-test.log" 2>&1
echo "swift test exit $?"

echo "=== rebuild and witness the candidate worker ==="
senpai/rebuild-and-assert-worker.sh "${WITNESS[@]}" \
  > "${out}/worker-assert-pre.txt" 2>&1 || {
  echo "e129_presubmit: worker assert failed" >&2
  tail -40 "${out}/worker-assert-pre.txt" >&2
  exit 5
}
worker_sha="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
worker_mtime="$(awk '/^worker_mtime /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
echo "worker ${worker_sha:0:16} mtime ${worker_mtime}"

echo "=== ./benchmark-qwen-mtp.sh --local-submit (real 40 C gate) ==="
head_dir="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e129_presubmit: declared head missing at ${head_dir}" >&2
  exit 6
fi
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${tokens}"
unset MLX_E58_BUFFER_LIMIT_OPS
unset MLXFAST_LOCAL_COOL_GATE
export MLXFAST_SCORE_PATH="${PWD}/${out}/submit-score.json"
./benchmark-qwen-mtp.sh --local-submit > "${out}/local-submit.log" 2>&1
echo "local-submit exit $?"

senpai/rebuild-and-assert-worker.sh --no-build "${WITNESS[@]}" \
  > "${out}/worker-assert-post.txt" 2>&1
post_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-post.txt" | tail -1)"

python3 research/e129_presubmit_receipt.py \
  --submit-log "${out}/local-submit.log" \
  --submit-score "${out}/submit-score.json" \
  --test-log "${out}/swift-test.log" \
  --worker-sha256 "${worker_sha}" \
  --worker-mtime "${worker_mtime}" \
  --worker-sha256-post "${post_worker}" \
  --out "${out}/rung0-presubmit.json"
exit $?
