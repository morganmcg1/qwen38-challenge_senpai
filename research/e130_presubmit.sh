#!/usr/bin/env bash
# E130 rung 3: the pre-submit chain on the exact tree that would be submitted.
#
#   usage: research/e130_presubmit.sh [TOKENS]
#
# Same shape as research/e121_presubmit.sh: cheap static gates first, then
# `swift test` on the candidate tree and on the same tree with the arm reverted,
# then the one expensive gated `--local-submit` leg.
#
# THE WITNESS SET IS NOT E121'S. Three needles in the E121 set describe the
# E121 arm, which the campaign reverted, and one more describes a Qwen35.swift
# state that Route B superseded. Carrying them forward would fail a correct
# build, so each is inverted here against the current base and the inversion is
# the guard that catches a stale worker:
#
#   E121 set                                    here            why
#   require 'constexpr bool SHARE_SUMS ...'     forbid          arm reverted
#   require 'threadgroup float sums_xchg[...]'  forbid          arm reverted
#   forbid  'sums[m] += load_vector'            require         base bytes back
#   forbid  'qwen35_dual_rms_norm_bf16_v1'      require         live on this base
#
# THE DISCRIMINATING WITNESS FOR THIS ARM IS AN ABSENCE. `prune_na5_pair`
# removes the `<T, 5, 5, true>` instantiation, so `--require` on it fails a
# correct build. `qmv_fast_crossrow_affine4_g64<T, 5>` is NOT a witness: the
# n < 4096 branch already instantiates it, so it is present on both sides.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
out="research/out/e130-presubmit"
mkdir -p "${out}" research/e130-artifacts

readonly HEADER="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
readonly TWIN="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
readonly PATCH="research/e130-artifacts/e130-prune-na5.patch"
readonly BASE_SHA="f880eb26517cd7a3a153799348cb4f2caf8e4e58"

readonly WITNESS=(
  --forbid  'qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>'
  --forbid  'qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>'
  --require 'qmv_fast_crossrow_affine4_g64_m<T, 4, 4, true>'
  --require 'qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>'
  --require 'qwen35_dual_rms_norm_concat_bf16_v1'
  --require 'qwen35_dual_rms_norm_bf16_v1'
  --require 'qwen_mtp_draft_selected_affine4_rerank_g64_v1'
  --require 'qwen_mtp_row_top32_partial'
  --require 'sums[m] += load_vector'
  --forbid  'constexpr bool SHARE_SUMS = NA <= 4;'
  --forbid  'threadgroup float sums_xchg[1 * 4 * 32];'
  --forbid  'MLX_E85_GATHER_QMM'
  --require-symbol 'snapshotScheduleSignal'
)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e130_presubmit: worktree is dirty; refusing to certify uncommitted work" >&2
  exit 1
fi
pre_patch_sha="$(git rev-parse HEAD)"

git diff "${BASE_SHA}..HEAD" -- "${HEADER}" "${TWIN}" > "${PATCH}"
if [[ ! -s "${PATCH}" ]]; then
  echo "e130_presubmit: no arm diff against ${BASE_SHA}; nothing to certify" >&2
  exit 2
fi

echo "=== swift test on the candidate tree ==="
swift test --force-resolved-versions > "${out}/swift-test-candidate.log" 2>&1
echo "candidate swift test exit $?"

echo "=== swift test on the same tree with the arm reverted ==="
transient_sha=""
unwind() {
  if [[ -n "${transient_sha}" && "$(git rev-parse HEAD)" == "${transient_sha}" ]]; then
    git reset -q "${pre_patch_sha}"
  fi
  git checkout -q "${pre_patch_sha}" -- "${HEADER}" "${TWIN}" 2>/dev/null || true
}
trap unwind EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

git apply -R "${PATCH}" || {
  echo "e130_presubmit: ${PATCH} does not revert against ${pre_patch_sha}" >&2
  exit 2
}
git add -- "${HEADER}" "${TWIN}"
git commit -q -m "E130 presubmit: TRANSIENT base bytes for the swift test floor

Unwound to ${pre_patch_sha} when the script exits."
transient_sha="$(git rev-parse HEAD)"
swift test --force-resolved-versions > "${out}/swift-test-base.log" 2>&1
echo "base swift test exit $?"
unwind
transient_sha=""

echo "=== rebuild and witness the candidate worker ==="
senpai/rebuild-and-assert-worker.sh "${WITNESS[@]}" \
  > "${out}/worker-assert-pre.txt" 2>&1 || {
  echo "e130_presubmit: worker assert failed" >&2
  tail -40 "${out}/worker-assert-pre.txt" >&2
  exit 5
}
worker_sha="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
worker_mtime="$(awk '/^worker_mtime /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"

echo "=== ./benchmark-qwen-mtp.sh --local-submit (real 40 C gate) ==="
head_dir="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
[[ -s "${head_dir}/config.json" ]] && export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${tokens}"
unset MLX_E58_BUFFER_LIMIT_OPS
export MLXFAST_SCORE_PATH="${PWD}/${out}/submit-score.json"
unset MLXFAST_LOCAL_COOL_GATE
./benchmark-qwen-mtp.sh --local-submit > "${out}/local-submit.log" 2>&1
echo "local-submit exit $?"

senpai/rebuild-and-assert-worker.sh --no-build "${WITNESS[@]}" \
  > "${out}/worker-assert-post.txt" 2>&1
post_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-post.txt" | tail -1)"

python3 research/e121_presubmit_receipt.py \
  --experiment e130-rung3-presubmit \
  --arm prune_na5_pair \
  --base-sha "${BASE_SHA}" \
  --patch "${PATCH}" \
  --submit-log "${out}/local-submit.log" \
  --submit-score "${out}/submit-score.json" \
  --test-log-candidate "${out}/swift-test-candidate.log" \
  --test-log-base "${out}/swift-test-base.log" \
  --worker-sha256 "${worker_sha}" \
  --worker-mtime "${worker_mtime}" \
  --worker-sha256-post "${post_worker}" \
  --out research/e130-artifacts/rung3-presubmit.json
exit $?
