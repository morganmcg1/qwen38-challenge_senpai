#!/usr/bin/env bash
# E121 rung 3: the pre-submit chain on the exact tree that would be submitted.
#
#   usage: research/e121_presubmit.sh [TOKENS]
#
# Order matters. The cheap static gates run first, then `swift test` on both
# trees, then the one expensive gated `--local-submit` leg. A static gate that
# fails must not cost a full submit leg.
#
# THE `swift test` CONTROL. Non-zero exit is the campaign floor, not a
# regression, so the chain records the failing-name set for the candidate tree
# AND for the same tree with `research/e121-artifacts/e121-share.patch`
# reverse-applied. Only names the candidate ADDS are a regression.
#
# THE WORKER WITNESS. `benchmark-qwen-mtp.sh` does not rebuild the worker
# (finding 28), so the worker is witnessed before and after the submit leg and
# both digests are recorded. Equal digests are what prove the timed binary is
# the witnessed binary.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
out="research/out/e121-presubmit"
mkdir -p "${out}" research/e121-artifacts

readonly HEADER="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
readonly TWIN="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
readonly PATCH="research/e121-artifacts/e121-share.patch"

readonly WITNESS=(
  --require 'qwen35_dual_rms_norm_concat_bf16_v1'
  --forbid  'qwen35_dual_rms_norm_bf16_v1'
  --require 'qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>'
  --forbid  'qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>'
  --require 'qwen_mtp_draft_selected_affine4_rerank_g64_v1'
  --require 'qwen_mtp_row_top32_partial'
  --forbid  'MLX_E85_GATHER_QMM'
  --require-symbol 'snapshotScheduleSignal'
  --require 'constexpr bool SHARE_SUMS = NA <= 4;'
  --require 'threadgroup float sums_xchg[1 * 4 * 32];'
  --forbid  'sums[m] += load_vector'
)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e121_presubmit: worktree is dirty; refusing to certify uncommitted work" >&2
  exit 1
fi
pre_patch_sha="$(git rev-parse HEAD)"

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
  echo "e121_presubmit: ${PATCH} does not revert against ${pre_patch_sha}" >&2
  exit 2
}
git add -- "${HEADER}" "${TWIN}"
git commit -q -m "E121 presubmit: TRANSIENT base bytes for the swift test floor

Unwound to ${pre_patch_sha} when the script exits."
transient_sha="$(git rev-parse HEAD)"
swift test --force-resolved-versions > "${out}/swift-test-base.log" 2>&1
echo "base swift test exit $?"
unwind
transient_sha=""

echo "=== rebuild and witness the candidate worker ==="
senpai/rebuild-and-assert-worker.sh "${WITNESS[@]}" \
  > "${out}/worker-assert-pre.txt" 2>&1 || {
  echo "e121_presubmit: worker assert failed" >&2
  tail -30 "${out}/worker-assert-pre.txt" >&2
  exit 5
}
worker_sha="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
worker_mtime="$(awk '/^worker_mtime /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"

echo "=== ./benchmark-qwen-mtp.sh --local-submit (real 40 C gate) ==="
head_dir="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
[[ -s "${head_dir}/config.json" ]] && export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${tokens}"
# MLX_E58_BUFFER_LIMIT_OPS stays unset. `0` isolates one dispatch per command
# buffer, which is right for an isolated kernel census and wrong here: this leg
# must reproduce the command-buffer concurrency of a real round, match the
# rung-3 timed legs, and match the ranked runner, which sets no such override.
unset MLX_E58_BUFFER_LIMIT_OPS
export MLXFAST_SCORE_PATH="${PWD}/${out}/submit-score.json"
unset MLXFAST_LOCAL_COOL_GATE
./benchmark-qwen-mtp.sh --local-submit > "${out}/local-submit.log" 2>&1
submit_status=$?
echo "local-submit exit ${submit_status}"

senpai/rebuild-and-assert-worker.sh --no-build "${WITNESS[@]}" \
  > "${out}/worker-assert-post.txt" 2>&1
post_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-post.txt" | tail -1)"

python3 research/e121_presubmit_receipt.py \
  --submit-log "${out}/local-submit.log" \
  --submit-score "${out}/submit-score.json" \
  --test-log-candidate "${out}/swift-test-candidate.log" \
  --test-log-base "${out}/swift-test-base.log" \
  --worker-sha256 "${worker_sha}" \
  --worker-mtime "${worker_mtime}" \
  --worker-sha256-post "${post_worker}" \
  --out research/e121-artifacts/rung3-presubmit.json
exit $?
