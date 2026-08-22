#!/usr/bin/env bash
# One E121 rung-3 leg: select an arm -> commit -> build -> witness -> measure
# -> unwind. Structure follows `research/e110_rung2_leg.sh`, which is the
# in-situ design this campaign already validated.
#
#   usage: research/e121_e2e_leg.sh ARM TAG [TOKENS]
#
# ARM is `share` (the branch tip bytes, gated cross-simdgroup chunk-sum share)
# or `base` (the same tree with `research/e121-artifacts/e121-share.patch`
# reverted, which restores the redundant per-simdgroup chunk sum).
#
# STRING WITNESSES, NOT SYMBOL WITNESSES. The runtime-effective source of the
# `quantized` family is the JIT string compiled into the worker binary, so the
# gate either is in that string or it is not. The two arms swap require and
# forbid on the same three needles, so every needle is proven able to fire in
# both directions inside one session.
#
# WHY THE GATE NEEDS ITS OWN NEEDLE. `if constexpr (NA <= 4)` is a construct
# whose CORRECT behaviour at NA = 5 is to look exactly like the base. The
# standing template-instantiation needles are therefore satisfied by both arms
# and cannot separate them. Without `SHARE_SUMS` and `sums_xchg` in the
# require set a silent transplant failure would assert green and this leg would
# time an unchanged binary.
#
# Needles are literal: `rebuild-and-assert-worker.sh` matches with `grep -F`.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: e121_e2e_leg.sh ARM TAG [TOKENS]}"
tag="${2:?usage: e121_e2e_leg.sh ARM TAG [TOKENS]}"
tokens="${3:-512}"

readonly HEADER="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
readonly TWIN="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
readonly PATCH="research/e121-artifacts/e121-share.patch"

readonly GATE='constexpr bool SHARE_SUMS = NA <= 4;'
readonly XCHG='threadgroup float sums_xchg[1 * 4 * 32];'
readonly BASE_SUM='sums[m] += load_vector'

case "${arm}" in
  share) require=("${GATE}" "${XCHG}"); forbid=("${BASE_SUM}") ;;
  base)  require=("${BASE_SUM}"); forbid=("${GATE}" "${XCHG}") ;;
  *) echo "e121_e2e_leg: unknown arm ${arm}" >&2; exit 2 ;;
esac

# The standing campaign witness set. It is invariant to this arm, so it guards
# the rest of the tree while the three needles above separate the two arms.
standing=(
  --require 'qwen35_dual_rms_norm_concat_bf16_v1'
  --forbid  'qwen35_dual_rms_norm_bf16_v1'
  --require 'qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>'
  --forbid  'qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>'
  --require 'qwen_mtp_draft_selected_affine4_rerank_g64_v1'
  --require 'qwen_mtp_row_top32_partial'
  --forbid  'MLX_E85_GATHER_QMM'
  --require-symbol 'snapshotScheduleSignal'
)

assert_args=("${standing[@]}")
for needle in "${require[@]}"; do assert_args+=(--require "${needle}"); done
for needle in "${forbid[@]}"; do assert_args+=(--forbid "${needle}"); done

pre_patch_sha="$(git rev-parse HEAD)"
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

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e121_e2e_leg: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi

if [[ "${arm}" == "base" ]]; then
  git apply -R "${PATCH}" || {
    echo "e121_e2e_leg: ${PATCH} does not revert against ${pre_patch_sha}" >&2
    exit 2
  }
  python3 research/twin_audit.py > "/tmp/e121-twin-${arm}.txt" 2>&1 || {
    echo "e121_e2e_leg: twin audit failed for arm ${arm}" >&2
    tail -20 "/tmp/e121-twin-${arm}.txt" >&2
    exit 3
  }
  git add -- "${HEADER}" "${TWIN}"
  git commit -q -m "E121 leg ${tag}: TRANSIENT base bytes under measurement

Unwound to ${pre_patch_sha} when the leg exits, including on a crash, so the
branch's scored surface stays byte-identical between legs."
  transient_sha="$(git rev-parse HEAD)"
fi

out="research/out/${tag}"
mkdir -p "${out}"

senpai/rebuild-and-assert-worker.sh "${assert_args[@]}" \
  > "${out}/worker-assert-pre.txt" 2>&1 || {
  echo "e121_e2e_leg: worker assert failed before ${tag}" >&2
  tail -30 "${out}/worker-assert-pre.txt" >&2
  exit 5
}
pre_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
cp "${out}/worker-assert-pre.txt" "/tmp/e121-assert-pre.txt"

# Timing legs run untraced. The trace writes one `mtp-row:` line per target
# row, which the exactness legs need and the timed legs must not pay for.
trace_flag="--no-trace"
[[ "${E121_TRACE:-0}" == "1" ]] && trace_flag=""
research/e79_trace_leg.sh "${tag}" "${tokens}" ${trace_flag}
status=$?

# e79_trace_leg.sh recreates the leg directory, so the pre-assert record is
# copied back after the run rather than before it.
cp "/tmp/e121-assert-pre.txt" "${out}/worker-assert-pre.txt"

senpai/rebuild-and-assert-worker.sh --no-build "${assert_args[@]}" \
  > "${out}/worker-assert-post.txt" 2>&1
post_rc=$?
post_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-post.txt" | tail -1)"

{
  echo "e121_arm=${arm}"
  echo "experiment=${E121_EXPERIMENT:-e121-rung3}"
  echo "branch_commit=${pre_patch_sha}"
  echo "measured_commit_unwound=${transient_sha:-<tip>}"
  echo "worker_sha256_pre=${pre_worker}"
  echo "worker_sha256_post=${post_worker}"
  echo "worker_assert_post_exit=${post_rc}"
} >> "${out}/meta.txt"

if ((post_rc != 0)) || [[ "${post_worker}" != "${pre_worker}" ]]; then
  echo "e121_e2e_leg: ${tag} worker changed or failed its post-assert; discarding" >&2
  status=7
fi

echo "status=${status}" >> "${out}/meta.txt"
exit "${status}"
