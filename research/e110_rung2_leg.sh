#!/usr/bin/env bash
# One E110 rung-2 leg: select an arm -> commit -> build -> witness -> measure
# -> unwind.
#
#   usage: research/e110_rung2_leg.sh ARM TAG [TOKENS]
#
# ARM is `xv4` (the branch tip bytes, unpatched) or `base` (the same tree with
# `research/e110-artifacts/e110-xv4.patch` reverted, which restores the four
# scalar activation reads).
#
# The witness is a STRING witness, which is the correct one here: the
# runtime-effective source of the `quantized` family is the JIT string compiled
# into the worker binary, so the vector load either is in that string or it is
# not. `--forbid` on the arm the leg is NOT running is what makes the witness
# able to fail, and because the two arms swap require and forbid, each of the
# three needles below is proven able to fire inside every ABBA session.
#
# The needles below are literal. `rebuild-and-assert-worker.sh` matches with
# `grep -F` since HARNESS DEFECT 14 was fixed, so brackets must NOT be escaped.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: e110_rung2_leg.sh ARM TAG [TOKENS]}"
tag="${2:?usage: e110_rung2_leg.sh ARM TAG [TOKENS]}"
tokens="${3:-512}"

readonly HEADER="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
readonly TWIN="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
readonly PATCH="research/e110-artifacts/e110-xv4.patch"

readonly XV4_LOAD='const vec<T, 4> xv = '
readonly XV4_SUM='sums[m] += xv[0] + xv[1] + xv[2] + xv[3];'
readonly BASE_SUM='sums[m] += xm[0] + xm[1] + xm[2] + xm[3];'

case "${arm}" in
  xv4)  require=("${XV4_LOAD}" "${XV4_SUM}"); forbid=("${BASE_SUM}") ;;
  base) require=("${BASE_SUM}"); forbid=("${XV4_LOAD}" "${XV4_SUM}") ;;
  *) echo "e110_rung2_leg: unknown arm ${arm}" >&2; exit 2 ;;
esac

assert_args=()
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
  echo "e110_rung2_leg: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi

if [[ "${arm}" == "base" ]]; then
  git apply -R "${PATCH}" || {
    echo "e110_rung2_leg: ${PATCH} does not revert against ${pre_patch_sha}" >&2
    exit 2
  }
  python3 research/twin_audit.py > /tmp/e110-rung2-twin-${arm}.txt 2>&1 || {
    echo "e110_rung2_leg: twin audit failed for arm ${arm}" >&2
    tail -20 /tmp/e110-rung2-twin-${arm}.txt >&2
    exit 3
  }
  git add -- "${HEADER}" "${TWIN}"
  git commit -q -m "E110 leg ${tag}: TRANSIENT base bytes under measurement

Unwound to ${pre_patch_sha} when the leg exits, including on a crash, so the
branch's scored surface stays byte-identical between legs."
  transient_sha="$(git rev-parse HEAD)"
fi

out="research/out/${tag}"
mkdir -p "${out}"

senpai/rebuild-and-assert-worker.sh "${assert_args[@]}" \
  > "${out}/worker-assert-pre.txt" 2>&1 || {
  echo "e110_rung2_leg: worker assert failed before ${tag}" >&2
  tail -20 "${out}/worker-assert-pre.txt" >&2
  exit 5
}
pre_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
cp "${out}/worker-assert-pre.txt" "/tmp/e110-rung2-assert-pre.txt"

# Timing legs run untraced. The trace writes one `mtp-row:` line per target
# row, which is the row-ledger evidence the exactness legs need and a cost the
# timed legs must not carry.
trace_flag="--no-trace"
[[ "${E110_TRACE:-0}" == "1" ]] && trace_flag=""
research/e79_trace_leg.sh "${tag}" "${tokens}" ${trace_flag}
status=$?

# e79_trace_leg.sh recreates the leg directory, so the pre-assert record is
# copied back after the run rather than before it.
cp "/tmp/e110-rung2-assert-pre.txt" "${out}/worker-assert-pre.txt"

senpai/rebuild-and-assert-worker.sh --no-build "${assert_args[@]}" \
  > "${out}/worker-assert-post.txt" 2>&1
post_rc=$?
post_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-post.txt" | tail -1)"

{
  echo "e110_arm=${arm}"
  echo "experiment=${E110_EXPERIMENT:-e110-rung2}"
  echo "branch_commit=${pre_patch_sha}"
  echo "measured_commit_unwound=${transient_sha:-<tip>}"
  echo "worker_sha256_pre=${pre_worker}"
  echo "worker_sha256_post=${post_worker}"
  echo "worker_assert_post_exit=${post_rc}"
} >> "${out}/meta.txt"

if ((post_rc != 0)) || [[ "${post_worker}" != "${pre_worker}" ]]; then
  echo "e110_rung2_leg: ${tag} worker changed or failed its post-assert; discarding" >&2
  status=7
fi

echo "status=${status}" >> "${out}/meta.txt"
exit "${status}"
