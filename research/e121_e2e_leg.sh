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

# The refusal is scoped to the SCORED surface, not to the whole worktree. What
# must be committed is the code that will run, so that the W&B result names a
# reproducible commit. An edit under `research/` cannot reach the timed binary,
# and forbidding it would mean no analysis or note could be written during the
# ninety minutes a session takes.
if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e121_e2e_leg: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
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

# MATCHED PREPARATION PHASE. `swift build` is driven by file stat, so a leg
# whose arm repeats the previous leg's arm used to find both sources untouched,
# skip the compile, and reach the timed run with far less GPU idle than a leg
# that rebuilt. Under the base/share/share/base order that short leg is always
# position 3 and always `share`, so the deficit was not a position effect that
# counterbalancing removes; it was attached to the candidate arm. Measured over
# the 2026-08-22T04:06Z session it was 50-53 s of prep for positions 1, 2 and 4
# against 11-12 s for position 3, worth 8 C of entry temperature, plus a further
# ~48 s because only a rebuilt worker pays the first-use Metal JIT compile.
#
# Touching both sources forces every leg through the same compile and the same
# JIT, which reproduces the thermal history rather than approximating it: an
# equal-length idle sleep would cool further, because it omits the CPU heat the
# compile puts into the package. `E121_PREP_FLOOR_SECONDS` then pads any
# residual so every leg reaches its timed run after the same wall time.
prep_started="$(date +%s)"
touch -- "${HEADER}" "${TWIN}"

senpai/rebuild-and-assert-worker.sh "${assert_args[@]}" \
  > "${out}/worker-assert-pre.txt" 2>&1 || {
  echo "e121_e2e_leg: worker assert failed before ${tag}" >&2
  tail -30 "${out}/worker-assert-pre.txt" >&2
  exit 5
}

prep_build_seconds=$(( $(date +%s) - prep_started ))
prep_floor="${E121_PREP_FLOOR_SECONDS:-0}"
prep_idle_seconds=0
if ((prep_build_seconds < prep_floor)); then
  prep_idle_seconds=$((prep_floor - prep_build_seconds))
  echo "e121_e2e_leg: ${tag} padding prep ${prep_build_seconds}s -> ${prep_floor}s"
  sleep "${prep_idle_seconds}"
fi
prep_seconds=$((prep_build_seconds + prep_idle_seconds))
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
  echo "e121_prep_build_seconds=${prep_build_seconds}"
  echo "e121_prep_idle_seconds=${prep_idle_seconds}"
  echo "e121_prep_seconds=${prep_seconds}"
  echo "e121_prep_floor_seconds=${prep_floor}"
} >> "${out}/meta.txt"

if ((post_rc != 0)) || [[ "${post_worker}" != "${pre_worker}" ]]; then
  echo "e121_e2e_leg: ${tag} worker changed or failed its post-assert; discarding" >&2
  status=7
fi

echo "status=${status}" >> "${out}/meta.txt"
exit "${status}"
