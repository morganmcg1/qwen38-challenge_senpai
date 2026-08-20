#!/usr/bin/env bash
# Rung 3: run the E61 whole-leg arms back to back inside ONE session.
#
#   research/e61_whole_leg_session.sh ARM:TAG [ARM:TAG ...] [-- e61-run.sh args]
#
# Counterbalancing is a property of the order the arms run in, so the caller
# passes the order explicitly and every arm is contiguous in time within one
# job. The intended order is `shipped:base t6:m6 shipped:base2`.
#
# Each arm is patched into the two scored twins, committed transiently so the
# bytes the compiler saw are reachable and e42-run.sh's dirty check passes, run,
# and then unwound on every exit path including a crash. The branch's scored
# surface is byte-identical to the campaign base between arms and after the
# session.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

readonly SCORED_FILES=(
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
)
readonly BASE_SHA="${E61_BASE_SHA:-d2139c924c7a7d98ca6026eea63867c2776abbca}"
readonly ARMS_MODULE="${E61_ARMS_MODULE:-research/e61_arms.py}"
readonly MANIFEST="${E61_MANIFEST_DIR:-${repo_root}/.mlxfast-private/e61/arms}"
# Rung 2 reuses this patch/commit/unwind driver with the ledger runner, which
# needs the same per-arm build and the same guarantee that the branch's scored
# surface returns to base bytes between arms.
readonly LEG_RUNNER="${E61_LEG_RUNNER:-research/e61-run.sh}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e61-single-stream-qmv-m6}"

legs=()
while [[ $# -gt 0 && "${1}" != "--" ]]; do
  legs+=("${1}")
  shift
done
[[ "${1:-}" == "--" ]] && shift
[[ ${#legs[@]} -gt 0 ]] || {
  echo "e61_whole_leg_session: no ARM:TAG legs given" >&2; exit 2; }

pre_patch_sha="$(git rev-parse HEAD)"
transient_sha=""

# `reset --mixed` + `checkout --` rather than `reset --hard`: the branch pointer
# and the two kernel files go back, and any unrelated edit in the worktree
# survives. The file restore names ${pre_patch_sha} explicitly, so if something
# commits on top of the transient commit while an arm runs, that work is left
# alone while the kernel files still return to base bytes.
unwind() {
  if [[ -n "${transient_sha}" ]]; then
    if [[ "$(git rev-parse HEAD)" == "${transient_sha}" ]]; then
      git reset -q "${pre_patch_sha}"
    else
      echo "e61_whole_leg_session: HEAD moved during an arm; restoring files only" >&2
    fi
    transient_sha=""
  fi
  git checkout -q "${pre_patch_sha}" -- "${SCORED_FILES[@]}" 2>/dev/null || true
}
trap unwind EXIT
# A bare EXIT trap is not guaranteed to run when the job is killed at its
# timeout, and a killed arm is exactly when the unwind matters most.
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e61_whole_leg_session: worktree is dirty; refusing to run over uncommitted work" >&2
  exit 1
fi
if ! git diff --quiet "${BASE_SHA}" -- "${SCORED_FILES[@]}"; then
  echo "e61_whole_leg_session: scored twins differ from the campaign base ${BASE_SHA}; refusing to stack patches" >&2
  exit 1
fi

mkdir -p "${MANIFEST}"
status=0
for spec in "${legs[@]}"; do
  arm="${spec%%:*}"
  tag="${spec##*:}"
  date -u "+e61_whole_leg_session: === %Y-%m-%dT%H:%M:%SZ arm ${tag} (${arm}) ==="

  python3 "${ARMS_MODULE}" "${arm}" --out "${MANIFEST}/${tag}-arm.json" || {
    echo "e61_whole_leg_session: ${tag}: arm patch failed" >&2; status=1; break; }
  git add -- "${SCORED_FILES[@]}"
  # --allow-empty: the `shipped` control arm is the tip unmodified, so it has no
  # diff. Every arm still gets a commit, which keeps the unwind uniform and lets
  # e42-run.sh record dirty=0 for treated and control arms alike.
  git commit -q --allow-empty -m "E61 arm ${tag}: TRANSIENT ${arm} bytes under measurement

Unwound to ${pre_patch_sha} when the session exits, including on a crash, so the
branch's scored surface stays byte-identical to ${BASE_SHA}. This commit exists
only so the bytes the compiler saw are reachable while the arm runs."
  transient_sha="$(git rev-parse HEAD)"

  "${LEG_RUNNER}" "${tag}" "$@"
  rc=$?
  echo "e61_whole_leg_session: ${tag} (${arm}) exited ${rc}" >&2

  unwind

  # Log this leg to W&B before the next arm starts, so a session that dies
  # part-way still leaves every completed leg on the board. It runs after the
  # unwind and outside the timed window, and `wandb/` is gitignored, so it can
  # neither perturb the measurement nor dirty the worktree. A logging failure
  # never fails the leg that already measured cleanly.
  if [[ "${E61_WANDB:-1}" != "0" && ${rc} -eq 0 ]]; then
    discarded=()
    [[ "${tag}" == warm* ]] && discarded=(--discarded)
    python3 research/e61_wandb_leg.py --tag "${tag}" --arm "${arm}" \
      --group "${WANDB_RUN_GROUP}" "${discarded[@]}" \
      || echo "e61_whole_leg_session: ${tag}: W&B logging failed" >&2
  fi

  ((rc == 0)) || { status="${rc}"; break; }
done

date -u "+e61_whole_leg_session: === %Y-%m-%dT%H:%M:%SZ session complete (status ${status}) ==="
exit "${status}"
