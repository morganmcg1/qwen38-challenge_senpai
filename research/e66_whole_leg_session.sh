#!/usr/bin/env bash
# Rung 3: run the E66 arms back to back inside ONE session.
#
#   research/e66_whole_leg_session.sh ARM:TAG:LEGS [ARM:TAG:LEGS ...]
#
# Position balance is a property of the order the arms run in, so the caller
# passes the order explicitly and every arm is contiguous in time within one job.
# The intended order is
#   b_t6:warm:1 a_neither:a1:2 b_t6:b1:2 c_t55_t6:c1:2 \
#   c_t55_t6:c2:2 b_t6:b2:2 a_neither:a2:2
# which gives leg positions A={1,2,11,12}, B={3,4,9,10}, C={5,6,7,8}: position
# sum 26 and mean 6.5 for all three arms.
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
readonly BASE_SHA="${E66_BASE_SHA:-45b4f3a800f879e3579ca27ef0b1c0ef40e4473d}"
readonly ARMS_MODULE="${E66_ARMS_MODULE:-research/e66_arms.py}"
readonly MANIFEST="${E66_MANIFEST_DIR:-${repo_root}/.mlxfast-private/e66/arms}"
# Rung 2 reuses this patch/commit/unwind driver with the ledger runner, which
# needs the same per-arm build and the same guarantee that the branch's scored
# surface returns to base bytes between arms.
readonly LEG_RUNNER="${E66_LEG_RUNNER:-research/e66-run.sh}"
readonly WANDB_LOGGER="${E66_WANDB_LOGGER:-research/e66_wandb_leg.py}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-e66-composition-certification}"

legs=("$@")
[[ ${#legs[@]} -gt 0 ]] || {
  echo "e66_whole_leg_session: no ARM:TAG:LEGS specs given" >&2; exit 2; }

pre_patch_sha="$(git rev-parse HEAD)"
transient_sha=""

# `reset --mixed` + `checkout --` rather than `reset --hard`: the branch pointer
# and the two kernel files go back, and any unrelated edit in the worktree
# survives.
unwind() {
  if [[ -n "${transient_sha}" ]]; then
    if [[ "$(git rev-parse HEAD)" == "${transient_sha}" ]]; then
      git reset -q "${pre_patch_sha}"
    else
      echo "e66_whole_leg_session: HEAD moved during an arm; restoring files only" >&2
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
  echo "e66_whole_leg_session: worktree is dirty; refusing to run over uncommitted work" >&2
  exit 1
fi
if ! git diff --quiet "${BASE_SHA}" -- "${SCORED_FILES[@]}"; then
  echo "e66_whole_leg_session: scored twins differ from the campaign base ${BASE_SHA}; refusing to stack patches" >&2
  exit 1
fi

mkdir -p "${MANIFEST}"
status=0
for spec in "${legs[@]}"; do
  arm="${spec%%:*}"
  rest="${spec#*:}"
  tag="${rest%%:*}"
  count="${rest##*:}"
  [[ "${count}" =~ ^[0-9]+$ ]] || {
    echo "e66_whole_leg_session: '${spec}' has no leg count" >&2; status=2; break; }
  date -u "+e66_whole_leg_session: === %Y-%m-%dT%H:%M:%SZ arm ${tag} (${arm}, ${count} leg(s)) ==="

  python3 "${ARMS_MODULE}" "${arm}" --out "${MANIFEST}/${tag}-arm.json" || {
    echo "e66_whole_leg_session: ${tag}: arm patch failed" >&2; status=1; break; }
  git add -- "${SCORED_FILES[@]}"
  # --allow-empty: arm b_t6 is the tip unmodified, so it has no diff. Every arm
  # still gets a commit, which keeps the unwind uniform and lets e42-run.sh
  # record dirty=0 for treated and control arms alike.
  git commit -q --allow-empty -m "E66 arm ${tag}: TRANSIENT ${arm} bytes under measurement

Unwound to ${pre_patch_sha} when the session exits, including on a crash, so the
branch's scored surface stays byte-identical to ${BASE_SHA}. This commit exists
only so the bytes the compiler saw are reachable while the arm runs."
  transient_sha="$(git rev-parse HEAD)"

  "${LEG_RUNNER}" "${tag}" --legs "${count}"
  rc=$?
  echo "e66_whole_leg_session: ${tag} (${arm}) exited ${rc}" >&2

  unwind

  # Log this leg group to W&B before the next arm starts, so a session that dies
  # part-way still leaves every completed leg on the board. It runs after the
  # unwind and outside the timed window, and `wandb/` is gitignored, so it can
  # neither perturb the measurement nor dirty the worktree. A logging failure
  # never fails a leg that already measured cleanly.
  if [[ "${E66_WANDB:-1}" != "0" && ${rc} -eq 0 ]]; then
    discarded=()
    [[ "${tag}" == warm* ]] && discarded=(--discarded)
    # bash 3.2 treats "${arr[@]}" on an EMPTY array as an unbound variable under
    # `set -u`, so the ${arr[@]+...} form is required.
    python3 "${WANDB_LOGGER}" --tag "${tag}" --arm "${arm}" \
      --group "${WANDB_RUN_GROUP}" ${discarded[@]+"${discarded[@]}"} \
      || echo "e66_whole_leg_session: ${tag}: W&B logging failed" >&2
  fi

  ((rc == 0)) || { status="${rc}"; break; }
done

date -u "+e66_whole_leg_session: === %Y-%m-%dT%H:%M:%SZ session complete (status ${status}) ==="
exit "${status}"
