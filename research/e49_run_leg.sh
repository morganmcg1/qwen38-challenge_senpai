#!/usr/bin/env bash
# One E49 timed leg: patch -> commit -> observe the GPU -> measure -> unwind.
#
#   research/e49_run_leg.sh ARM TAG [--widths L] [--reps N] [--inner N]
#
# The arm patch is committed while it is being compiled and that commit is
# unwound on every exit path, including a crash, so the worktree is never dirty
# between legs and the branch's scored surface stays byte-identical to the
# campaign base.
set -euo pipefail

arm="${1:?usage: e49_run_leg.sh ARM TAG}"
tag="${2:?usage: e49_run_leg.sh ARM TAG}"
shift 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# The per-role $HOME default shards this lock into one lock per student, which
# is worse than no lock because both peers believe they hold the machine
# (advisor, PR 53). askeladd exports the same directory on PR 52.
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

readonly SCORED_FILES=(
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
)

readonly E49_BASE_SHA="${E49_BASE_SHA:-fb0a09d3912477d94ed631bdb90fd04172d7b4cf}"
# Later experiments reuse this runner with their own arm definitions. The module
# must expose the same CLI: `MODULE ARM --out JSON`, patching both scored files.
readonly LEG_ARMS_MODULE="${LEG_ARMS_MODULE:-research/e49_arms.py}"

pre_patch_sha="$(git rev-parse HEAD)"
transient_sha=""

# `reset --mixed` + `checkout --` rather than `reset --hard`: the branch pointer
# and the two kernel files go back, and any unrelated edit in the worktree
# survives. A hard reset here silently ate an unrelated file once already.
#
# The file restore names ${pre_patch_sha} explicitly instead of trusting HEAD.
# If something commits on top of the transient commit while the leg runs, the
# branch is left alone -- unwinding it would discard that work -- but the
# kernel files still return to base bytes rather than keeping the arm patch.
unwind() {
  if [[ -n "${transient_sha}" ]]; then
    if [[ "$(git rev-parse HEAD)" == "${transient_sha}" ]]; then
      git reset -q "${pre_patch_sha}"
    else
      echo "e49_run_leg: HEAD moved during the leg; restoring files only, leaving the branch alone" >&2
    fi
  fi
  git checkout -q "${pre_patch_sha}" -- "${SCORED_FILES[@]}" 2>/dev/null || true
}
trap unwind EXIT
# A bare EXIT trap is not guaranteed to run when the session is killed at the
# job timeout, and a killed leg is exactly when the unwind matters most.
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e49_run_leg: worktree is dirty; refusing to run a leg over uncommitted work" >&2
  exit 1
fi
if ! git diff --quiet "${E49_BASE_SHA}" -- "${SCORED_FILES[@]}"; then
  echo "e49_run_leg: scored kernel files differ from the campaign base ${E49_BASE_SHA}; refusing to stack patches" >&2
  exit 1
fi

out_dir="${repo_root}/.mlxfast-private/qmv-curve/${tag}"
mkdir -p "${out_dir}"
manifest="${LEG_MANIFEST_DIR:-${repo_root}/.mlxfast-private/e49-legs}"
mkdir -p "${manifest}"

# --- who else is on this GPU --------------------------------------------------
gate_json="${manifest}/${tag}-gpu-gate.json"
set +e
python3 research/e49_gpu_gate.py --samples 5 --out "${gate_json}"
gate_rc=$?
set -e
case "${gate_rc}" in
  0) echo "e49_run_leg: GPU gate idle" >&2 ;;
  1) echo "e49_run_leg: GPU gate reports BUSY; another process is using the GPU. Not timing." >&2
     exit 3 ;;
  *) echo "e49_run_leg: GPU utilization counter unavailable; refusing to time blind." >&2
     exit 4 ;;
esac

# --- patch --------------------------------------------------------------------
python3 "${LEG_ARMS_MODULE}" "${arm}" --out "${manifest}/${tag}-arm.json"
git add -- "${SCORED_FILES[@]}"
# --allow-empty: the `shipped` control arm is the tip unmodified, so it has no
# diff to commit. Every leg still gets a commit, which keeps the unwind and the
# leg record uniform across treated and control arms.
git commit -q --allow-empty -m "E49 leg ${tag}: TRANSIENT ${arm} arm bytes under measurement

Unwound to ${pre_patch_sha} when the leg exits, including on a crash, so the
branch's scored surface stays byte-identical to ${E49_BASE_SHA}. This commit
exists only so the bytes the compiler saw are reachable while the leg runs."
transient_sha="$(git rev-parse HEAD)"

# --- measure ------------------------------------------------------------------
# run-qmv-curve.sh writes vendored.json before it summarizes, and the summary
# is a report over a full 1..10 sweep. A partial sweep therefore kills the
# driver AFTER a good measurement, and `set -e` would then discard the rest of
# a counterbalanced session. Gate on the measurement artifact instead of on the
# driver's exit code, and carry the code through to the leg record.
set +e
research/run-qmv-curve.sh "${tag}" "${pre_patch_sha}" --skip-stock "$@"
curve_rc=$?
set -e
if ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))["shapes"]' \
     "${out_dir}/vendored.json" >/dev/null 2>&1; then
  echo "e49_run_leg: ${tag} produced no readable vendored.json (rc=${curve_rc}); failing the leg" >&2
  exit 5
fi
[[ "${curve_rc}" -eq 0 ]] \
  || echo "e49_run_leg: ${tag} measured, but run-qmv-curve.sh exited ${curve_rc} (post-measurement stage)" >&2

# --- record what actually ran -------------------------------------------------
CURVE_RC="${curve_rc}" MEASURED_COMMIT="${transient_sha}" BRANCH_COMMIT="${pre_patch_sha}" \
  python3 - "${tag}" "${arm}" "${out_dir}" "${manifest}" <<'PY'
import hashlib, json, os, pathlib, sys

tag, arm, out_dir, manifest = sys.argv[1:5]
files = ["Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
         "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"]
leg = {
    "tag": tag,
    "arm": arm,
    "run_qmv_curve_rc": int(os.environ["CURVE_RC"]),
    "measured_commit_unwound": os.environ["MEASURED_COMMIT"],
    "branch_commit": os.environ["BRANCH_COMMIT"],
    "sources_as_measured": {
        f: hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in files
    },
    "gpu_gate": json.loads(pathlib.Path(manifest, tag + "-gpu-gate.json").read_text()),
    "arm_patch": json.loads(pathlib.Path(manifest, tag + "-arm.json").read_text()),
    "identity": pathlib.Path(out_dir, "identity.txt").read_text().splitlines(),
}
pathlib.Path(manifest, tag + "-leg.json").write_text(
    json.dumps(leg, indent=2, sort_keys=True))
print("e49_run_leg: wrote %s/%s-leg.json" % (manifest, tag))
PY

unwind
transient_sha=""
echo "e49_run_leg: ${tag} (${arm}) done; unwound to ${pre_patch_sha}" >&2
