#!/usr/bin/env bash
# Prove that an E59 timed leg really runs at the ranked command-buffer geometry.
#
#   research/e59_geometry_proof.sh [--skip-dose]
#
# This replaces the withdrawn `worker_low_memory_notices=0` grep. That grep
# could never fail: the notice goes to worker stderr and `mtp-timed` installs a
# swallowing emitter, so a zero count carried no information.
#
# Part A -- profile resolution, four direct worker launches, seconds each.
#   `applyQwenMTPStartupMemoryProfile` runs at QwenRuntimeMTPWorker.swift:133,
#   immediately after standard-IO isolation and before any weight load, so all
#   four probes point `--weights` at a path that does not exist. The launch
#   fails either way; what differs is HOW, and that difference is the proof.
#
#     profile   expected                                             proves
#     bogus     trap at the `preconditionFailure` in `resolve`       the parser is live
#     low       low-memory notice on stderr, then a weight error     the notice can fire
#     full      no notice, then a weight error                       `case "full"` returns early
#     unset     low-memory notice on stderr                          48 GiB defaults to LOW
#
#   The fourth probe is the one that makes the export load-bearing: without
#   `=full` this host resolves to the low profile and the worker force-sets
#   128 MiB / 64 ops over anything the parent exported.
#
# Part B -- geometry dose response, two short legs, expensive.
#   Nothing in `Sources/` counts dispatches any more, so the only honest test
#   that the exported cap reaches MLX is a dose response. MLX latches
#   `MLX_MAX_OPS_PER_BUFFER` once and commits a command buffer every N
#   operations, so a round issuing roughly a thousand operations must be
#   materially slower at 8 than at 50.
#
#   The doses run 8 FIRST, then 50. Monotone warming makes later legs faster,
#   so the order works AGAINST the prediction: leg 1 is the colder leg and must
#   still be the slower one.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

skip_dose=0
while (($#)); do
  case "$1" in
    --skip-dose) skip_dose=1; shift ;;
    *) echo "e59_geometry_proof: unknown argument $1" >&2; exit 2 ;;
  esac
done

out_json="research/e59-artifacts/e59-geometry-proof.json"
work="${repo_root}/.mlxfast-private/e59-geometry-proof"
mkdir -p "${work}" research/e59-artifacts

worker="${repo_root}/.build-worker/release/mlxfast-runtime-worker"
head_dir="${E59_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
missing_weights="${work}/weights-that-do-not-exist"

if [[ ! -x "${worker}" ]]; then
  echo "e59_geometry_proof: ${worker} is not built" >&2
  exit 2
fi

echo "e59_geometry_proof: === $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "e59_geometry_proof: worker=$(shasum -a 256 "${worker}" | cut -d' ' -f1)"

probe() {
  local name="$1" profile="$2"
  local log="${work}/profile-${name}.log"
  echo "e59_geometry_proof: --- profile probe ${name} (${profile}) ---"
  if [[ "${profile}" == "unset" ]]; then
    unset DARKBLOOM_STARTUP_MEMORY_PROFILE
  else
    export DARKBLOOM_STARTUP_MEMORY_PROFILE="${profile}"
  fi
  "${worker}" mtp-runtime-worker \
    --weights "${missing_weights}" \
    --mtp-head "${head_dir}" \
    < /dev/null > "${log}" 2>&1
  local rc=$?
  local notice precond
  notice="$(grep -c 'low-memory startup profile engaged' "${log}")"
  precond="$(grep -c 'must be auto, full, or low' "${log}")"
  echo "e59_geometry_proof: ${name} rc=${rc} notice=${notice} precondition=${precond}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${profile}" "${rc}" "${notice}" "${precond}" \
    >> "${work}/profile-probes.tsv"
}

rm -f "${work}/profile-probes.tsv"
probe bogus bogus
probe low low
probe full full
probe auto unset
export DARKBLOOM_STARTUP_MEMORY_PROFILE=full

dose_ran=0
if ((skip_dose == 0)); then
  # Short and `--hot`: this is an instrument check, not a scored comparison, and
  # the 8-vs-50 effect is expected to dwarf any thermal term. The ungated mode
  # is recorded in each leg's own meta.txt.
  export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
  export E59_BASE_SHA="${E59_BASE_SHA:-$(git rev-parse origin/senpai/qwen38-mtp-r1)}"
  export E59_E2E_ROOT="${E59_E2E_ROOT:-${repo_root}/.mlxfast-private/e59-e2e}"
  export E59_LEG_STAGE=""
  for dose in 8 50; do
    echo "e59_geometry_proof: --- dose leg ops=${dose} $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    research/e59_e2e_run.sh shipped "e59-geom-ops${dose}" \
      --tokens 64 --hot --ops "${dose}"
    rc=$?
    if ((rc != 0)); then
      echo "e59_geometry_proof: dose leg ops=${dose} exited ${rc}" >&2
      exit "${rc}"
    fi
  done
  dose_ran=1
fi

E59_GEOM_WORK="${work}" E59_GEOM_DOSE_RAN="${dose_ran}" \
  python3 research/e59_geometry_proof.py --out "${out_json}"
rc=$?
echo "e59_geometry_proof: === done rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit "${rc}"
