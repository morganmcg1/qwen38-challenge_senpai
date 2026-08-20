#!/usr/bin/env bash
# Prove that an E59 timed leg really runs at the ranked command-buffer geometry.
#
#   research/e59_geometry_proof.sh [--skip-dose] [--skip-probes] [--extreme]
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
#     bogus     exit 133, the SIGTRAP from `preconditionFailure`     the parser is live
#     low       low-memory notice on stderr, then a weight error     the notice can fire
#     full      no notice, then a weight error                       `case "full"` returns early
#     unset     low-memory notice on stderr                          48 GiB defaults to LOW
#
#   The release build strips `preconditionFailure` text, so the bogus probe is
#   read from its exit code, never from a message.
#
#   The fourth probe is the one that makes the export load-bearing: without
#   `=full` this host resolves to the low profile and the worker force-sets
#   128 MiB / 64 ops over anything the parent exported.
#
# Part B -- geometry dose response, two short legs, expensive.
#   Nothing in `Sources/` counts dispatches any more, so the only honest test
#   that the exported cap reaches MLX is a dose response. MLX latches
#   `MLX_MAX_OPS_PER_BUFFER` once and commits a command buffer whenever
#   `buffer_ops_` passes the cap.
#
#   The default `--extreme`-free run uses caps 8 and 50. That pair measured a
#   null, which two different stories explain: the export never arrives, or
#   commits are free because decode sits at 97 % of the DRAM roofline. Run
#   `--extreme` to separate them. A cap of 1 commits about every second
#   operation, which no amount of GPU saturation can hide, so a null there
#   means the export never reaches MLX.
#
#   Each pair runs the TIGHTER cap first. Monotone warming makes later legs
#   faster, so the order works AGAINST the prediction: leg 1 is the colder leg
#   and must still be the slower one.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

skip_dose=0
skip_probes=0
dose_set="moderate"
while (($#)); do
  case "$1" in
    --skip-dose) skip_dose=1; shift ;;
    --skip-probes) skip_probes=1; shift ;;
    --extreme) dose_set="extreme"; shift ;;
    *) echo "e59_geometry_proof: unknown argument $1" >&2; exit 2 ;;
  esac
done

# tag suffix -> exported MLX_MAX_OPS_PER_BUFFER. The tighter cap always runs
# first so that monotone warming works against the prediction.
case "${dose_set}" in
  moderate) dose_tags=(8 50); dose_ops=(8 50) ;;
  extreme) dose_tags=(1 50x); dose_ops=(1 50) ;;
esac

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

if ((skip_probes == 0)); then
  rm -f "${work}/profile-probes.tsv"
  probe bogus bogus
  probe low low
  probe full full
  probe auto unset
fi
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
  for i in "${!dose_tags[@]}"; do
    tag="${dose_tags[$i]}"
    ops="${dose_ops[$i]}"
    echo "e59_geometry_proof: --- dose leg ops=${ops} tag=${tag} $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
    research/e59_e2e_run.sh shipped "e59-geom-ops${tag}" \
      --tokens 64 --hot --ops "${ops}"
    rc=$?
    if ((rc != 0)); then
      echo "e59_geometry_proof: dose leg ops=${ops} exited ${rc}" >&2
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
