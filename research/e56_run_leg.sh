#!/usr/bin/env bash
# One E56 timed leg against a prebuilt arm.
#
#   research/e56_run_leg.sh ARM TAG [--tokens N]
#
# ARM is `base` (the campaign base's scalar-price schedule) or `sched` (this
# branch's stream-aware price). research/e56_build_arms.sh has already built
# both worker binaries and published them outside the checkout, so this script
# only selects one of them. It never edits, checks out, or stashes anything:
# the work tree stays on HEAD for the whole session, which removes the failure
# mode where a hard kill during a `base` leg leaves base bytes on the branch.
set -uo pipefail

arm="${1:?usage: e56_run_leg.sh ARM TAG [--tokens N]}"
tag="${2:?usage: e56_run_leg.sh ARM TAG [--tokens N]}"
shift 2

tokens=256
depth=""
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    # The parent's offered per-round draft ceiling. The base walk wants depth 8
    # at this fixture's acceptance, so an offer of k pins every round to verify
    # width k+1 and turns this leg into one point of a fixed-width cost curve.
    --depth) depth="$2"; shift 2 ;;
    *) echo "e56_run_leg: unknown argument $1" >&2; exit 2 ;;
  esac
done

case "${arm}" in
  base|sched) ;;
  *) echo "e56_run_leg: unknown arm ${arm}" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# One lock for the machine, not one per student $HOME (advisor, PR 53).
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

readonly SCHEDULE_FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
arm_dir="${E56_ARM_DIR:-${HOME}/e56-arms}/${arm}"
worker="${arm_dir}/mlxfast-runtime-worker"
metallib="${arm_dir}/mlx.metallib"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e56_run_leg: work tree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi
if [[ ! -x "${worker}" || ! -s "${metallib}" ]]; then
  echo "e56_run_leg: arm ${arm} is not built; run research/e56_build_arms.sh first" >&2
  exit 1
fi

# benchmark.sh and the trusted CLI both resolve the scored binary through this
# variable, and the CLI verifies the metallib beside it, so the sidecar has to
# be named too.
export MLXFAST_RUNTIME_WORKER_EXECUTABLE="${worker}"
export MLXFAST_MLX_METALLIB="${metallib}"
if [[ -n "${depth}" ]]; then
  export MLXFAST_QWEN_MTP_DEPTH="${depth}"
fi

# Same search order as benchmark.sh's find_macmon, so this leg records the
# temperature from the same reader the cool gate itself used. setup.sh installs
# macmon into ${HOME}/bin, which is not on PATH here.
find_macmon() {
  local candidate
  if [[ -n "${MLXFAST_MACMON_BIN:-}" && -x "${MLXFAST_MACMON_BIN}" ]]; then
    printf '%s\n' "${MLXFAST_MACMON_BIN}"
    return 0
  fi
  if candidate="$(command -v macmon 2>/dev/null)"; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  for candidate in /opt/homebrew/bin/macmon /usr/local/bin/macmon "${HOME}/bin/macmon"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

gpu_temp() {
  local bin
  bin="$(find_macmon)" || return 0
  "${bin}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

echo "=== e56 leg ${tag} (${arm}) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cat "${arm_dir}/arm.txt"

out="research/out/${tag}"
entry_temp="$(gpu_temp)"
research/run-arm.sh "${tag}" --trace --tokens "${tokens}"
status=$?
exit_temp="$(gpu_temp)"

# benchmark.sh reports the cool gate on stderr, which run-arm.sh captures into
# the trace, so the gate's own words are the disclosure -- not a claim of mine.
gate_lines="$(grep -c 'GPU cool-down gate passed' "${out}/trace.txt" 2>/dev/null)" || gate_lines="${gate_lines:-0}"
gate_skipped="$(grep -c 'skipping the GPU cool-down gate' "${out}/trace.txt" 2>/dev/null)" || gate_skipped="${gate_skipped:-0}"

{
  echo "e56_arm=${arm}"
  echo "e56_tag=${tag}"
  echo "offered_depth=${MLXFAST_QWEN_MTP_DEPTH:-8}"
  echo "e56_head_sha=$(git rev-parse HEAD)"
  echo "checkout_schedule_blob=$(git hash-object "${SCHEDULE_FILE}")"
  echo "worker_path=${worker}"
  echo "worker_sha256=$(shasum -a 256 "${worker}" | cut -d' ' -f1)"
  echo "metallib_sha256=$(shasum -a 256 "${metallib}" | cut -d' ' -f1)"
  grep -E '^(schedule_blob|built)=' "${arm_dir}/arm.txt" 2>/dev/null | sed 's/^/arm_/'
  echo "entry_gpu_temp_c=${entry_temp:-unavailable}"
  echo "exit_gpu_temp_c=${exit_temp:-unavailable}"
  echo "cool_gate_passes=${gate_lines}"
  echo "cool_gate_skips=${gate_skipped}"
  echo "cool_gate_passed_real_gate=$([[ "${gate_lines}" -gt 0 && "${gate_skipped}" -eq 0 ]] && echo true || echo false)"
} >> "${out}/meta.txt" 2>/dev/null

# Log while the session is still running. A leg that logs only at session end
# is lost if the workspace is retagged mid-session (thorfinn, 15:02Z).
python3 research/e56_log_leg.py --tag "${tag}" --arm "${arm}" || true

exit "${status}"
