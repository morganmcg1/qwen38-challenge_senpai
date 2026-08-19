#!/usr/bin/env bash
# One E56 timed leg: select the arm's schedule, rebuild, measure, log, unwind.
#
#   research/e56_run_leg.sh ARM TAG [--tokens N]
#
# ARM is `base` (the campaign base's scalar-price schedule) or `sched` (this
# branch's stream-aware price). The only file that differs between the two arms
# is the block session, so a `base` leg checks that one file out at the base
# commit, rebuilds, measures, and puts it back on every exit path -- including
# a crash or a job timeout -- so the branch's scored surface is never left
# holding another arm's bytes.
set -uo pipefail

arm="${1:?usage: e56_run_leg.sh ARM TAG [--tokens N]}"
tag="${2:?usage: e56_run_leg.sh ARM TAG [--tokens N]}"
shift 2

tokens=256
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    *) echo "e56_run_leg: unknown argument $1" >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# One lock for the machine, not one per student $HOME (advisor, PR 53).
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

readonly SCHEDULE_FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
readonly E56_BASE_SHA="${E56_BASE_SHA:-a2c3dbc497fd76b3e4f99c529a3eb5e8b2090abf}"

head_sha="$(git rev-parse HEAD)"
patched=0

unwind() {
  if ((patched)); then
    git checkout -q "${head_sha}" -- "${SCHEDULE_FILE}" || true
  fi
}
trap unwind EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e56_run_leg: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi

case "${arm}" in
  base)
    git checkout -q "${E56_BASE_SHA}" -- "${SCHEDULE_FILE}" || exit 1
    patched=1
    ;;
  sched) ;;
  *) echo "e56_run_leg: unknown arm ${arm}" >&2; exit 2 ;;
esac

gpu_temp() {
  local bin
  bin="$(command -v macmon 2>/dev/null)" || return 0
  "${bin}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

echo "=== e56 leg ${tag} (${arm}) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
research/rebuild.sh || exit 1

out="research/out/${tag}"
entry_temp="$(gpu_temp)"
research/run-arm.sh "${tag}" --trace --tokens "${tokens}"
status=$?
exit_temp="$(gpu_temp)"

{
  echo "e56_arm=${arm}"
  echo "e56_tag=${tag}"
  echo "e56_base_sha=${E56_BASE_SHA}"
  echo "e56_head_sha=${head_sha}"
  echo "schedule_file_sha=$(git hash-object "${SCHEDULE_FILE}")"
  echo "entry_gpu_temp_c=${entry_temp:-unavailable}"
  echo "exit_gpu_temp_c=${exit_temp:-unavailable}"
} >> "${out}/meta.txt" 2>/dev/null

# Log while the session is still running. A leg that logs only at session end
# is lost if the workspace is retagged mid-session (thorfinn, 15:02Z).
python3 research/e56_log_leg.py --tag "${tag}" --arm "${arm}" || true

exit "${status}"
