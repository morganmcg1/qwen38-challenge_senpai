#!/usr/bin/env bash
# Does the lock my timing harness takes actually exclude a peer student?
#
# Two ways this fails silently, both real (advisor, PR 53):
#   1. `local_run_lock_path` is anchored at $HOME, and every role has its own,
#      so the default shards one lock into one-per-student and both peers
#      believe they hold the machine. MLXFAST_LOCAL_RUN_LOCK_DIR fixes it.
#   2. `acquire_local_run_lock` opens with `local_run_guard_enabled || return 0`.
#      A harness that lifts the guard functions but not that predicate gets
#      command-not-found (127), `|| return 0` fires, and the guard returns
#      success having done nothing.
#
# So: assert the predicate exists and is true in run-qmv-curve.sh's own
# extraction, assert the lock lands in the SHARED directory, and assert a second
# holder is actually refused. A lock that cannot refuse is not a lock.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

fail() { echo "e49_lock_selftest: FAIL: $*" >&2; exit 1; }

# 1. the predicate the harness relies on is defined by the harness itself
grep -q '^local_run_guard_enabled() {' research/run-qmv-curve.sh \
  || fail "run-qmv-curve.sh does not define local_run_guard_enabled; the guard would fail open"

# 2. same extraction as research/run-qmv-curve.sh, verbatim
LOCAL_RUN_LOCK_OWNED=""
local_run_guard_enabled() {
  [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]
}
eval "$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
)"
local_run_guard_enabled || fail "the guard predicate is false; the harness would run unlocked"

lock_path="$(local_run_lock_path)"
case "${lock_path}" in
  "${MLXFAST_LOCAL_RUN_LOCK_DIR%/}"/*) ;;
  *) fail "lock path ${lock_path} ignores MLXFAST_LOCAL_RUN_LOCK_DIR" ;;
esac

[[ -e "${lock_path}" ]] && fail "lock ${lock_path} is already held; a peer may be timing right now"

acquire_local_run_lock
[[ -d "${lock_path}" ]] || fail "acquire_local_run_lock returned success without creating ${lock_path}"
[[ "$(cat "${lock_path}/pid")" == "$$" ]] || fail "lock does not record this pid"

# 3. a second holder must be refused -- the answer the gate has to be able to give
set +e
bash -c '
  set -euo pipefail
  cd "'"${repo_root}"'"
  LOCAL_RUN_LOCK_OWNED=""
  local_run_guard_enabled() { [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]; }
  eval "$(
    awk "/^local_run_lock_path\(\) \{/,/^\}/" benchmark.sh
    awk "/^acquire_local_run_lock\(\) \{/,/^\}/" benchmark.sh
  )"
  acquire_local_run_lock
' >/dev/null 2>&1
second_rc=$?
set -e
[[ "${second_rc}" -eq 1 ]] || fail "a second holder was NOT refused (rc=${second_rc})"

release_local_run_lock
[[ -e "${lock_path}" ]] && fail "release did not remove ${lock_path}"

echo "e49_lock_selftest: PASS  shared lock ${lock_path}; second holder refused (rc=1)"
