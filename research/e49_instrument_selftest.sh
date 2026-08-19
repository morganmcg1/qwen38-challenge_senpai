#!/usr/bin/env bash
# Prove both contention instruments can give BOTH answers, before any leg.
#
# The GPU half deliberately puts Metal load on the device, so it holds the
# shared run lock while it does: a self-test that corrupts a peer student's
# timing would be worse than no self-test.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

bash research/e49_lock_selftest.sh

LOCAL_RUN_LOCK_OWNED=""
local_run_guard_enabled() {
  [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]
}
eval "$(
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
)"
trap release_local_run_lock EXIT
acquire_local_run_lock

out=".mlxfast-private/e49-legs"
mkdir -p "${out}"
python3 research/e49_gpu_gate.py --selftest | tee "${out}/gpu-gate-selftest.json"
