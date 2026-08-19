#!/usr/bin/env bash
# Research-only launcher for the E57 rung-1 dispatch counter.
#
# run_job takes an argv list with no environment field, so the switches the
# probe reads have to be set inside a script. The probe runs in-process, so the
# MLXFAST_ prefix is fine here: the runtime-worker environment allowlist that
# forced MLX_E57_* on the decode instrument does not apply.
#
# usage:
#   research/e57_run_dispatch_count.sh            # the legal cells
#   research/e57_run_dispatch_count.sh --throw    # + the illegal cell
#
# --throw adds one unchunked qL=6 call at kL=1030. sdpa_vector_2pass asks for
# 32 * 6 * 6 = 1152 threads per threadgroup there, utils.h throws, and an
# uncaught C++ exception ends the process. A non-zero exit is therefore the
# EXPECTED result of --throw, so this script reports the status instead of
# failing on it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

throwing=0
[[ "${1:-}" == "--throw" ]] && throwing=1

out="research/out/e57-rung1"
mkdir -p "${out}" research/e57-artifacts

export MLXFAST_E57_DISPATCH_COUNT=1
export MLXFAST_E57_DISPATCH_COUNT_OUT="${PWD}/research/e57-artifacts/dispatch-counts.json"
if ((throwing)); then
  export MLXFAST_E57_DISPATCH_COUNT_THROW=1
  log="${out}/throw.log"
else
  log="${out}/legal.log"
fi

# Cmlx searches for mlx.metallib next to the RUNNING executable, and the xctest
# bundle is a third location that only exists after the tests are built. Build
# the tests first, then publish, then run.
{
  swift build --build-tests --force-resolved-versions \
    && tools/build-mlx-metallib.sh --all-build-roots \
    && swift test --force-resolved-versions \
      --filter E57SdpaChunkDispatchCountTests
} > "${log}" 2>&1
status=$?
echo "e57_run_dispatch_count: throwing=${throwing} exit=${status} log=${log}"
tail -40 "${log}"
# The illegal cell is expected to abort, so only the legal pass gates on status.
((throwing)) && exit 0
exit "${status}"
