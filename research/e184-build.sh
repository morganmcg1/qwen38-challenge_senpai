#!/usr/bin/env bash
# E184: rebuild both binaries the profiling session needs.
#
# `benchmark-qwen-mtp.sh` does NOT rebuild `.build-worker/release/mlxfast-runtime-worker`,
# and the worker is the only process that runs model code -- so a change under
# Sources/ or Vendor/ is invisible to a measurement until this runs. The worker
# uses its own scratch path (setup.sh:2909) so participant-code builds never
# write into the trusted CLI's build directory.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache

CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker || exit $?

swift build -c release --force-resolved-versions --product mlxfast-swift || exit $?

echo "e184-build: worker $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "e184-build: cli    $(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
echo "e184-build: worker profiler env names:"
strings .build-worker/release/mlxfast-runtime-worker | grep 'E184_PREFILL' || {
  echo "e184-build: FAILED -- the worker carries no E184 profiler env name" >&2
  exit 3
}
