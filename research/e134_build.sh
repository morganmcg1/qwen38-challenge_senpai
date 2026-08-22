#!/usr/bin/env bash
# Build the trusted CLI and the participant runtime worker with the same
# scratch paths and clang module caches benchmark.sh uses, so a research
# session does not invalidate the wrapper's incremental build state.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache

CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift

CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
    --scratch-path .build-worker --product mlxfast-runtime-worker
