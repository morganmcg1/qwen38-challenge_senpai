#!/usr/bin/env bash
# Research-only (qwen38-r1-e168-margin-clamp-calibration): rebuild both release
# products and REFUSE to hand back a worker that predates the trace fields this
# experiment reads.
#
# benchmark-qwen-mtp.sh drives the CLI directly and reaches benchmark.sh only
# through --local-cool-gate-only, which returns before the build gate. It
# therefore runs a stale `.build-worker` twin without saying so. One p7 census
# leg was already lost that way: the round lines carried `d=` and `acc=` but no
# `m=`, `offer=` or `ema=`, because the binary predated the commit that adds
# them. The tripwire below turns that silent loss into a build failure.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift || exit 1
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker || exit 1

worker=".build-worker/release/mlxfast-runtime-worker"
# grep -c, not grep -q: under `pipefail` an early-exiting grep SIGPIPEs strings
# and the pipeline reports 141 even on a match.
for token in "wcap=%d" "MLX_E159_FIXED_DRAFT_DEPTH" "MLX_QWEN_MTP_TRACE"; do
  if [[ "$(strings -a "${worker}" | grep -c -- "${token}" || true)" -eq 0 ]]; then
    echo "e168_build: ${worker} has no '${token}'; refusing to census a stale worker" >&2
    exit 1
  fi
done

echo "e168_build: cli    $(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
echo "e168_build: worker $(shasum -a 256 "${worker}" | cut -d' ' -f1)"
echo "e168_build: head   $(git rev-parse HEAD)"
