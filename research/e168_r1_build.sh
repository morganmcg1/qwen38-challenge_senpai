#!/usr/bin/env bash
# Research-only (qwen38-r1-e168, revision r1): rebuild every release product of
# the organizer-pure cap-4 candidate, including mlx.metallib.
#
# The r1 tree reverts the campaign quantized kernels to organizer content, so
# the mlx.metallib on this host was compiled from kernel sources that the
# candidate no longer contains. Timing against it would measure neither arm.
# The metallib is therefore rebuilt here, after the Swift products, so it is
# newer than every vendored kernel source.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift || exit 1
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker || exit 1

tools/build-mlx-metallib.sh --all-build-roots || exit 1

worker=".build-worker/release/mlxfast-runtime-worker"
metallib=".build-worker/release/mlx.metallib"
[[ -s "${metallib}" ]] || {
  echo "e168_r1_build: ${metallib} is missing after the rebuild" >&2
  exit 1
}
if [[ "$(strings -a "${worker}" | grep -c -- "MLX_QWEN_MTP_TRACE" || true)" -eq 0 ]]; then
  echo "e168_r1_build: ${worker} looks stale" >&2
  exit 1
fi

echo "e168_r1_build: cli      $(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
echo "e168_r1_build: worker   $(shasum -a 256 "${worker}" | cut -d' ' -f1)"
echo "e168_r1_build: metallib $(shasum -a 256 "${metallib}" | cut -d' ' -f1)"
echo "e168_r1_build: head     $(git rev-parse HEAD)"
echo "e168_r1_build: dirty    $(git status --porcelain | wc -l | tr -d ' ')"
