#!/usr/bin/env bash
# Pay the cold `--build-tests -enable-testing` build once, as its own job, so a
# later parity job only rebuilds the one changed translation unit per arm and
# cannot time out mid-arm.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

mkdir -p .build/clang-module-cache
export CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache"

swift build -c release --build-tests --force-resolved-versions -Xswiftc -enable-testing
tools/build-mlx-metallib.sh --all-build-roots

echo "e42-parity-prebuild: test bundle and metallib ready"
