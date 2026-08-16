#!/usr/bin/env bash
# Research-only: rebuild the trusted CLI and the participant runtime worker with
# the same flags setup.sh uses, so an edit under Sources/MLXFastModel actually
# reaches the scored worker binary at .build-worker/release/mlxfast-runtime-worker.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache

echo "rebuild.sh: mlxfast-swift"
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift || exit 1

echo "rebuild.sh: mlxfast-runtime-worker"
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions --scratch-path .build-worker \
  --product mlxfast-runtime-worker || exit 1

for bin in .build/release/mlxfast-swift .build-worker/release/mlxfast-runtime-worker; do
  if [[ ! -x "${bin}" ]]; then
    echo "rebuild.sh: missing ${bin}" >&2
    exit 1
  fi
  echo "rebuild.sh: ok ${bin} mtime=$(stat -f %Sm -t '%Y-%m-%dT%H:%M:%S' "${bin}")"
done
