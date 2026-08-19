#!/usr/bin/env bash
# Build the three E60 arms as prebuilt binaries, one arm at a time.
#
#   A  upstream/main versions of the two scored files. That ref is the promoted
#      frontier 9e1ff9ec (submission 59b321ee), which is organizer main
#      0c90733d plus the +70-line untimed warmTargetLaterWindowSDPA.
#   B  our campaign base, unchanged.
#   C  arm B plus the hand-applied warm (research/e60-artifacts/arm-c-warm.patch).
#
# Each arm leaves .build/release/mlxfast-swift and
# .build-worker/release/mlxfast-runtime-worker copied into
# research/out/e60/bin/<arm>/, and the working tree is restored to the base
# revision afterwards. Timed legs then select an arm by binary, never by a dirty
# checkout.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

files=(
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
)
bin_root="research/out/e60/bin"
log_root="research/out/e60/build"
mkdir -p "${bin_root}" "${log_root}"

restore_base() {
  git checkout -- "${files[@]}"
}

dirty="$(git status --porcelain -- "${files[@]}" | wc -l | tr -d ' ')"
if [[ "${dirty}" != "0" ]]; then
  echo "e60: the two scored files are dirty before the first build; refusing" >&2
  exit 1
fi

build_arm() {
  local arm="$1"
  echo "=== arm ${arm}: preparing source ==="
  case "${arm}" in
    A) git checkout upstream/main -- "${files[@]}" || return 1 ;;
    B) : ;;
    C)
      git apply --check research/e60-artifacts/arm-c-warm.patch || return 1
      git apply research/e60-artifacts/arm-c-warm.patch || return 1
      ;;
    *) echo "e60: unknown arm ${arm}" >&2; return 2 ;;
  esac

  echo "=== arm ${arm}: building both products ==="
  mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
  CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
    swift build -c release --force-resolved-versions --product mlxfast-swift \
    > "${log_root}/${arm}-cli.log" 2>&1 || {
      echo "e60: arm ${arm} trusted CLI build failed" >&2
      restore_base
      return 1
    }
  CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
    swift build -c release --force-resolved-versions --scratch-path .build-worker \
    --product mlxfast-runtime-worker \
    > "${log_root}/${arm}-worker.log" 2>&1 || {
      echo "e60: arm ${arm} worker build failed" >&2
      restore_base
      return 1
    }

  mkdir -p "${bin_root}/${arm}"
  cp .build/release/mlxfast-swift "${bin_root}/${arm}/mlxfast-swift"
  cp .build-worker/release/mlxfast-runtime-worker \
     "${bin_root}/${arm}/mlxfast-runtime-worker"
  {
    echo "arm=${arm}"
    echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "base_sha=$(git rev-parse HEAD)"
    for file in "${files[@]}"; do
      echo "source_sha256 ${file} $(shasum -a 256 "${file}" | awk '{print $1}')"
    done
    echo "cli_sha256=$(shasum -a 256 "${bin_root}/${arm}/mlxfast-swift" | awk '{print $1}')"
    echo "worker_sha256=$(shasum -a 256 "${bin_root}/${arm}/mlxfast-runtime-worker" | awk '{print $1}')"
  } > "${bin_root}/${arm}/provenance.txt"
  cat "${bin_root}/${arm}/provenance.txt"

  restore_base
  echo "=== arm ${arm}: done, tree restored ==="
}

for arm in "$@"; do
  build_arm "${arm}" || exit 1
done

echo "=== all requested arms built ==="
git status --porcelain -- "${files[@]}"
