#!/usr/bin/env bash
# Build the E62 binaries, one at a time, and restore the tree after each build.
#
#   stock   HEAD unchanged. Every timed leg of rungs 1, 2 and 3 runs this
#           binary, so those arms differ ONLY by environment.
#   wired   THROWAWAY research binary. It lowers the 96 GiB wired-residency
#           gate at Qwen36MTPBlockSession.swift:225 to 32 GiB so the ranked-only
#           residency path can run on this 48 GiB host, and it adds a file sink
#           for the residency outcome line, which the timed parent otherwise
#           swallows. NEVER submitted. Rung 1b times wiring-on against
#           wiring-off on THIS binary, selected by DARKBLOOM_QWEN_MTP_WIRED_ZH,
#           so that contrast is also environment-only.
#   census  THROWAWAY research binary carrying the E58 in-process dispatch
#           census, restored from d0b337d^. The census takes a lock on every
#           dispatch, so census legs are counted, never timed.
#   cachegate THROWAWAY research binary for the rung 4 gate. It adds
#           Memory.cacheMemory, Memory.activeMemory and Memory.cacheLimit to the
#           session's existing per-round trace line, so one leg shows how close
#           the charged window comes to the trusted 6 GiB cap.
#
# Each arm leaves its two products in research/out/e62/bin/<arm>/ with a
# provenance file that records the __TEXT,__text sha256 of both products.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

bin_root="research/out/e62/bin"
log_root="research/out/e62/build"
mkdir -p "${bin_root}" "${log_root}"

sources_dirty() {
  git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' '
}

if [[ "$(sources_dirty)" != "0" ]]; then
  echo "e62: Sources are dirty before the first build; refusing" >&2
  git status --porcelain -- Sources Vendor Package.swift >&2
  exit 1
fi

restore_base() {
  git restore --source=HEAD --staged --worktree -- Sources Vendor Package.swift
  # The census patch ADDS a file, which `git restore` cannot remove.
  git clean -fdq -- Sources Vendor
}

build_arm() {
  local arm="$1"
  echo "=== arm ${arm}: preparing source ==="
  case "${arm}" in
    stock) : ;;
    wired) git apply research/e62-artifacts/wired-gate-local.patch || return 1 ;;
    census) git apply research/e62-artifacts/census-probe.patch || return 1 ;;
    cachegate) git apply research/e62-artifacts/cache-gate-local.patch || return 1 ;;
    *) echo "e62: unknown arm ${arm}" >&2; return 2 ;;
  esac

  echo "=== arm ${arm}: building both products ==="
  mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
  CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
    swift build -c release --force-resolved-versions --product mlxfast-swift \
    > "${log_root}/${arm}-cli.log" 2>&1 || {
      echo "e62: arm ${arm} trusted CLI build failed" >&2
      tail -30 "${log_root}/${arm}-cli.log" >&2
      restore_base
      return 1
    }
  CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
    swift build -c release --force-resolved-versions --scratch-path .build-worker \
    --product mlxfast-runtime-worker \
    > "${log_root}/${arm}-worker.log" 2>&1 || {
      echo "e62: arm ${arm} worker build failed" >&2
      tail -30 "${log_root}/${arm}-worker.log" >&2
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
    for file in Sources/MLXFastModel/Qwen36MTPBlockSession.swift \
                Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift; do
      echo "source_sha256 ${file} $(shasum -a 256 "${file}" | awk '{print $1}')"
    done
    echo "cli_sha256=$(shasum -a 256 "${bin_root}/${arm}/mlxfast-swift" | awk '{print $1}')"
    echo "cli_text_sha256=$(python3 research/e60_text_section_sha.py \
      "${bin_root}/${arm}/mlxfast-swift" | awk '{print $1}')"
    echo "worker_sha256=$(shasum -a 256 "${bin_root}/${arm}/mlxfast-runtime-worker" | awk '{print $1}')"
    echo "worker_text_sha256=$(python3 research/e60_text_section_sha.py \
      "${bin_root}/${arm}/mlxfast-runtime-worker" | awk '{print $1}')"
    echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  } > "${bin_root}/${arm}/provenance.txt"
  cat "${bin_root}/${arm}/provenance.txt"

  restore_base
  if [[ "$(sources_dirty)" != "0" ]]; then
    echo "e62: tree still dirty after restoring arm ${arm}" >&2
    return 1
  fi
  echo "=== arm ${arm}: done, tree restored clean ==="
}

for arm in "$@"; do
  build_arm "${arm}" || exit 1
done

echo "=== all requested arms built ==="
