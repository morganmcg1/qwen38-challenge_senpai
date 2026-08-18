#!/usr/bin/env bash
# E20: build the two binary arms into a gitignored stash.
#
#   INSTR  current HEAD (MLX_QWEN_ATTRIB instrumentation present)
#   BASE   the two touched files restored to $BASE_SHA (null control)
#
# Both arms ship mlxfast-swift and mlxfast-runtime-worker, because the
# benchmark wrapper drives the CLI and the CLI spawns the worker.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_SHA="${BASE_SHA:-c0f7e370921a14f348fa1872f2176b1b43028752}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/.mlxfast-private/e20/bins}"

TOUCHED=(
  "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
  "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
)

for f in "${TOUCHED[@]}"; do
  if ! git diff --quiet -- "$f" || ! git diff --cached --quiet -- "$f"; then
    echo "refusing to run: $f is dirty; commit or stash first" >&2
    exit 1
  fi
done

# Always restore from HEAD, never from the index: the BASE arm stages the old
# blobs, so `git checkout -- <f>` would restore the polluted index copy.
restore_head() {
  git checkout HEAD -- "${TOUCHED[@]}"
  git restore --staged "${TOUCHED[@]}" 2>/dev/null || true
}
trap restore_head EXIT

build_arm() {
  local arm="$1"
  local dest="$OUT_ROOT/$arm"
  mkdir -p "$dest"
  echo "=== building arm $arm ==="

  swift build -c release --force-resolved-versions --product mlxfast-swift \
    2>&1 | tail -5 || return 1
  CLANG_MODULE_CACHE_PATH="$REPO_ROOT/.build-worker/module-cache" \
    swift build -c release --force-resolved-versions \
      --scratch-path "$REPO_ROOT/.build-worker" \
      --product mlxfast-runtime-worker 2>&1 | tail -5 || return 1

  cp -f "$REPO_ROOT/.build/release/mlxfast-swift" "$dest/mlxfast-swift"
  cp -f "$REPO_ROOT/.build-worker/release/mlxfast-runtime-worker" \
        "$dest/mlxfast-runtime-worker"
  ( cd "$dest" && shasum -a 256 mlxfast-swift mlxfast-runtime-worker \
      > sha256.txt && cat sha256.txt )
  echo "=== arm $arm done ==="
}

build_arm INSTR || exit 1

git checkout "$BASE_SHA" -- "${TOUCHED[@]}"
build_arm BASE || exit 1
restore_head

echo "=== all arms built under $OUT_ROOT ==="
grep -H . "$OUT_ROOT"/*/sha256.txt
