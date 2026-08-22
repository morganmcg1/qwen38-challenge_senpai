#!/usr/bin/env bash
# Read-only diagnostic for the rung 5e worker failure.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== mlx.metallib copies ==="
find . -name 'mlx.metallib' -not -path './.git/*' -print0 \
  | xargs -0 -I{} sh -c 'printf "%s  %s  %s\n" "$(shasum -a 256 "{}" | cut -c1-64)" "$(stat -f %z "{}")" "{}"'

echo
echo "=== worker binary ==="
ls -la .build-worker/release/mlxfast-runtime-worker 2>/dev/null || echo "(missing)"
shasum -a 256 .build-worker/release/mlxfast-runtime-worker 2>/dev/null || true

echo
echo "=== top memory consumers ==="
ps -Ao rss,pid,comm -m | head -12

echo
echo "=== disk ==="
df -h . | tail -2
