#!/usr/bin/env bash
# Start the MTP runtime worker directly with stdin closed so its full stderr is
# visible. The parent harness swallows everything except the first stderr line.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"

echo "=== metallib copies ==="
find . -name 'mlx.metallib' -not -path './.git/*' -exec ls -la '{}' ';'

echo
echo "=== mtp-runtime-worker, stdin closed ==="
./.build-worker/release/mlxfast-runtime-worker mtp-runtime-worker \
  --weights "${MLXFAST_WEIGHTS_PATH:-weights}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" </dev/null
echo "rc=$?"
