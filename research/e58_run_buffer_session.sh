#!/usr/bin/env bash
# One counterbalanced session that varies only MLX_MAX_OPS_PER_BUFFER at a fixed
# 512 MiB referenced-byte budget, which is the ranked command-buffer geometry.
#
# The census is OFF in every arm, so these arms are timeable. The buffer limits
# are installed by the E58 research probe before MLX caches them, so no
# candidate file changes. Order is the palindrome A B B A, so monotone thermal
# drift cancels to first order between the two arms.
#
# usage: research/e58_run_buffer_session.sh [TOKENS]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"

run_arm() {
  local tag="$1" ops="$2"
  echo "=== ${tag}: 512 MiB / ${ops} ops, ${tokens} tokens ==="
  research/e58_run_arm.sh "${tag}" \
    --tokens "${tokens}" \
    --buffer-mb 512 \
    --buffer-ops "${ops}" \
    --hot || return 1
}

run_arm e58-bufops-a1 50   || exit 1
run_arm e58-bufops-b1 256  || exit 1
run_arm e58-bufops-b2 256  || exit 1
run_arm e58-bufops-a2 50   || exit 1

echo "=== session complete ==="
