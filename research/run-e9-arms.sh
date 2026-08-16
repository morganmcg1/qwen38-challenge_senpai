#!/usr/bin/env bash
# Research-only driver: run several draft-bits arms back to back on ONE prompt.
#
#   research/run-e9-arms.sh LABEL FIXTURE BITS [BITS...]
#
# Arms must share a host and hot state to be comparable, so they run
# sequentially in a single job rather than as separate launches. FIXTURE
# selects the seed via MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE; the timed legs
# still regenerate their own reference rows, so the fixture only pins the seed
# and the drift tripwire.
set -euo pipefail

label="${1:?usage: run-e9-arms.sh LABEL FIXTURE BITS [BITS...]}"
fixture="${2:?usage: run-e9-arms.sh LABEL FIXTURE BITS [BITS...]}"
shift 2
(($# > 0)) || {
  echo "run-e9-arms.sh: need at least one BITS value" >&2
  exit 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

[[ -f "${fixture}" ]] || {
  echo "run-e9-arms.sh: no such fixture: ${fixture}" >&2
  exit 1
}

export MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="$(cd "$(dirname "${fixture}")" && pwd)/$(basename "${fixture}")"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qwen38-r1-e9-draft-bits-default}"

base_sha="${E9_BASE_SHA:-8970d775a63a28b610fd418c68873c236ce6b86c}"

for bits in "$@"; do
  echo "run-e9-arms: === ${label} bits=${bits} ==="
  research/run-draft-bits-arm.sh "${bits}" "e9-${label}-b${bits}" 512 "${base_sha}"
done

echo "run-e9-arms: all arms complete for ${label}"
