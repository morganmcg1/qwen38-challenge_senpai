#!/usr/bin/env bash
# E158 R1.B -- the precision-island arm curve on the DECLARED head.
#
#   usage: research/e158_island_session.sh ARMS PROMPT_ID [PROMPT_ID ...]
#          ARMS is a comma-separated subset of: all,q,kv,none
#
# `Qwen35IslandArm.fromEnvironment` reads `DARKBLOOM_QWEN_MTP_ISLAND_ARM` and
# defaults to `all`. The `DARKBLOOM_` prefix crosses the runtime-worker
# environment filter; an `MLXFAST_`-spelled selector does not and would leave
# every arm running the shipped default.
#
# Only the declared head carries `precision_islands.*`, so the curve is defined
# on that head alone. The pinned bf16 head has no island tensors and no arm.
#
# NOT A TIMING SESSION. The recall audit stays on so each arm also yields the
# proposal-recall decomposition; every leg records `timing_valid=false`.
# Acceptance is a deterministic function of prompt, head, arm and depth under
# greedy decoding, so arm order cannot bias the accuracy comparison and no
# counterbalancing is required.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arms_spec="${1:-}"
shift || true
[[ -n "${arms_spec}" ]] && (($#)) || {
  echo "usage: research/e158_island_session.sh ARMS PROMPT_ID [...]" >&2
  exit 2
}

IFS=',' read -r -a arms <<<"${arms_spec}"
for arm in "${arms[@]}"; do
  case "${arm}" in
    all | q | kv | none) ;;
    *)
      echo "e158_island_session: bad arm '${arm}'" >&2
      exit 2
      ;;
  esac
done

head_dir="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
mkdir -p .mlxfast-private/e158
status=0

for arm in "${arms[@]}"; do
  trace="${PWD}/.mlxfast-private/e158/trace-island-${arm}.log"
  rm -f "${trace}"
  echo "=== e158 island arm ${arm} -> ${trace} ==="
  env DARKBLOOM_QWEN_MTP_ISLAND_ARM="${arm}" \
    MLX_QWEN_MTP_TRACE_PATH="${trace}" \
    E128_HEAD_DIR="${head_dir}" \
    E155_AUDIT_DIR=".mlxfast-private/e158/island-${arm}" \
    E128_RUNS_DIR="runs-e158-island-${arm}" \
    E128_TOKENS="${E128_TOKENS:-512}" \
    E128_DEPTH="${E128_DEPTH:-8}" \
    research/e155_audit_session.sh "$@" || status=1
  if grep -q "qwen-mtp-island-arm: ${arm}" "${trace}" 2>/dev/null; then
    echo "e158 island witness ${arm}: $(sort -u "${trace}" | tr '\n' ' ')"
  else
    echo "e158 island arm ${arm}: NO TRACE WITNESS" >&2
    status=1
  fi
done

exit "${status}"
