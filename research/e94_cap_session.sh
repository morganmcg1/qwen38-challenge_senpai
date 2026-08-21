#!/usr/bin/env bash
# E94 rung 1: one ordered sequence of offered-cap legs on the SHIPPED arm.
#
#   usage: research/e94_cap_session.sh SUFFIX CAPS [TOKENS]
#
#   SUFFIX  one letter appended to each tag, for example `a` or `b`. Run the
#           reverse cap order under the other letter so the session is
#           counterbalanced against thermal drift.
#   CAPS    comma-separated values for MLXFAST_QWEN_MTP_DEPTH, the OFFERED
#           per-round draft ceiling. The policy still chooses its own depth
#           below the offer, so the histogram is the deliverable.
#
# Every leg is traced, ungated and 512 decode tokens by default. Tags are
# `e94c<CAP><SUFFIX>`.
#
# The session builds nothing. Build and witness the worker before it starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

suffix="${1:?usage: e94_cap_session.sh SUFFIX CAPS [TOKENS]}"
caps="${2:?usage: e94_cap_session.sh SUFFIX CAPS [TOKENS]}"
tokens="${3:-512}"
arm="${E94_ARM:-ship}"

failures=0
IFS=',' read -r -a cap_list <<< "${caps}"
for cap in "${cap_list[@]}"; do
  tag="e94c${cap}${suffix}"
  echo "=== ${tag}: arm=${arm} offered cap=${cap} tokens=${tokens} ==="
  MLXFAST_QWEN_MTP_DEPTH="${cap}" research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?
  {
    echo "e94_cap=${cap}"
    echo "e94_arm=${arm}"
    echo "e94_order=${suffix}"
    echo "experiment=e94-rung1"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e94_cap_session: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

echo "e94_cap_session: ${failures} failed legs"
exit "${failures}"
