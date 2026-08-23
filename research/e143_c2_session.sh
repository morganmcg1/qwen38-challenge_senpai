#!/usr/bin/env bash
# E143 C2: the live acceptance and time exchange for an affine-4 K/V island.
#
#   usage: research/e143_c2_session.sh [TOKENS] [--ungated]
#
# C2 asks what happens when the head's K and V precision islands stop being
# dense BF16 and become affine-4 group-64. `research/e143-c2-head.json` proves
# the declared head's own affine-4 `layers.0.self_attn.{k,v}_proj` rows are
# bit-identical to `quantize(precision_islands.{k,v}.weight, 64, 4)`, so the
# shipped selector `DARKBLOOM_QWEN_MTP_ISLAND_ARM=q` already computes C2's
# EXACT numerics with no source edit. This session measures that arm live,
# which is what Rule 107 demands and what an offline screen cannot supply.
#
# ABBA, so monotone thermal drift cancels to first order in the arm contrast.
# `--cool-gate` is the default here: the expected effect is about 2 % of the
# leg and the whole point is to price it, so the legs are gate-qualified
# unless the caller explicitly asks otherwise.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
[[ "${tokens}" =~ ^[0-9]+$ ]] || { echo "usage: e143_c2_session.sh [TOKENS] [--ungated]" >&2; exit 2; }
shift || true

gate="--cool-gate"
[[ "${1:-}" == "--ungated" ]] && gate=""

fail=0
for leg in 1:all 2:q 3:q 4:all; do
  index="${leg%%:*}"
  arm="${leg##*:}"
  tag="e143c2-${index}-${arm}"
  echo "=== ${tag} ($(date -u +%H:%M:%SZ)) ==="
  research/e124_leg.sh "${tag}" "${arm}" "${tokens}" ${gate}
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "e143_c2_session.sh: ${tag} exited ${status}" >&2
    fail=1
    break
  fi
done

exit "${fail}"
