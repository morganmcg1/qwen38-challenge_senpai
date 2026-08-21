#!/usr/bin/env bash
# E96 smoke: prove every ablation arm runs before the counterbalanced session.
#
#   usage: research/e96_smoke.sh [TOKENS] [FORCE_DRAFTS]
#
# A smoke leg is a 32-token window and is never comparable timing evidence.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-32}"
drafts="${2:-4}"
failures=0

for spec in off:4 t1:4 clone:16; do
  IFS=: read -r mode y <<<"${spec}"
  research/e96_leg.sh "e96smoke-${mode}-y${y}" "${tokens}" "${mode}" "${y}" "${drafts}"
  status=$?
  failures=$((failures + (status != 0)))
  echo "e96_smoke: mode=${mode} y=${y} exit=${status}"
done

echo "e96_smoke: ${failures} failed legs"
exit "${failures}"
