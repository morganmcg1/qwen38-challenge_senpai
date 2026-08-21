#!/usr/bin/env bash
# E96 chain smoke: prove the chained repeat arm runs and stays bit-identical.
#
#   usage: research/e96_chain_smoke.sh [TOKENS] [FORCE_DRAFTS]
#
# The chained arm adds an integer carry to every address. The carry is always
# zero, so the arm must emit exactly the clone's tokens and exactly the clone's
# reference rows at every repeat count. This script checks that claim on a
# short window before the timing session spends an hour on it.
#
# A smoke leg is a 32-token window and is never comparable timing evidence.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-32}"
drafts="${2:-4}"
failures=0

for spec in clone:1 repchain:1 repchain:4 rep:4; do
  IFS=: read -r mode reps <<<"${spec}"
  MLX_E96_REPEAT="${reps}" \
    research/e96_direct_leg.sh "e96chain-${mode}${reps}" "${tokens}" \
      "${mode}" 4 "${drafts}"
  status=$?
  failures=$((failures + (status != 0)))
  echo "e96_chain_smoke: mode=${mode} R=${reps} exit=${status}"
done

reference="research/out/e96chain-clone1/golden-rows.json"
for tag in e96chain-repchain1 e96chain-repchain4 e96chain-rep4; do
  if cmp -s "${reference}" "research/out/${tag}/golden-rows.json"; then
    echo "e96_chain_smoke: ${tag} rows identical to clone"
  else
    echo "e96_chain_smoke: ${tag} ROWS DIFFER from clone" >&2
    failures=$((failures + 1))
  fi
done

echo "e96_chain_smoke: ${failures} failures"
exit "${failures}"
