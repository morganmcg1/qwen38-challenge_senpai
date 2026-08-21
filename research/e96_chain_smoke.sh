#!/usr/bin/env bash
# E96 exactness smoke: prove every repeat arm runs and stays bit-identical.
#
#   usage: research/e96_chain_smoke.sh [TOKENS] [FORCE_DRAFTS]
#
# A repeat arm adds an integer carry, or a runtime-zero index, to every store
# address. Both are always zero, so the arm must emit exactly the control's
# tokens and exactly the control's reference rows at every repeat count. This
# script checks that claim on a short window before a timing session spends
# half an hour on an arm that is not bit-exact.
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
  echo "e96_chain_smoke: step=${mode} R=${reps} exit=${status}"
done

# The fused-norm family runs on the unmodified step arm, so its reference rows
# must match the same control.
for reps in 1 4; do
  MLX_E96_NORM=rep MLX_E96_NORM_REPEAT="${reps}" \
    research/e96_direct_leg.sh "e96chain-normrep${reps}" "${tokens}" \
      vendor 4 "${drafts}"
  status=$?
  failures=$((failures + (status != 0)))
  echo "e96_chain_smoke: norm=rep R=${reps} exit=${status}"
done

reference="research/out/e96chain-clone1/golden-rows.json"
for tag in e96chain-repchain1 e96chain-repchain4 e96chain-rep4 \
           e96chain-normrep1 e96chain-normrep4; do
  if cmp -s "${reference}" "research/out/${tag}/golden-rows.json"; then
    echo "e96_chain_smoke: ${tag} rows identical to clone"
  else
    echo "e96_chain_smoke: ${tag} ROWS DIFFER from clone" >&2
    failures=$((failures + 1))
  fi
done

echo "e96_chain_smoke: ${failures} failures"
exit "${failures}"
