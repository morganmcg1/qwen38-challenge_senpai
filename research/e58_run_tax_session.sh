#!/usr/bin/env bash
# Runs the E58 in-situ dispatch-tax regression as ONE counterbalanced session.
#
# Arms run in palindrome order A B C C B A, so a monotone thermal or clock drift
# across the session cancels to first order in each arm's mean. Every arm runs
# with the census OFF, because the census perturbs timing, and with
# MLXFAST_LOCAL_COOL_GATE=0, which program.md permits for local timed arms when
# the arms are counterbalanced, entry and exit temperature are recorded, and the
# result preserves cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false verbatim. Every arm records its own entry and
# exit GPU temperature in its meta.txt.
#
# The two taxed arms answer two different questions with the same trivial
# kernel and the same count:
#   wait=0  prices one PIPELINED extra dispatch, which is what a decode round
#           actually issues.
#   wait=1  prices one SERIALISED extra dispatch: encode, submit and wait.
# Their slopes bracket the marginal price of a dispatch.
#
# usage:
#   research/e58_run_tax_session.sh SESSION_TAG [TAX_PER_ROUND] [TOKENS]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="${1:?usage: e58_run_tax_session.sh SESSION_TAG [tax] [tokens]}"
tax="${2:-4096}"
tokens="${3:-512}"

arms=(
  "a1:0:1"
  "b1:${tax}:0"
  "c1:${tax}:1"
  "c2:${tax}:1"
  "b2:${tax}:0"
  "a2:0:1"
)

for arm in "${arms[@]}"; do
  IFS=: read -r name arm_tax wait <<<"${arm}"
  tag="${session}-${name}"
  echo "=== ${tag}: tax=${arm_tax} wait=${wait} tokens=${tokens} ==="
  args=("${tag}" --tokens "${tokens}" --hot)
  if ((arm_tax > 0)); then
    args+=(--tax "${arm_tax}" --tax-mode metal --tax-wait "${wait}")
  fi
  research/e58_run_arm.sh "${args[@]}" || {
    echo "e58: arm ${tag} failed" >&2
    exit 1
  }
done
echo "=== session ${session} complete ==="
