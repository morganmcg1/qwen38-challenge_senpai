#!/usr/bin/env bash
# E130 rung 10, F13 section 4: run the admission probe on both arms.
#
#   usage: research/e130_admission_session.sh PREFIX REPS TOKENS
#
# Alternates the arms so any drift in machine state is shared between them.
# There is no timing here, so the alternation is hygiene rather than a
# counterbalanced design.
#
# A failed leg does not stop the session. The reader reports how many draws
# each arm actually produced.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e130_admission_session.sh PREFIX REPS TOKENS}"
reps="${2:?usage: e130_admission_session.sh PREFIX REPS TOKENS}"
tokens="${3:?usage: e130_admission_session.sh PREFIX REPS TOKENS}"

for rep in $(seq 1 "${reps}"); do
  for arm in ${E130_ADMISSION_ARMS:-s64 s512}; do
    tag="${prefix}-${rep}-${arm}"
    echo "############ rep ${rep}/${reps} arm=${arm} tag=${tag} ############"
    research/e130_admission_leg.sh "${tag}" "${arm}" "${tokens}"
    echo "############ rep ${rep}/${reps} arm=${arm} exit=$? ############"
  done
done

echo "=== admission session complete: ${prefix} ==="
