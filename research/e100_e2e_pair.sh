#!/usr/bin/env bash
# E100 -- run both depth sessions on ONE build.
#
#   usage: research/e100_e2e_pair.sh SLOT ARM
#
#   SLOT  a1 | b1 | b2 | a2, the ABBA position
#   ARM   base | collapse
#
# Two builds and four slots give the ABBA session. Running the depth-8 and
# depth-4 legs back to back inside one slot keeps the number of build switches
# at three while still counterbalancing each session on its own.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

slot="${1:?usage: e100_e2e_pair.sh SLOT ARM}"
arm="${2:?usage: e100_e2e_pair.sh SLOT ARM}"

rc=0
for depth in 8 4; do
  MLXFAST_QWEN_MTP_DEPTH="${depth}" \
    research/e100_e2e_leg.sh "e100-e2e-d${depth}-${slot}" "${arm}" || rc=$?
done
exit "${rc}"
