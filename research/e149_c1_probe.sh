#!/usr/bin/env bash
# E149 arm C1 -- run the SDPA query-split cost probe.
#
# The probe is a research instrument under Tests/, so it never reaches a
# submission archive. It is off unless MLXFAST_RUN_E149_PROBE=1.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MLXFAST_RUN_E149_PROBE=1
export MLXFAST_E149_KL="${MLXFAST_E149_KL:-513,769,1024}"
export MLXFAST_E149_WIDTHS="${MLXFAST_E149_WIDTHS:-6,7,8,9}"
export MLXFAST_E149_OUT="${MLXFAST_E149_OUT:-research/e149-c1-sdpa.json}"

swift test --force-resolved-versions --filter E149SdpaSplitCost
rc=$?
echo "e149_c1_probe: swift test exit ${rc}"
exit "${rc}"
