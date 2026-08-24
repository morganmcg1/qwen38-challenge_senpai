#!/bin/bash
# E173 instrument launcher.
#
#   research/e173_run_instruments.sh admission   # host only, evaluates nothing
#   research/e173_run_instruments.sh gpu         # non-weight-pass GPU line items
#
# The admission arm never calls `eval`, so it allocates no device buffer, holds
# no model and needs no thermal gate. The GPU arm runs small real-geometry
# kernels and records entry and exit GPU temperature; it is not gate qualified
# for timing and makes no absolute leg claim.
set -uo pipefail

arm="${1:?usage: e173_run_instruments.sh admission|gpu}"
out_dir="research/e173-artifacts"
mkdir -p "${out_dir}"

case "${arm}" in
admission)
  export MLXFAST_RUN_E173_ADMISSION=1
  export MLXFAST_E173_ADMISSION_OUT="${PWD}/${out_dir}/admission.json"
  export MLXFAST_E173_REPS="${MLXFAST_E173_REPS:-11}"
  export MLXFAST_E173_INNER="${MLXFAST_E173_INNER:-4096}"
  filter="E173AdmissionCensusTests"
  ;;
gpu)
  export MLXFAST_RUN_E173_GPU=1
  export MLXFAST_E173_GPU_OUT="${PWD}/${out_dir}/gpu-line-items.json"
  export MLXFAST_E173_GPU_REPS="${MLXFAST_E173_GPU_REPS:-9}"
  export MLXFAST_E173_GPU_INNER="${MLXFAST_E173_GPU_INNER:-8}"
  filter="E173GPULineItemTests"
  ;;
*)
  echo "unknown arm: ${arm}" >&2
  exit 2
  ;;
esac

# Host-side line items are the deliverable, and the scored worker is a release
# build. A debug `swift test` inflates every host measurement, so set
# E173_RELEASE=1 for any number that will be quoted. `-enable-testing` is
# mandatory: SwiftPM refuses to test in release without it.
config_args=(--force-resolved-versions)
if [[ "${E173_RELEASE:-0}" == "1" ]]; then
  config_args+=(-c release -Xswiftc -enable-testing)
  export MLXFAST_E173_BUILD_CONFIGURATION=release
else
  export MLXFAST_E173_BUILD_CONFIGURATION=debug
fi

echo "arm=${arm} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "configuration=${MLXFAST_E173_BUILD_CONFIGURATION}"
sysctl -n machdep.cpu.brand_string
git rev-parse HEAD
set -o pipefail
swift test "${config_args[@]}" --filter "${filter}" 2>&1 | tail -40
status=$?
echo "arm=${arm} finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}"
exit "${status}"
