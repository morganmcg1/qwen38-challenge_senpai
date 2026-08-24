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

echo "arm=${arm} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sysctl -n machdep.cpu.brand_string
git rev-parse HEAD
swift test --force-resolved-versions --filter "${filter}" 2>&1 | tail -40
status=$?
echo "arm=${arm} finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}"
exit "${status}"
