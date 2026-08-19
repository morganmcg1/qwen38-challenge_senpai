#!/usr/bin/env bash
# Research-only (qwen38-r1-e61): build and run the rung 0.1 lane probe.
#
#   research/e61_vec_probe.sh [NAME=FLAGS ...]
#
# Each spec compiles the same probe under one set of floating-point / language
# rules and writes research/e61-artifacts/e61-vec-probe-NAME.json. That is how
# the NA >= 6 lane mismatch is attributed to the build rather than to the
# vector layout.
#
# `mlxmatch` reproduces what MLX's runtime JIT actually does for the scored
# quantized kernels: Device::build_library_ calls setFastMathEnabled(false) and
# setLanguageVersion(get_metal_version()), which is LanguageVersion4_0 on
# macOS 26. `fastmath31` is the loose build used for the first probe pass.
#
# One tiny single-thread dispatch per kernel per variant. It holds no model, so
# it does not take the resident-model run lock, but it does touch the GPU, so
# it runs through run_job like every other GPU step.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

specs=("$@")
if [[ ${#specs[@]} -eq 0 ]]; then
  specs=(
    "mlxmatch=-std=metal4.0 -fno-fast-math"
    "fastmath40=-std=metal4.0"
    "fastmath31=-std=metal3.1"
    "nocontract31=-std=metal3.1 -ffp-contract=off"
  )
fi

tmp="${TMPDIR:-/tmp}"
mkdir -p research/e61-artifacts
swiftc -O research/e61_vec_check.swift -o "${tmp}/e61_vec_check"

status=0
for spec in "${specs[@]}"; do
  name="${spec%%=*}"
  flags="${spec#*=}"
  [[ "${flags}" == "${name}" ]] && flags=""
  out="research/e61-artifacts/e61-vec-probe-${name}.json"
  echo "=== e61_vec_probe: variant=${name} flags='${flags}' ==="
  # shellcheck disable=SC2086
  xcrun -sdk macosx metal -O2 ${flags} \
    -c research/e61_vec_probe.metal -o "${tmp}/e61-${name}.air"
  xcrun -sdk macosx metallib "${tmp}/e61-${name}.air" -o "${tmp}/e61-${name}.metallib"
  "${tmp}/e61_vec_check" "${tmp}/e61-${name}.metallib" "${out}" || status=$?
done
exit "${status}"
