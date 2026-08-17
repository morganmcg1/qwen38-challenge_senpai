#!/bin/bash
# Research-only passthrough for MLXFAST_SWIFT_BIN (e12 seed-prefill charge).
#
# benchmark-qwen-mtp.sh deletes its scratch run directory on exit, and the per-arm
# `mtp-timed` payload -- the only place `seed_prefill_seconds` is reported -- lives
# in there. mtp-timed's `--output PATH` writes the SAME payload it already wrote to
# stdout, after the timed window has closed, so asking for a persistent copy adds a
# post-measurement file write and changes nothing the wrapper reads or times.
set -uo pipefail

real="${MLXFAST_E12_REAL_SWIFT_BIN:-.build/release/mlxfast-swift}"
tag="${MLXFAST_E12_TAG:-untagged}"

args=("$@")
if [[ "${1:-}" == "mtp-timed" ]]; then
  depth="unknown"
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--mtp-depth" ]]; then
      depth="${args[i + 1]:-unknown}"
      break
    fi
  done
  args+=(--output "research/score-e12-${tag}-depth${depth}.json")
fi

exec "${real}" "${args[@]}"
