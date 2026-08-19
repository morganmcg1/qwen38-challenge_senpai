#!/usr/bin/env bash
# Research-only (qwen38-r1-e61): build and run the rung 0.1 lane probe.
#
#   research/e61_vec_probe.sh
#
# One tiny single-thread dispatch per kernel. It holds no model, so it does not
# take the resident-model run lock, but it does touch the GPU, so it runs
# through run_job like every other GPU step.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tmp="${TMPDIR:-/tmp}"
out="research/e61-artifacts/e61-vec-probe.json"
mkdir -p research/e61-artifacts

xcrun -sdk macosx metal -std=metal3.1 -O2 -c research/e61_vec_probe.metal -o "${tmp}/e61.air"
xcrun -sdk macosx metallib "${tmp}/e61.air" -o "${tmp}/e61.metallib"
swiftc -O research/e61_vec_check.swift -o "${tmp}/e61_vec_check"
"${tmp}/e61_vec_check" "${tmp}/e61.metallib" "${out}"
