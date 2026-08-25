#!/usr/bin/env bash
# E213 Stage-0 desk probe: register-pressure occupancy of every QMV width plan
# and of every legal (m, IPG) instantiation.
#
#   research/e213_occupancy.sh [TAG]
#
# `maxTotalThreadsPerThreadgroup` is the register budget the Metal back end
# assigned to a kernel, so it reads the IPG cliff that the plan table cannot
# show. No dispatch, no timing, no model: this compiles kernels and builds
# pipeline states only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:-e213/occupancy}"
out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

metal_src="${out}/probe.metal"
python3 research/e213_probe_gen.py "${metal_src}" || exit 1

xcrun -sdk macosx metal -std=metal3.1 -O2 -c "${metal_src}" -o "${out}/probe.air" \
  2> "${out}/compile.log" || { cat "${out}/compile.log"; exit 1; }
xcrun -sdk macosx metallib "${out}/probe.air" -o "${out}/probe.metallib" || exit 1

swiftc -O research/crossrow_na_occupancy.swift -o "${out}/na_occupancy" || exit 1
"${out}/na_occupancy" "${out}/probe.metallib" | tee "${out}/occupancy.txt"

python3 research/e213_regs.py "${metal_src}" "${out}/regs.json" | tee "${out}/regs.txt"
