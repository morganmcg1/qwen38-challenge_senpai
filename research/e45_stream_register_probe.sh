#!/usr/bin/env bash
# E45 section 5: is the board's width-8 stream A/B confounded by occupancy?
#
# The 14 dispatch-only pairs in research/e45-stream-ab.json differ at exactly one
# cell: `qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>` (ceil(8/4) = 2 weight
# streams) versus `<T, 8, 3, true>` (ceil(8/3) = 3). Their score difference is
# read as the marginal cost of one stream. That reading is only valid if the two
# arms are otherwise identical to the machine.
#
# They might not be. `affine_qmv_fast` switches on the RUNTIME value `ntg.x`
# inside ONE [[kernel]] (quantized.h:1925), so every width cell 2..9 is inlined
# into a single function with a SINGLE register allocation equal to the worst
# branch. If the width-8 cell sets that kernel-wide peak, swapping its IPG moves
# occupancy for EVERY width, and the measured difference is an occupancy effect
# wearing a stream effect's clothes.
#
# The prediction is that it does NOT move, and the reason is structural rather
# than empirical. `_m<T, M, IPG>` dispatches to `_wide<T, NA>` bodies:
#
#     <T,8,4>  TAIL=0             -> wide<T,4>
#     <T,8,3>  TAIL = 8 % 3 = 2   -> wide<T,3> and wide<T,2>
#
# and the shipped table already instantiates all three NA values elsewhere
# (M4/M7 -> wide<T,4>; M3/M5/M6/M9 -> wide<T,3>; M5/M7 tails -> wide<T,2>).
# Neither arm introduces a body the other lacks, so the max over inlined bodies
# has nothing new to range over. But the register allocator, not this comment, is
# the authority: `_m` is templated on M as well as IPG, so the two arms are
# distinct instantiations whose bound checks, branch layout and inlining
# decisions the compiler is free to treat differently.
#
# Two readouts, weakest to strongest:
#   1. AIR peak-live-register proxy and private-memory inventory at the
#      production entry point. Compile-only, zero GPU.
#   2. `maxTotalThreadsPerThreadgroup` from a real pipeline state, which on Apple
#      GPUs is capped by the register budget the back end actually assigned. This
#      is the authoritative number. It creates an MTLDevice and a pipeline state
#      but dispatches NOTHING and times NOTHING, so it is a compile-time query,
#      not a measurement. Skipped automatically when no device is present.
#
# The arms differ ONLY in the one template argument, injected through a shadow
# include directory. The working tree is never modified.
#
# Usage: research/e45_stream_register_probe.sh
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

HDR="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
INC="Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE="research/e40_qmv_entry_probe.metal"
OUT="${MLXFAST_E45_OUT:-/tmp/e45-registers}"
SHIPPED='qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>'
ALTERNATE='qmv_fast_crossrow_affine4_g64_m<T, 8, 3, true>'

KERNEL=e40_affine_qmv_fast_bf16_gs64_b4_batch0

rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "E45 section 5: width-8 stream A/B occupancy confound"
echo "  toolchain: $(xcrun --sdk macosx metal --version 2>&1 | head -1)"
echo "  probe:     ${PROBE}"
echo "  header:    ${HDR} @ $(git rev-parse --short HEAD)"
echo "  scored kernel: ${KERNEL}"
echo

# The swap has to hit exactly one cell. A count other than 1 means the dispatch
# table was restructured under this script and every number below would be
# attributed to the wrong change, so refuse to guess.
occurrences="$(grep -c -F "${SHIPPED}" "${HDR}")"
if [[ "${occurrences}" != "1" ]]; then
  echo "FAIL: expected exactly 1 occurrence of the width-8 cell, found ${occurrences}"
  echo "      the dispatch table changed shape; re-derive the swap before trusting this probe"
  exit 1
fi
echo "  swap site: 1 occurrence of '${SHIPPED}' (verified unique)"
echo "             s2 arm keeps IPG 4 -> ceil(8/4) = 2 streams"
echo "             s3 arm uses  IPG 3 -> ceil(8/3) = 3 streams"
echo

build_arm() {
  local tag="$1" ipg="$2"
  local shadow="${OUT}/shadow-${tag}/mlx/backend/metal/kernels"
  mkdir -p "${shadow}"
  if [[ "${ipg}" == "4" ]]; then
    cp "${HDR}" "${shadow}/quantized.h" || return 1
  else
    # -F-style literal replacement via awk: sed's metacharacter handling would
    # have to be escaped around the angle brackets and commas.
    awk -v old="${SHIPPED}" -v new="${ALTERNATE}" '
      { i = index($0, old); if (i) { $0 = substr($0, 1, i-1) new substr($0, i+length(old)) } print }
    ' "${HDR}" > "${shadow}/quantized.h" || return 1
    if ! grep -q -F "${ALTERNATE}" "${shadow}/quantized.h"; then
      echo "  FAIL: [${tag}] swap did not apply"; return 1
    fi
  fi
  echo "  [${tag}] quantized.h sha256 $(shasum -a 256 "${shadow}/quantized.h" | cut -c1-16)" \
       "IPG=${ipg} streams=$(( (8 + ipg - 1) / ipg ))"
  xcrun -sdk macosx metal -std=metal3.1 -S -O2 \
    -I "${OUT}/shadow-${tag}" -I "${INC}" \
    "${PROBE}" -o "${OUT}/${tag}.ll" || return 1
  xcrun -sdk macosx metal-opt -passes='default<O3>' -S \
    "${OUT}/${tag}.ll" -o "${OUT}/${tag}.o3.ll" || return 1
  echo "  [${tag}] AIR $(wc -l < "${OUT}/${tag}.ll") lines, O3 $(wc -l < "${OUT}/${tag}.o3.ll") lines"
  # Object + metallib for the occupancy readout below.
  xcrun -sdk macosx metal -std=metal3.1 -O2 -c \
    -I "${OUT}/shadow-${tag}" -I "${INC}" \
    "${PROBE}" -o "${OUT}/${tag}.air" || return 1
  xcrun -sdk macosx metallib "${OUT}/${tag}.air" -o "${OUT}/${tag}.metallib" || return 1
}

build_arm s2 4 || { echo "FAIL: s2 (shipped, 2-stream) arm did not compile"; exit 1; }
build_arm s3 3 || { echo "FAIL: s3 (alternate, 3-stream) arm did not compile"; exit 1; }

echo
echo "=== readout 1: AIR statistics at the production entry point (zero GPU) ==="
for k in "${KERNEL}" \
         e40_affine_qmv_fast_bf16_gs64_b4_batch1 \
         e40_affine_qmv_fast_bf16_gs64_b2_batch0; do
  echo
  echo "--- ${k} ---"
  for tag in s2 s3; do
    printf '  %-3s ' "${tag}"
    python3 research/air_kernel_stats.py "${OUT}/${tag}.o3.ll" --match "${k}" | sed 's/^/ /'
  done
done

echo
echo "=== whole-module optimised AIR diff ==="
if diff -q "${OUT}/s2.o3.ll" "${OUT}/s3.o3.ll" >/dev/null; then
  echo "  IDENTICAL: the two tables compile to byte-identical optimised AIR."
  echo "  (would mean the swap was optimised away -- treat as instrument failure)"
else
  diff "${OUT}/s2.o3.ll" "${OUT}/s3.o3.ll" | grep -c '^[<>]' \
    | sed 's/^/  changed AIR lines: /'
  echo "  (a non-zero count is EXPECTED: the width-8 body really did change)"
fi

echo
echo "=== readout 2: register-limited occupancy from the back end (authoritative) ==="
if ! /usr/bin/xcrun swiftc -O research/crossrow_na_occupancy.swift \
      -o "${OUT}/occupancy" 2>"${OUT}/swiftc.log"; then
  echo "  SKIP: could not build the occupancy reader"
  sed 's/^/    /' "${OUT}/swiftc.log" | head -20
else
  for tag in s2 s3; do
    echo "--- ${tag} ---"
    if ! "${OUT}/occupancy" "${OUT}/${tag}.metallib" 2>"${OUT}/${tag}.occ.err" \
         | sed 's/^/  /'; then
      echo "  SKIP: no Metal device or pipeline creation failed"
      sed 's/^/    /' "${OUT}/${tag}.occ.err" | head -10
    fi
  done
fi

echo
echo "=== verdict inputs ==="
echo "  Compare peak_live_regs / allocas / maxThreads for ${KERNEL}."
echo "  EQUAL      -> the width-8 stream A/B is NOT confounded by kernel-wide"
echo "                occupancy; the board contrast isolates stream count."
echo "  s3 WORSE   -> the 3-stream arm pays an occupancy penalty that every"
echo "                width shares, and +0.4910%/stream is an upper bound on"
echo "                the stream mechanism alone."
echo "  s3 BETTER  -> the shipped 2-stream table is leaving occupancy on the"
echo "                table and the sign of the board estimate is suspect."
echo
echo "  Working tree untouched: $(git status --porcelain "${HDR}" | wc -l | tr -d ' ') modified lines in ${HDR}"
