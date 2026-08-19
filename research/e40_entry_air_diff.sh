#!/usr/bin/env bash
# E40 deliverable 2: adjudicate H1 from the compiler, at the PRODUCTION entry
# point, with ZERO GPU.
#
# H1 says E27's `<T,*,3> -> <T,*,5>` register pressure costs occupancy and the
# loss is width-proportional. Because `affine_qmv_fast` switches on the RUNTIME
# value `ntg.x` inside one [[kernel]], all width cells share one register
# allocation, so the testable form of H1 is:
#
#     does the KERNEL-WIDE footprint of affine_qmv_fast<bfloat16_t,64,4,false>
#     differ between the campaign baseline and our shipped tree?
#
# If it does not, H1's "every dispatch pays" premise is false at the AIR level
# and the leading candidate dies at zero GPU cost.
#
# The two arms differ ONLY in `quantized.h`, injected through a shadow include
# directory so the working tree is never modified. Compile-only: `metal -S`
# plus `metal-opt`, no metallib, no MTLDevice, no pipeline state, no dispatch.
#
# Usage: research/e40_entry_air_diff.sh [<baseline-rev>] [<candidate-rev>]
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

BASE_REV="${1:-527306761f70e2c4024f347915328894db80c181}"
CAND_REV="${2:-HEAD}"
HDR="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
INC="Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE="research/e40_qmv_entry_probe.metal"
OUT="${MLXFAST_E40_OUT:-/tmp/e40-air}"

mkdir -p "${OUT}"

echo "E40 production-entry AIR comparison (compile-only, zero GPU)"
echo "  toolchain: $(xcrun --sdk macosx metal --version 2>&1 | head -1)"
echo "  probe:     ${PROBE}"
echo "  base rev:  $(git rev-parse --short "${BASE_REV}") $(git log -1 --format=%s "${BASE_REV}")"
echo "  cand rev:  $(git rev-parse --short "${CAND_REV}") $(git log -1 --format=%s "${CAND_REV}")"
echo

build_arm() {
  local rev="$1" tag="$2"
  local shadow="${OUT}/shadow-${tag}/mlx/backend/metal/kernels"
  mkdir -p "${shadow}"
  git show "${rev}:${HDR}" > "${shadow}/quantized.h" || return 1
  echo "  [${tag}] quantized.h sha256 $(shasum -a 256 "${shadow}/quantized.h" | cut -c1-16)"
  xcrun -sdk macosx metal -std=metal3.1 -S -O2 \
    -I "${OUT}/shadow-${tag}" -I "${INC}" \
    "${PROBE}" -o "${OUT}/${tag}.ll" || return 1
  xcrun -sdk macosx metal-opt -passes='default<O3>' -S \
    "${OUT}/${tag}.ll" -o "${OUT}/${tag}.o3.ll" || return 1
  echo "  [${tag}] AIR $(wc -l < "${OUT}/${tag}.ll") lines, O3 $(wc -l < "${OUT}/${tag}.o3.ll") lines"
}

build_arm "${BASE_REV}" base || { echo "FAIL: base arm did not compile"; exit 1; }
build_arm "${CAND_REV}" cand || { echo "FAIL: candidate arm did not compile"; exit 1; }

for k in e40_affine_qmv_fast_bf16_gs64_b4_batch0 \
         e40_affine_qmv_fast_bf16_gs64_b4_batch1 \
         e40_affine_qmv_fast_bf16_gs64_b2_batch0; do
  echo
  echo "=== ${k} ==="
  for tag in base cand; do
    printf '  %-5s ' "${tag}"
    python3 research/air_kernel_stats.py "${OUT}/${tag}.o3.ll" --match "${k}" \
      | sed 's/^/ /'
  done
done

echo
echo "=== whole-module AIR textual diff (O3) ==="
if diff -q "${OUT}/base.o3.ll" "${OUT}/cand.o3.ll" >/dev/null; then
  echo "  IDENTICAL: the two commits compile to byte-identical optimised AIR."
else
  diff "${OUT}/base.o3.ll" "${OUT}/cand.o3.ll" | grep -c '^[<>]' \
    | sed 's/^/  changed AIR lines: /'
fi

echo
echo "=== private-memory (alloca) inventory per kernel ==="
python3 - "${OUT}/base.o3.ll" "${OUT}/cand.o3.ll" <<'PY'
import re, sys
ALLOCA = re.compile(r'alloca\s+([^,]+),')
def scan(path):
    out, cur = {}, None
    for line in open(path):
        m = re.match(r'define .*@([A-Za-z0-9_.]+)\(', line)
        if m:
            cur = m.group(1); out[cur] = []
        elif line.startswith('}'):
            cur = None
        elif cur is not None:
            a = ALLOCA.search(line)
            if a:
                out[cur].append(a.group(1).strip())
    return out
b, c = scan(sys.argv[1]), scan(sys.argv[2])
for k in sorted(set(b) | set(c)):
    if not k.startswith('e40_'):
        continue
    bv, cv = b.get(k, []), c.get(k, [])
    flag = 'SAME' if bv == cv else 'CHANGED'
    print('  %-8s %s' % (flag, k))
    print('    base: %s' % (bv or '(none)'))
    print('    cand: %s' % (cv or '(none)'))
PY
