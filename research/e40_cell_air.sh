#!/usr/bin/env bash
# E40 deliverable 2, per-cell arm. Compiles research/e40_cell_probe.metal against
# the SHIPPED quantized.h (which has the NA<=5 static_assert, so both the base
# and candidate M=5/M=9 cells are legal in one module) and prints every cell's
# AIR footprint, then the kernel-wide maximum for each arm's dispatch table.
#
# Compile-only: `metal -S` plus `metal-opt`. No metallib, no MTLDevice, no
# pipeline state, no dispatch, zero GPU.
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

INC="Vendor/mlx-swift/Source/Cmlx/mlx"
PROBE="research/e40_cell_probe.metal"
OUT="${MLXFAST_E40_OUT:-/tmp/e40-air}"
mkdir -p "${OUT}"

echo "E40 per-cell AIR footprint (compile-only, zero GPU)"
echo "  toolchain: $(xcrun --sdk macosx metal --version 2>&1 | head -1)"
echo "  probe:     ${PROBE}"
echo

xcrun -sdk macosx metal -std=metal3.1 -S -O2 -I "${INC}" "${PROBE}" \
  -o "${OUT}/cells.ll" 2>/dev/null || { echo "FAIL: probe did not compile"; exit 1; }
xcrun -sdk macosx metal-opt -passes='default<O3>' -S "${OUT}/cells.ll" \
  -o "${OUT}/cells.o3.ll" || exit 1
echo "  AIR $(wc -l < "${OUT}/cells.ll") lines, O3 $(wc -l < "${OUT}/cells.o3.ll") lines"
echo

python3 research/air_kernel_stats.py "${OUT}/cells.o3.ll" --match e40_ \
  | sed 's/ fma_f32.*peak_live_regs=/  regs=/; s/ peak_live_values.*types=/  allocas=/' \
  | sed 's/^/  /'

echo
python3 - "${OUT}/cells.o3.ll" <<'PY'
import pathlib, sys
sys.path.insert(0, 'research')
from air_kernel_stats import peak_live_registers, kernels

body = {name: peak_live_registers(lines)[0]
        for name, lines in kernels(pathlib.Path(sys.argv[1])).items()
        if name.startswith('e40_')}

# The dispatch tables as the switch actually instantiates them (quantized.h:1918+
# for >=4096, :1971+ for <4096). The kernel-wide allocation must satisfy the max.
BASE = {3: 'e40_m3_ipg3', 4: 'e40_m4_ipg4', 5: 'e40_m5_ipg3_base',
        6: 'e40_m6_ipg3', 7: 'e40_m7_ipg4', 8: 'e40_m8_ipg4',
        9: 'e40_m9_ipg3_base'}
CAND = dict(BASE)
CAND.update({5: 'e40_m5_ipg5_cand', 9: 'e40_m9_ipg5_cand'})
NARROW = {m: 'e40_narrow_m%d' % m for m in range(2, 10)}

print('KERNEL-WIDE MAXIMUM (what one register allocation must satisfy)')
print('%-8s %-24s %-8s' % ('arm', 'binding cell', 'regs'))
for lbl, tbl in (('base', BASE), ('cand', CAND)):
    cells = dict(tbl)
    cells.update({('n%d' % m): v for m, v in NARROW.items()})
    hi = max(cells.items(), key=lambda kv: body[kv[1]])
    print('%-8s %-24s %-8d' % (lbl, hi[1], body[hi[1]]))

print()
print('per-M delta, >=4096 dispatch table (cand - base)')
print('%-4s %-22s %-8s %-22s %-8s %-8s' % ('M', 'base cell', 'regs', 'cand cell', 'regs', 'delta'))
for m in sorted(BASE):
    b, c = BASE[m], CAND[m]
    print('%-4d %-22s %-8d %-22s %-8d %+8d'
          % (m, b, body[b], c, body[c], body[c] - body[b]))

print()
print('<4096 dispatch table (E27 never touched it) -- is it already the binding cell?')
for m in sorted(NARROW):
    print('  M=%d %-20s regs=%d' % (m, NARROW[m], body[NARROW[m]]))

print()
print('inner packing factor alone (E13/E27/E32 anchor was 62/83/104/125)')
for na in (2, 3, 4, 5):
    k = 'e40_wide_na%d' % na
    print('  NA=%d %-20s regs=%d' % (na, k, body[k]))
PY
