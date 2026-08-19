#!/usr/bin/env bash
# E44 Gate 0: adjudicate the simdgroup-matrix QMV candidate's REGISTER claim
# from the compiler, with ZERO GPU seconds and zero timed runs.
#
# `affine_qmv_fast` switches on the runtime value `ntg.x` inside one [[kernel]]
# with every helper inlined, so all width cells share ONE register allocation
# equal to the max over cells. E27's revert measured what that costs: a cell
# that raises the kernel-wide max taxes the widths it never touched. So the
# candidate is gated on the allocation before it is gated on speed.
#
# Reports, per arm:
#   * per-cell and KERNEL-WIDE peak_live_regs, naive and lane-corrected;
#   * spill allocas (the only genuine compiler outcome available on this box);
#   * static threadgroup-memory bytes, which is the OTHER shared allocation the
#     single-kernel structure makes kernel-wide and which E27 never touched.
#
# The lane correction exists because AIR models an 8x8 simdgroup matrix as one
# simdgroup-wide `<64 x float>` value, not a per-lane vector: its 64 elements are
# distributed over 32 lanes, so the per-lane cost is 2 registers and the
# uncorrected instrument over-reports it by exactly 32x. Both numbers are
# printed; see research/air_kernel_stats.py.
#
# Usage: research/e44_sgmm_air.sh [<baseline-rev>]
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

BASE_REV="${1:-efff400c1b5554be2e8993b01856653d55de7664}"
HDR="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
INC="Vendor/mlx-swift/Source/Cmlx/mlx"
CELLS="research/e44_sgmm_probe.metal"
ENTRY="research/e40_qmv_entry_probe.metal"
OUT="${MLXFAST_E44_OUT:-/tmp/e44-air}"
mkdir -p "${OUT}"

echo "E44 Gate 0 register readout (compile-only, zero GPU)"
echo "  toolchain:  $(xcrun --sdk macosx metal --version 2>&1 | head -1)"
echo "  worktree:   $(git rev-parse --short HEAD)$(git diff --quiet -- "${HDR}" || echo ' +dirty quantized.h')"
echo "  baseline:   $(git rev-parse --short "${BASE_REV}") $(git log -1 --format=%s "${BASE_REV}")"
echo "  cells:      ${CELLS}"
echo "  entry:      ${ENTRY}"
echo

# Does the worktree header carry the candidate cell yet?
SGMM_FLAG=()
if grep -q "qmv_fast_crossrow_affine4_g64_sgmm" "${HDR}"; then
  SGMM_FLAG=(-DE44_SGMM=1)
  echo "  arm:        CANDIDATE (worktree quantized.h defines the sgmm cell)"
else
  echo "  arm:        BASELINE (worktree quantized.h has no sgmm cell)"
fi
echo

compile() {
  local src="$1" tag="$2"
  shift 2
  xcrun -sdk macosx metal -std=metal3.1 -S -O2 "$@" -I "${INC}" "${src}" \
    -o "${OUT}/${tag}.ll" || return 1
  xcrun -sdk macosx metal-opt -passes='default<O3>' -S \
    "${OUT}/${tag}.ll" -o "${OUT}/${tag}.o3.ll" || return 1
  echo "  [${tag}] AIR $(wc -l < "${OUT}/${tag}.ll") lines, O3 $(wc -l < "${OUT}/${tag}.o3.ll") lines"
}

echo "=== 1. per-cell footprint (worktree header) ==="
compile "${CELLS}" cells ${SGMM_FLAG[@]+"${SGMM_FLAG[@]}"} \
  || { echo "FAIL: cell probe did not compile"; exit 1; }
echo
python3 research/e44_air_summary.py "${OUT}/cells.o3.ll" --cells --dispatch-from "${HDR}"
echo

echo "=== 2. production entry point, baseline vs worktree ==="
BASE_SHADOW="${OUT}/shadow-base/mlx/backend/metal/kernels"
mkdir -p "${BASE_SHADOW}"
git show "${BASE_REV}:${HDR}" > "${BASE_SHADOW}/quantized.h" || exit 1
echo "  [base] quantized.h sha256 $(shasum -a 256 "${BASE_SHADOW}/quantized.h" | cut -c1-16)"
echo "  [cand] quantized.h sha256 $(shasum -a 256 "${HDR}" | cut -c1-16)"
python3 - "${BASE_SHADOW}/quantized.h" <<'PY'
import pathlib, sys

sys.path.insert(0, "research")
from e44_air_summary import BASE_TABLE, dispatch_table_from_header

derived = dispatch_table_from_header(pathlib.Path(sys.argv[1]))
same = derived == BASE_TABLE
print(f"  [base] dispatch table matches the built-in BASE_TABLE: "
      f"{'yes' if same else 'NO'}")
if not same:
    print(f"         derived: {derived}")
    print(f"         builtin: {BASE_TABLE}")
    raise SystemExit(1)
PY
compile "${ENTRY}" entry-base -I "${OUT}/shadow-base" || { echo "FAIL: baseline entry arm"; exit 1; }
compile "${ENTRY}" entry-cand || { echo "FAIL: candidate entry arm"; exit 1; }
echo
python3 research/e44_air_summary.py "${OUT}/entry-base.o3.ll" "${OUT}/entry-cand.o3.ll" --entry
