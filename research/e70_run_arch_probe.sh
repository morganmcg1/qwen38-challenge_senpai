#!/usr/bin/env bash
# Research-only launcher for the E70 rung-1 architecture probe.
#
# run_job takes an argv list with no environment field, so MLX_METAL_GPU_ARCH
# has to be exported inside a script.
#
# usage:
#   research/e70_run_arch_probe.sh real                    # one process, all cells
#   research/e70_run_arch_probe.sh applegpu_g17s           # one process PER CELL
#   research/e70_run_arch_probe.sh real applegpu_g17s      # both arms, in order
#
# 🔴 A forced architecture makes is_nax_available() return true on gen-16
# silicon that has no neural accelerators. A compile failure, wrong numbers, or
# a process abort are all valid outcomes and none of them is a bug. That is why
# the forced arm runs one cell per process: an uncaught C++ throw ends the
# process, and per-cell isolation turns a total loss into honest partial
# coverage. This arm is NEVER timed and never produces a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(($# >= 1)) || {
  echo "usage: e70_run_arch_probe.sh <real|applegpu_g17s|applegpu_g17g>..." >&2
  exit 2
}

cells=(
  qmv_m1 qmv_m5 qmv_m9
  qmm_m10 qmm_m511 qmm_m512
  sdpa_prefill_512
  sdpa_vector_q1_k768 sdpa_vector_q1_k1030
  sdpa_vector_q5_k768 sdpa_vector_q5_k1030
  dense_gemv_m1 dense_matmul_m511
)

mkdir -p research/out/e70-rung1

# Cmlx searches for mlx.metallib next to the RUNNING executable, and the xctest
# bundle is a third location that only exists after the tests are built. Build
# the tests first, then publish, then run.
build_log="research/out/e70-rung1/build.log"
if ! { swift build --build-tests --force-resolved-versions \
  && tools/build-mlx-metallib.sh --all-build-roots ; } > "${build_log}" 2>&1
then
  echo "e70_run_arch_probe: BUILD FAILED, log=${build_log}"
  tail -40 "${build_log}"
  exit 1
fi

export MLXFAST_E70_ARCH_PROBE=1

run_cell() {
  local arch="$1" out="$2" cell="$3"
  (
    if [[ "${arch}" != "real" ]]; then export MLX_METAL_GPU_ARCH="${arch}"; fi
    export MLXFAST_E70_CELL="${cell}"
    export MLXFAST_E70_OUT="${PWD}/${out}/${cell}.json"
    swift test --force-resolved-versions --filter E70ArchDispatchProbeTests
  ) > "${out}/${cell}.log" 2>&1
  local status=$?
  echo "e70_run_arch_probe: arch=${arch} cell=${cell} exit=${status}"
  # A non-zero exit under a forced architecture is a reportable datum, not a
  # failure of the probe, so record where it stopped and keep going.
  if ((status != 0)); then
    grep -E '^e70-probe:' "${out}/${cell}.log" | tail -2
    grep -iE 'error|exception|assert|abort|terminating' "${out}/${cell}.log" | tail -5
  fi
  printf '{"arch":"%s","cell":"%s","exit":%d}\n' "${arch}" "${cell}" "${status}" \
    >> "${out}/manifest.jsonl"
}

for arch in "$@"; do
  out="research/out/e70-rung1/${arch}"
  mkdir -p "${out}"
  : > "${out}/manifest.jsonl"
  if [[ "${arch}" == "real" ]]; then
    # The real architecture cannot abort on a nax pipeline, so one process
    # covers every cell.
    run_cell "${arch}" "${out}" all
  else
    for cell in "${cells[@]}"; do
      run_cell "${arch}" "${out}" "${cell}"
    done
  fi
  echo "e70_run_arch_probe: done arch=${arch} out=${out}"
  cat "${out}/manifest.jsonl"
done
exit 0
