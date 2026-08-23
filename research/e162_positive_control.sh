#!/usr/bin/env bash
# E162 positive control: prove the exactness check can FAIL, and prove which
# source form the `quantized` family actually executes.
#
# The tree this runs against carries a deliberate one-character corruption in
# `Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp` ONLY: the
# pipelined k-loop starts at `cur = 1`, so iteration 0 consumes the half of Ws
# that has not been written yet and the half alternation stays inverted for the
# whole loop. The readable header `kernels/quantized.h` is untouched, and
# `mlx.metallib` is NOT rebuilt here, so the metallib keeps the CORRECT kernel.
#
# The wrapper's step-1 gate is the oracle. It runs `mlxfast-swift correctness`
# against `correctness_prompts/public_longcopy_gate_english_512_256.json`,
# which is CHECKED IN and generated on the organizer's M5 -- an external
# reference, not rows this build produced. The unperturbed base and arm A both
# pass it.
#
# Outcomes:
#   passed=false -> the exactness check can fail (control is live), the
#                   generated twin IS the runtime-effective source, and
#                   mlx.metallib does NOT govern this family.
#   passed=true  -> the twin is inert. Arm A would then be a no-op and every
#                   timing contrast from it is meaningless. Stop and report.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="research/out/e162/positive-control"
rm -rf "${out}"
mkdir -p "${out}"

swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker \
  > "${out}/build.log" 2>&1 || { echo "control: worker build failed"; tail -20 "${out}/build.log"; exit 1; }

eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?no MTP head}"

worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"
export MLXFAST_USE_RUNTIME_WORKER=1
export MLXFAST_RUNTIME_WORKER_EXECUTABLE="${worker}"
export MLXFAST_MLX_METALLIB="${PWD}/.build-worker/release/mlx.metallib"
export MLXFAST_NO_SANDBOX=1

{
  echo "worker=${worker}"
  echo "worker_sha256=$(shasum -a 256 "${worker}" | cut -d' ' -f1)"
  echo "witness_qmm_t_pipelined_k_loop=$(strings -a "${worker}" | grep -c -F 'qmm_t_pipelined_k_loop' || true)"
  echo "metallib_sha256=$(shasum -a 256 "${MLXFAST_MLX_METALLIB}" | cut -d' ' -f1)"
  echo "twin_sha256=$(shasum -a 256 Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp | cut -d' ' -f1)"
  echo "header_sha256=$(shasum -a 256 Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h | cut -d' ' -f1)"
} | tee "${out}/meta.txt"

.build/release/mlxfast-swift correctness \
  --weights weights \
  --golden correctness_prompts/public_longcopy_gate_english_512_256.json \
  > "${out}/correctness.json" 2> "${out}/correctness.err"
status=$?
echo "correctness_exit=${status}" | tee -a "${out}/meta.txt"

if ! jq -r '"passed=\(.passed) error=\(.error // "-")"' "${out}/correctness.json" 2>/dev/null \
    | tee -a "${out}/meta.txt"; then
  echo "control: no JSON object produced; stderr tail:" | tee -a "${out}/meta.txt"
  tail -10 "${out}/correctness.err" | tee -a "${out}/meta.txt"
fi
