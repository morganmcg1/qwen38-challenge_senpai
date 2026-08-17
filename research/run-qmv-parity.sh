#!/usr/bin/env bash
# Research-only driver: digest the full `quantized_matmul` output at every
# scored shape and width, once per arm, so the arms can be compared bit-for-bit
# across builds.
#
#   research/run-qmv-parity.sh ARM=SHA [ARM=SHA ...]
#
# The cost curve's `row0_bitwise_matches_m1` only compares a build against
# itself. This rebuilds the runtime-effective twin per arm and digests the
# actual kernel output, which is what "bit-exact" has to mean.
#
# Untimed on purpose: one call per (shape, width), no repetitions, no cool gate.
# It still takes the local run lock so it never overlaps a model-holding run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

TWINS=(
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
)

LOCAL_RUN_LOCK_OWNED=""
local_run_guard_enabled() { [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]; }
run_lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
eval "${run_lock_definitions}"
for reused in local_run_lock_path acquire_local_run_lock release_local_run_lock \
              list_resident_model_processes abort_if_model_already_resident; do
  declare -F "${reused}" >/dev/null 2>&1 || {
    echo "run-qmv-parity.sh: could not reuse benchmark.sh's ${reused}(); refusing to run unguarded" >&2
    exit 1
  }
done

out_dir="${repo_root}/.mlxfast-private/qmv-parity"
rm -rf -- "${out_dir}"
mkdir -p "${out_dir}"

cleanup() {
  # Restore from HEAD, not the index: the per-arm `git checkout <sha> -- ...`
  # below rewrites the index too, so a plain `git checkout --` would "restore"
  # the tree to the last arm rather than to the commit under test.
  git checkout HEAD -- "${TWINS[@]}" 2>/dev/null || true
  release_local_run_lock
}
trap cleanup EXIT

acquire_local_run_lock
abort_if_model_already_resident

for spec in "$@"; do
  arm="${spec%%=*}"
  rhs="${spec##*=}"
  sha="${rhs%%+*}"
  patch=""
  if [[ "${rhs}" == *+* ]]; then
    patch="${rhs##*+}"
  fi
  echo "=== arm=${arm} sha=${sha} patch=${patch:-none} ===" >&2
  git checkout "${sha}" -- "${TWINS[@]}"
  if [[ -n "${patch}" ]]; then
    # A silently-skipped patch would compare a build against itself and report a
    # false bit-exact pass, so record the twin digests and require them to move.
    before="$(shasum -a 256 "${TWINS[@]}" | awk '{print $1}' | tr '\n' ' ')"
    "${MLXFAST_PYTHON_BIN:-python3}" research/roofline_arm_patch.py "${patch}"
    after="$(shasum -a 256 "${TWINS[@]}" | awk '{print $1}' | tr '\n' ' ')"
    if [[ "${before}" == "${after}" ]]; then
      echo "run-qmv-parity.sh: patch ${patch} left both twins unchanged" >&2
      exit 1
    fi
  fi
  shasum -a 256 "${TWINS[@]}" > "${out_dir}/${arm}.twins.txt"

  swift build -c release --build-tests --force-resolved-versions -Xswiftc -enable-testing
  tools/build-mlx-metallib.sh --all-build-roots

  MLXFAST_RUN_QMV_PARITY=1 \
  MLXFAST_QMV_PARITY_OUT="${out_dir}/${arm}.json" \
    swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
    --filter QwenQMVParityTests 2>&1 | tee "${out_dir}/${arm}.log"

  # `swift test --filter` exits 0 when the pattern matches nothing, so an
  # unwritten digest file is the only reliable "the test did not run" signal.
  [[ -s "${out_dir}/${arm}.json" ]] || {
    echo "run-qmv-parity.sh: arm ${arm} produced no digests" >&2
    exit 1
  }
done

git checkout HEAD -- "${TWINS[@]}"

# Command-line order, not glob order: the comparator treats its first file as
# the reference arm, and a glob would silently promote `armA` over `ref`.
ordered=()
for spec in "$@"; do ordered+=("${out_dir}/${spec%%=*}.json"); done
"${MLXFAST_PYTHON_BIN:-python3}" research/qmv_parity_compare.py "${ordered[@]}"
