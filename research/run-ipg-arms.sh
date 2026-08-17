#!/usr/bin/env bash
# Research-only driver for the IPG weight-pass arms (E14).
#
#   research/run-ipg-arms.sh TAG:ARM [TAG:ARM ...]
#
# ARM is `ref` (both quantized twins exactly as committed) or an arm name known
# to research/roofline_arm_patch.py. Each spec restores both twins from HEAD,
# applies its arm, records a freshness + digest receipt, and runs one full
# research/run-qmv-curve.sh sweep. The twins are always restored on exit, so a
# cancelled job never leaves a patched worktree behind.
#
# The stock pip-MLX leg is skipped on purpose: it links an unchanged binary, so
# it cannot see a vendored-kernel arm and only costs wall clock.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

readonly TWIN_H="Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
readonly TWIN_CPP="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

restore_twins() {
  git checkout -- "${TWIN_H}" "${TWIN_CPP}"
}
trap restore_twins EXIT

if ! git diff --quiet -- "${TWIN_H}" "${TWIN_CPP}"; then
  echo "run-ipg-arms.sh: quantized twins are already dirty; refusing to start" >&2
  exit 1
fi

digest() { shasum -a 256 "$1" | cut -d' ' -f1; }
mtime() { stat -f %Sm -t '%Y-%m-%dT%H:%M:%SZ' "$1" 2>/dev/null || echo "missing"; }

for spec in "$@"; do
  tag="${spec%%:*}"
  arm="${spec##*:}"
  if [[ -z "${tag}" || -z "${arm}" || "${tag}" == "${spec}" ]]; then
    echo "run-ipg-arms.sh: bad spec '${spec}', want TAG:ARM" >&2
    exit 1
  fi

  restore_twins
  if [[ "${arm}" != "ref" ]]; then
    python3 research/roofline_arm_patch.py "${arm}"
  fi

  out_dir="${repo_root}/.mlxfast-private/ipg-arms/${tag}"
  mkdir -p "${out_dir}"
  {
    echo "{"
    printf '  "tag": "%s",\n' "${tag}"
    printf '  "arm": "%s",\n' "${arm}"
    printf '  "head_sha": "%s",\n' "$(git rev-parse HEAD)"
    printf '  "started_utc": "%s",\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '  "vendored_source_fingerprint": "%s",\n' \
      "$(tools/build-mlx-metallib.sh --print-fingerprint)"
    printf '  "quantized_h_sha256": "%s",\n' "$(digest "${TWIN_H}")"
    printf '  "quantized_cpp_sha256": "%s",\n' "$(digest "${TWIN_CPP}")"
    printf '  "metallib_release_mtime_before": "%s",\n' \
      "$(mtime .build/arm64-apple-macosx/release/mlx.metallib)"
    printf '  "xctest_metallib_mtime_before": "%s",\n' \
      "$(mtime .build/arm64-apple-macosx/release/mlxfast-challenge-devPackageTests.xctest/Contents/MacOS/mlx.metallib)"
    printf '  "worker_binary_mtime_before": "%s"\n' \
      "$(mtime .build-worker/release/mlxfast-runtime-worker)"
    echo "}"
  } >"${out_dir}/arm-state.json"
  cat "${out_dir}/arm-state.json" >&2

  MLXFAST_MLX_PYTHON_BIN=/usr/bin/python3 \
    research/run-qmv-curve.sh "${tag}"

  {
    echo "finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "vendored_source_fingerprint_after=$(tools/build-mlx-metallib.sh --print-fingerprint)"
    for f in .build/arm64-apple-macosx/release/mlx.metallib \
             .build/arm64-apple-macosx/release/mlxfast-challenge-devPackageTests.xctest/Contents/MacOS/mlx.metallib
    do
      echo "after ${f} mtime=$(mtime "${f}") sha256=$(digest "${f}")"
    done
  } >"${out_dir}/arm-state-after.txt"
  cat "${out_dir}/arm-state-after.txt" >&2
done

restore_twins
echo "run-ipg-arms.sh: done, twins restored" >&2
