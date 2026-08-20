#!/usr/bin/env bash
# E80 rung 1 -- run the E71 width-tax census with the GPU-time instrument on.
#
#   usage: research/e80_gate_session.sh TAG [smoke|full]
#
# The E71 driver owns the local run lock, the ABBA order and the per-block
# thermal record. This wrapper only adds the E80 environment, because run_job
# takes an argv list with no environment field.
#
# The GPU ledger writes JSONL next to the E71 census JSON. Snapshots are
# emitted per BLOCK (`endWindow`), so `MLX_E80_SNAPSHOT_ROUNDS` is set far above
# the per-block rep count on purpose: a rep-triggered snapshot would split one
# block across two records.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e80_gate_session.sh TAG [smoke|full]}"
profile="${2:-full}"

gpu_out="research/out/${tag}-gpu"
rm -rf "${gpu_out}"
mkdir -p "${gpu_out}"

export MLX_E80_GPU_TIME=1
export MLX_E80_SNAPSHOT_ROUNDS=1000000
export MLX_E58_DISPATCH_CENSUS=1
export MLX_E58_DISPATCH_CENSUS_SHAPES=1
export MLX_E58_DISPATCH_CENSUS_PATH="${PWD}/${gpu_out}/census.jsonl"
: > "${MLX_E58_DISPATCH_CENSUS_PATH}"

# Isolated mode forces one MLX op per command buffer, so a command-buffer GPU
# interval is one kernel's GPU time. Leave unset for the in-situ arm.
if [[ -n "${E80_OPS_PER_BUFFER:-}" ]]; then
  export MLX_E58_BUFFER_LIMIT_OPS="${E80_OPS_PER_BUFFER}"
fi

# --- publish mlx.metallib into the release xctest bundle ---------------------
# Cmlx searches next to the RUNNING executable. `swift test -c release` recreates
# the bundle whenever the test target relinks, and that wipes the metallib the
# previous `tools/build-mlx-metallib.sh --all-build-roots` published there. The
# first MLXArray then dies with "Failed to load the default metallib", which is
# the exact gap that flag's comment describes. e71_census.sh does not publish, so
# this wrapper does it.
#
# The metallib is compiled AOT from vendored Metal sources and takes minutes. A
# rebuild is pure waste when those sources have not moved, so publish the
# existing artifact and let the fingerprint sidecar prove it is current. A
# mismatch is fatal: a stale metallib would silently measure the wrong kernels.
publish_metallib() {
  local want source bundle_dir
  want="$(tools/build-mlx-metallib.sh --print-fingerprint)" || return 1

  source=""
  local candidate
  for candidate in .build-worker/release/mlx.metallib \
                   .build/arm64-apple-macosx/release/mlx.metallib; do
    [[ -f "${candidate}" && -f "${candidate}.fingerprint" ]] || continue
    if grep -qF "${want}" "${candidate}.fingerprint"; then source="${candidate}"; break; fi
  done
  if [[ -z "${source}" ]]; then
    echo "e80_gate_session.sh: no mlx.metallib matches fingerprint ${want}." >&2
    echo "e80_gate_session.sh: run tools/build-mlx-metallib.sh --all-build-roots" >&2
    return 1
  fi

  while IFS= read -r bundle_dir; do
    if [[ -f "${bundle_dir}/mlx.metallib.fingerprint" ]] \
       && grep -qF "${want}" "${bundle_dir}/mlx.metallib.fingerprint"; then
      continue
    fi
    cp "${source}" "${bundle_dir}/mlx.metallib"
    cp "${source}.fingerprint" "${bundle_dir}/mlx.metallib.fingerprint"
    echo "e80_gate_session.sh: published mlx.metallib into ${bundle_dir}"
  done < <(find .build .build-worker -path "*.xctest/Contents/MacOS" -type d 2>/dev/null)
}

# Relink the test bundle first, so the publish lands in the bundle the session
# will actually load. Publishing before the relink would be undone by it.
swift build -c release --force-resolved-versions -Xswiftc -enable-testing \
  --build-tests || exit 1
publish_metallib || exit 1

exec research/e71_census.sh "${tag}" "${profile}"
