#!/usr/bin/env bash
# E186 isolated decode-width kernel probes.
#
#   research/e186_session.sh <tag> [dispatch|probe|both]
#
# Runs the probe sections in the test process, never in the timed worker.
# harness=local, no thermal gate, no score. Entry and exit GPU temperature are
# recorded here and again per block inside the test.
set -uo pipefail

tag="${1:?usage: e186_session.sh <tag> [dispatch|probe|both]}"
section="${2:-both}"
out="research/out/e186/${tag}"
mkdir -p "${out}"

gpu_temp() {
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

# Cmlx loads mlx.metallib from beside the RUNNING executable, and relinking the
# test bundle wipes the copy a previous publish put there. Publish after the
# build, and fail closed when no artifact matches the current source
# fingerprint: a stale metallib would silently measure the wrong kernels.
publish_metallib() {
  local want source candidate bundle_dir
  want="$(tools/build-mlx-metallib.sh --print-fingerprint)" || return 1
  source=""
  for candidate in .build-worker/release/mlx.metallib \
                   .build/arm64-apple-macosx/release/mlx.metallib; do
    [[ -f "${candidate}" && -f "${candidate}.fingerprint" ]] || continue
    if grep -qF "${want}" "${candidate}.fingerprint"; then source="${candidate}"; break; fi
  done
  if [[ -z "${source}" ]]; then
    echo "e186_session.sh: no mlx.metallib matches fingerprint ${want}." >&2
    echo "e186_session.sh: run tools/build-mlx-metallib.sh --all-build-roots" >&2
    return 1
  fi
  while IFS= read -r bundle_dir; do
    if [[ -f "${bundle_dir}/mlx.metallib.fingerprint" ]] \
       && grep -qF "${want}" "${bundle_dir}/mlx.metallib.fingerprint"; then
      continue
    fi
    cp "${source}" "${bundle_dir}/mlx.metallib"
    cp "${source}.fingerprint" "${bundle_dir}/mlx.metallib.fingerprint"
    echo "e186_session.sh: published mlx.metallib into ${bundle_dir}"
  done < <(find .build .build-worker -path "*.xctest/Contents/MacOS" -type d 2>/dev/null)
}

swift build -c release --force-resolved-versions -Xswiftc -enable-testing \
  --build-tests || exit 1
publish_metallib || exit 1

{
  echo "tag=${tag}"
  echo "section=${section}"
  echo "experiment=e186-decode-width-kernel-probes"
  echo "harness=local"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "gpu_cores=$(ioreg -l 2>/dev/null \
    | LC_ALL=C sed -n 's/.*"gpu-core-count" = \([0-9][0-9]*\).*/\1/p' | head -1)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "os=$(sw_vers -productVersion)"
  echo "swift=$(swift --version 2>&1 | head -1)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "blocks=${MLXFAST_E186_BLOCKS:-<test-default>}"
  echo "warmup=${MLXFAST_E186_WARMUP:-<test-default>}"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

status=0
if [[ "${section}" == "dispatch" || "${section}" == "both" ]]; then
  MLXFAST_RUN_E186_DISPATCH=1 \
  MLXFAST_E186_DISPATCH_OUT="${PWD}/${out}/dispatch.json" \
    swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
      --filter E186DecodeWidthProbeTests 2>&1 | tee "${out}/dispatch.log"
  status=${PIPESTATUS[0]}
fi

if [[ "${status}" -eq 0 && ( "${section}" == "probe" || "${section}" == "both" ) ]]; then
  MLXFAST_RUN_E186_PROBE=1 \
  MLXFAST_E186_OUT="${PWD}/${out}/widths.json" \
    swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
      --filter E186DecodeWidthProbeTests 2>&1 | tee "${out}/probe.log"
  status=${PIPESTATUS[0]}
fi

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
