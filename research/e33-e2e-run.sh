#!/bin/bash
# E33 deliverable (d): one end-to-end `--local-iterate` arm.
#
# Lives outside the checkout so the identical file can drive both the candidate
# branch and a detached base checkout at BASE_SHA, where a branch-only research
# script would not exist. A copy is committed as research/e33-e2e-run.sh.
#
# benchmark-qwen-mtp.sh does NOT rebuild mlx.metallib when the vendored Metal
# sources change -- it only forwards the CLI's warning. run-qmv-curve.sh:130
# rebuilds, benchmark.sh:1774 rebuilds, this path does not. Switching checkouts
# and running --local-iterate therefore measures whichever kernel was compiled
# last. Rebuild first, then fail closed if the warning still appears.
set -uo pipefail

TAG="${1:?usage: e33_e2e.sh TAG}"
repo_root="$(pwd)"
out_dir="${repo_root}/.mlxfast-private/e33-e2e/${TAG}"
mkdir -p "${out_dir}"

# Reuse benchmark.sh's own temperature seam rather than inventing a source.
eval "$(awk '/^find_macmon\(\) \{/,/^\}/' benchmark.sh)"
eval "$(awk '/^local_gpu_temp\(\) \{/,/^\}/' benchmark.sh)"
COOL_GATE_MACMON_BIN="$(find_macmon || true)"
export COOL_GATE_MACMON_BIN

# program.md permits MLXFAST_LOCAL_COOL_GATE=0 for local timed arms. This box
# idles above the 40C gate, so the real gate aborts rather than passes; the
# arms are counterbalanced and both gate-qualification flags stay false.
export MLXFAST_LOCAL_COOL_GATE=0
export MLXFAST_SCORE_PATH="${out_dir}/score.json"
unset MLXFAST_QWEN_MTP_HEAD_DIR

tools/build-mlx-metallib.sh --all-build-roots >"${out_dir}/metallib-build.log" 2>&1
build_rc=$?

{
  echo "e33-e2e: tag=${TAG}"
  echo "e33-e2e: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "e33-e2e: host=$(sysctl -n machdep.cpu.brand_string) mem=$(sysctl -n hw.memsize)"
  echo "e33-e2e: metallib_rebuild_rc=${build_rc}"
  echo "e33-e2e: metallib_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
  echo "e33-e2e: started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "gpu_temp_c_entry=$(local_gpu_temp || true)"
} | tee "${out_dir}/identity.txt"

if (( build_rc != 0 )); then
  echo "e33-e2e: metallib rebuild FAILED; refusing to time a stale kernel" | tee -a "${out_dir}/identity.txt"
  exit 90
fi

./benchmark-qwen-mtp.sh --local-iterate 2>&1 | tee "${out_dir}/run.log"
rc="${PIPESTATUS[0]}"

stale=$(grep -c 'built from different vendored Metal sources' "${out_dir}/run.log")
{
  echo "gpu_temp_c_exit=$(local_gpu_temp || true)"
  echo "stale_metallib_warnings=${stale}"
  echo "e33-e2e: finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}"
} | tee -a "${out_dir}/identity.txt"

if (( stale != 0 )); then
  echo "e33-e2e: ARM INVALID -- ${stale} stale-metallib warnings; timings measured the wrong kernel" \
    | tee -a "${out_dir}/identity.txt"
  exit 91
fi

exit "${rc}"
