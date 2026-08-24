#!/usr/bin/env bash
# Research-only (qwen38-r1-e168, revision r1): one 512-token --local-submit
# confirmation leg on the organizer-pure cap-4 ship tree.
#
#   research/e168_r1_confirm.sh
#
# This is the ladder's confirmation step, not a screen: real 40 C gate, default
# sandbox, no trace, no head override, exact post-EOS continuation and
# row-ledger closure checked on the timed pass. It records the identity tuple
# beside the score so the leg can never be read as belonging to another build.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="research/out/e174"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"

gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

{
  echo "tag=e174-r1-cap4-organizer-pure-confirm"
  echo "experiment=e168-r1-depth-cap-4-organizer-pure"
  echo "harness=local"
  echo "tokens=512"
  echo "local_mode=--local-submit"
  echo "sandbox=on"
  echo "trace=0"
  echo "cool_gate=1"
  echo "cool_gate_passed_real_gate=true"
  echo "gate_qualified_for_timing=true"
  echo "official_or_ranked_score=false"
  # Sampled BEFORE the benchmark runs its own 40 C gate, so this is a pre-gate
  # reading and not the temperature at which timing started. Disclosed in the
  # E168 r0 result and kept explicit here.
  echo "gpu_temp_entry_is_pre_gate=true"
  echo "candidate_sha=$(git rev-parse HEAD)"
  echo "organizer_sha=$(git rev-parse upstream/main)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift mtp-head.manifest.json \
      | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "metallib_sha256=$(
    shasum -a 256 .build-worker/release/mlx.metallib | awk '{print $1}')"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-submit \
  > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "post_run_worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "post_run_cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "post_run_candidate_sha=$(git rev-parse HEAD)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
