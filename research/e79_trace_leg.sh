#!/usr/bin/env bash
# Run one traced, UNCHANGED-base leg of ./benchmark-qwen-mtp.sh for E79.
#
#   usage: research/e79_trace_leg.sh TAG TOKENS [--sync-head] [--cool-gate]
#
#   --sync-head  MLX_QWEN_MTP_TRACE_SYNC_HEAD=1. Drains the head chain before
#                the verify-build window, so head-chain GPU time moves out of
#                verify_build_us and into draft_build_us. Attribution only.
#   --cool-gate  keep the real 40 C gate (MLXFAST_LOCAL_COOL_GATE unset).
#                Without it the gate is disabled and the leg is labelled
#                gate_qualified_for_timing=false.
#
# The leg stages nothing and builds nothing. It reads the binaries already in
# .build/release and .build-worker/release and records their digests, so a leg
# that timed some other build is visible in meta.txt.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e79_trace_leg.sh TAG TOKENS [--sync-head] [--cool-gate]}"
tokens="${2:?usage: e79_trace_leg.sh TAG TOKENS [--sync-head] [--cool-gate]}"
shift 2

sync_head=0
cool_gate=0
while (($#)); do
  case "$1" in
    --sync-head) sync_head=1; shift ;;
    --cool-gate) cool_gate=1; shift ;;
    *) echo "e79_trace_leg.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

# `setup-qwen-mtp.sh` provisions only the ORGANIZER-PINNED head and
# `benchmark-qwen-mtp.sh` takes that as its default, but the ranked candidate
# leg runs the head `mtp-head.manifest.json` DECLARES. Default to the declared
# run tree that `research/fetch-declared-head.sh` stages, and keep the pinned
# head reachable through `E79_HEAD_DIR` as an explicit head-variant arm.
head_dir="${E79_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e79_trace_leg.sh: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export MLXFAST_NO_SANDBOX=1
((cool_gate)) || export MLXFAST_LOCAL_COOL_GATE=0

trace_path="${PWD}/${out}/trace.txt"
: > "${trace_path}"
export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"
((sync_head)) && export MLX_QWEN_MTP_TRACE_SYNC_HEAD=1

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
  echo "tag=${tag}"
  echo "experiment=e79-head-economics-census"
  echo "harness=local"
  echo "tokens=${tokens}"
  echo "local_mode=--local-iterate"
  echo "sync_head=${sync_head}"
  echo "cool_gate=${cool_gate}"
  if ((cool_gate)); then
    echo "cool_gate_passed_real_gate=true"
    echo "gate_qualified_for_timing=true"
  else
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
  fi
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "ladder=${MLX_QWEN_MTP_LADDER:-<unset>}"
  echo "worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate \
  > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}" || true)"
  echo "post_run_worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "post_run_cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
