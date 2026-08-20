#!/usr/bin/env bash
# Run one traced E65 leg of ./benchmark-qwen-mtp.sh.
#
#   usage: research/e65_run_leg.sh TAG ARM TOKENS [options]
#
#   --sync-head   MLX_QWEN_MTP_TRACE_SYNC_HEAD=1. Attribution only: it drains
#                 the head chain before the verify-build window, so a head-chain
#                 GPU stall moves out of verify_build_us. It destroys the
#                 head/verify overlap, so never use it in a timed contrast.
#   --no-trace    run without the phase trace (timed confirmation legs)
#   --submit      use --local-submit instead of --local-iterate
#   --label TEXT  free-text arm label carried into meta.txt and W&B
#
# The arm binaries must already be staged by research/e65_build_arm.sh. The
# wrapper never runs `swift build` itself (it returns before benchmark.sh:1773),
# so a staged worker survives the run; only the metallib guard can rebuild, and
# no E65 arm touches Metal source.
#
# MLXFAST_LOCAL_COOL_GATE=0 is set. Sessions are position-balanced, entry and
# exit GPU temperature are recorded per leg, and every result carries
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e65_run_leg.sh TAG ARM TOKENS [options]}"
arm="${2:?usage: e65_run_leg.sh TAG ARM TOKENS [options]}"
tokens="${3:?usage: e65_run_leg.sh TAG ARM TOKENS [options]}"
shift 3

trace=1
sync_head=0
mode=--local-iterate
label=
while (($#)); do
  case "$1" in
    --sync-head) sync_head=1; shift ;;
    --no-trace) trace=0; shift ;;
    --submit) mode=--local-submit; shift ;;
    --label) label="$2"; shift 2 ;;
    *) echo "e65_run_leg.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

bin_dir="research/out/e65/bin/${arm}"
for product in mlxfast-swift mlxfast-runtime-worker; do
  if [[ ! -x "${bin_dir}/${product}" ]]; then
    echo "e65: missing staged ${product} for arm ${arm}" >&2
    exit 1
  fi
done

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

cp "${bin_dir}/mlxfast-swift" .build/release/mlxfast-swift
cp "${bin_dir}/mlxfast-runtime-worker" .build-worker/release/mlxfast-runtime-worker

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export MLXFAST_LOCAL_COOL_GATE=0

trace_path="${PWD}/${out}/trace.txt"
if ((trace)); then
  export MLXFAST_NO_SANDBOX=1
  : > "${trace_path}"
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"
  ((sync_head)) && export MLX_QWEN_MTP_TRACE_SYNC_HEAD=1
fi

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
  echo "arm=${arm}"
  echo "label=${label:-${arm}}"
  echo "tokens=${tokens}"
  echo "local_mode=${mode}"
  echo "trace=${trace}"
  echo "sync_head=${sync_head}"
  echo "cool_gate=0"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "staged_cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "staged_worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  cat "${bin_dir}/provenance.txt"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh "${mode}" > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"
exit "${status}"
