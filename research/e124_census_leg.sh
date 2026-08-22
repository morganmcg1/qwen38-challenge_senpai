#!/usr/bin/env bash
# E124 stage 0.3 -- ONE dispatch-census leg under one island arm.
#
#   usage: research/e124_census_leg.sh TAG ARM TOKENS [SNAPSHOT_ROUNDS]
#
# This is NEVER a timing leg. The census swizzle takes a lock on every
# dispatch, so host wall clock in this process is invalid and the leg records
# timing_valid=false. Harness defect 7: MLX_E58_BUFFER_LIMIT_OPS is census-only
# and this script therefore never sets it on anything that reports a time.
#
# What the leg answers: how many dispatches, of which kernels, at which grid
# shapes, does one proposal step issue under this arm. Arm `all` should carry
# a narrowed Q affine-4 qmv plus two BF16 gemv matmuls plus one scatter; arm
# `none` should carry one wider QKV affine-4 qmv and none of the other three.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e124_census_leg.sh TAG ARM TOKENS [SNAPSHOT_ROUNDS]}"
arm="${2:?usage: e124_census_leg.sh TAG ARM TOKENS [SNAPSHOT_ROUNDS]}"
tokens="${3:?usage: e124_census_leg.sh TAG ARM TOKENS [SNAPSHOT_ROUNDS]}"
snapshot_rounds="${4:-64}"

case "${arm}" in
  all|none|q|kv) ;;
  *) echo "e124_census_leg.sh: unknown arm '${arm}'" >&2; exit 2 ;;
esac

head_dir="${E124_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e124_census_leg.sh: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_QWEN_MTP_ISLAND_ARM="${arm}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export MLXFAST_NO_SANDBOX=1
export MLXFAST_LOCAL_COOL_GATE=0

census_path="${PWD}/${out}/census.jsonl"
: > "${census_path}"
export MLX_E58_DISPATCH_CENSUS=1
export MLX_E58_DISPATCH_CENSUS_SHAPES=1
export MLX_E58_DISPATCH_CENSUS_PATH="${census_path}"
export MLX_E80_GPU_TIME=1
export MLX_E80_SNAPSHOT_ROUNDS="${snapshot_rounds}"

trace_path="${PWD}/${out}/trace.txt"
: > "${trace_path}"
export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"

{
  echo "tag=${tag}"
  echo "experiment=e124-noislands-acceptance-exchange"
  echo "leg_kind=e124-dispatch-census"
  echo "harness=local"
  echo "e124_arm=${arm}"
  echo "tokens=${tokens}"
  echo "snapshot_rounds=${snapshot_rounds}"
  echo "local_mode=--local-iterate"
  echo "census=1"
  echo "timing_valid=false"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate \
  > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

records="$(wc -l < "${census_path}" | tr -d ' ')"
witness="$(grep -h '^qwen-mtp-island-arm: ' "${out}/wrapper.err" 2>/dev/null \
  | sort -u | tr '\n' ';')"

{
  echo "census_records=${records}"
  echo "e124_arm_witness=${witness:-<absent>}"
  echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}" 2>/dev/null || echo 0)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

if [[ "${status}" -eq 0 && "${records}" -eq 0 ]]; then
  echo "e124_census_leg.sh: ${tag} exited 0 but wrote 0 census records;" \
       "MLX_E58_DISPATCH_CENSUS is absent from the worker." >&2
  exit 3
fi
if [[ "${status}" -eq 0 && "${witness}" != "qwen-mtp-island-arm: ${arm} "* ]]; then
  echo "e124_census_leg.sh: ${tag} requested arm ${arm} but the worker" \
       "reported '${witness:-<absent>}'" >&2
  exit 4
fi

exit "${status}"
