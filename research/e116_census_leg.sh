#!/usr/bin/env bash
# E116 -- ONE dispatch-census leg. Never a timing leg.
#
#   usage: research/e116_census_leg.sh TAG DRAFTS TOKENS DOSE [OPS_PER_BUFFER]
#
#   DRAFTS           an integer pins the verify width through
#                    MLX_E80_FORCE_DRAFTS; the literal `realised` leaves the
#                    variable unset so the shipped schedule chooses the width
#                    and the census reports the realised width histogram.
#                    Rung 4 needs `realised`; a per-width kernel rate needs a
#                    pinned width.
#   DOSE             MLX_E116_DOSE. `0` arms the instrument and allocates its
#                    100.27 MB weight without dispatching anything, so the
#                    dose census is a within-leg difference against the same
#                    resident memory.
#   OPS_PER_BUFFER   MLX_E58_BUFFER_LIMIT_OPS. `0` isolates one dispatch per
#                    command buffer, which makes `exclusive_kernels` an exact
#                    per-kernel GPU time; omit it for the in-situ leg whose
#                    concurrency is the one a real round has.
#
# WHY A SEPARATE SCRIPT AND NOT research/e96_census_leg.sh. That chain reaches
# `research/e85_census_leg.sh`, which exports MLX_E80_FORCE_DRAFTS
# unconditionally. Rung 4 must observe the widths the schedule actually
# chooses, so this leg has to be able to leave that variable unset.
#
# TIMING. The census swizzle takes a lock on every dispatch, so host wall clock
# in this process is invalid. Only Metal's GPU clock counts. The leg records
# timing_valid=false and is never a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e116_census_leg.sh TAG DRAFTS TOKENS DOSE [OPS_PER_BUFFER]}"
drafts="${2:?usage: e116_census_leg.sh TAG DRAFTS TOKENS DOSE [OPS_PER_BUFFER]}"
tokens="${3:?usage: e116_census_leg.sh TAG DRAFTS TOKENS DOSE [OPS_PER_BUFFER]}"
dose="${4:?usage: e116_census_leg.sh TAG DRAFTS TOKENS DOSE [OPS_PER_BUFFER]}"
ops_per_buffer="${5:-}"

head_dir="${E116_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e116_census_leg.sh: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
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
export MLXFAST_LOCAL_COOL_GATE=0

census_path="${PWD}/${out}/census.jsonl"
: > "${census_path}"
export MLX_E58_DISPATCH_CENSUS=1
export MLX_E58_DISPATCH_CENSUS_SHAPES=1
export MLX_E58_DISPATCH_CENSUS_PATH="${census_path}"
export MLX_E80_GPU_TIME=1
export MLX_E80_SNAPSHOT_ROUNDS="${MLX_E80_SNAPSHOT_ROUNDS:-1}"
if [[ -n "${ops_per_buffer}" ]]; then
  export MLX_E58_BUFFER_LIMIT_OPS="${ops_per_buffer}"
  export MLX_E58_BUFFER_LIMIT_MB="${MLX_E58_BUFFER_LIMIT_MB:-1}"
fi
if [[ "${drafts}" != "realised" ]]; then
  export MLX_E80_FORCE_DRAFTS="${drafts}"
fi
export MLX_E116_DOSE="${dose}"

trace_path="${PWD}/${out}/trace.txt"
: > "${trace_path}"
export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"

{
  echo "tag=${tag}"
  echo "experiment=e116-measured-transfer-from-kernel-percent-to-leg-seconds"
  echo "leg_kind=e116-dispatch-census"
  echo "harness=local"
  echo "forced_drafts=${drafts}"
  echo "tokens=${tokens}"
  echo "MLX_E116_DOSE=${dose}"
  echo "ops_per_buffer=${ops_per_buffer:-default}"
  echo "snapshot_rounds=${MLX_E80_SNAPSHOT_ROUNDS}"
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

{
  echo "census_records=${records}"
  echo "dose_trace_lines=$(
    grep -c '^mtp-trace: e116 dose ' "${trace_path}" 2>/dev/null || echo 0)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

# A leg that decodes and records nothing is the failure this catches: the
# census gate is compiled out of the worker, not the round is empty.
if [[ "${status}" -eq 0 && "${records}" -eq 0 ]]; then
  echo "e116_census_leg.sh: ${tag} exited 0 but wrote 0 census records to" \
       "${census_path}; MLX_E58_DISPATCH_CENSUS is absent from the worker." >&2
  exit 3
fi

exit "${status}"
