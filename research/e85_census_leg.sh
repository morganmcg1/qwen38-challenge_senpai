#!/usr/bin/env bash
# Run one E85 census leg: forced draft width, dispatch + allocation census on.
#
#   usage: research/e85_census_leg.sh TAG DRAFTS TOKENS
#
# The census swizzle serialises every dispatch behind its own lock, so a census
# leg is NEVER a timing leg. It reports COUNTS only: dispatches, kernels,
# command-buffer commits and device buffer allocations per round, keyed to the
# round's forced draft width. Two legs at different widths give the per-draft
# slope of each count, which is the census target (c) deliverable and is immune
# to the phase-attribution smear that MLX's asynchronous encoding creates.
#
# The head is the DECLARED one. Under the setup default the declared rerank
# path is unreachable (`draftTokenIDWithDeclaredRerank` returns nil without
# `mtp.draft_lm_head.*`), so a census taken there would miss target (b) whole.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e85_census_leg.sh TAG DRAFTS TOKENS}"
drafts="${2:?usage: e85_census_leg.sh TAG DRAFTS TOKENS}"
tokens="${3:?usage: e85_census_leg.sh TAG DRAFTS TOKENS}"

head_dir="${E85_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e85_census_leg.sh: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
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
export MLX_E85_ALLOC_CENSUS=1
export MLX_E80_FORCE_DRAFTS="${drafts}"

{
  echo "tag=${tag}"
  echo "experiment=e85-materialised-intermediate-census"
  echo "forced_drafts=${drafts}"
  echo "tokens=${tokens}"
  echo "local_mode=--local-iterate"
  echo "census=1"
  echo "timing_valid=false"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
  echo "worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate \
  > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "census_records=$(wc -l < "${census_path}" | tr -d ' ')"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
