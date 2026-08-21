#!/usr/bin/env bash
# Run one E85 census leg: forced draft width, dispatch + allocation census on.
#
#   usage: research/e85_census_leg.sh TAG DRAFTS TOKENS [ARM]
#
# ARM is one of base|a|b|ab and defaults to `ab`, matching the code defaults.
# Running the same census binary at both ARM settings is the execution proof:
# if the fused paths really run, the five gather and dequantize kernels leave
# the `draft_head` phase. A guard that fails silently would leave them there.
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

tag="${1:?usage: e85_census_leg.sh TAG DRAFTS TOKENS [ARM]}"
drafts="${2:?usage: e85_census_leg.sh TAG DRAFTS TOKENS [ARM]}"
tokens="${3:?usage: e85_census_leg.sh TAG DRAFTS TOKENS [ARM]}"
arm="${4:-ab}"

case "${arm}" in
  base) export MLX_E85_FUSED_EMBED=0 MLX_E85_GATHER_QMM=0 ;;
  a)    export MLX_E85_FUSED_EMBED=1 MLX_E85_GATHER_QMM=0 ;;
  b)    export MLX_E85_FUSED_EMBED=0 MLX_E85_GATHER_QMM=1 ;;
  ab)   export MLX_E85_FUSED_EMBED=1 MLX_E85_GATHER_QMM=1 ;;
  *) echo "e85_census_leg.sh: unknown arm ${arm}" >&2; exit 2 ;;
esac

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
  echo "arm=${arm}"
  echo "MLX_E85_FUSED_EMBED=${MLX_E85_FUSED_EMBED}"
  echo "MLX_E85_GATHER_QMM=${MLX_E85_GATHER_QMM}"
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

records="$(wc -l < "${census_path}" | tr -d ' ')"

{
  echo "census_records=${records}"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

# A census leg that decodes successfully but records nothing is the failure this
# assertion exists to catch. The gate is compiled out of the worker unless
# `Sources/MLXFastModel/E58DispatchCensus.swift` is present and installed, so a
# cleanup that removes the instrument leaves every caller exiting 0 with an
# empty census and a reducer that silently reports no roster at all.
if [[ "${status}" -eq 0 && "${records}" -eq 0 ]]; then
  echo "e85_census_leg.sh: leg ${tag} exited 0 but wrote 0 census records to" \
       "${census_path}; the MLX_E58_DISPATCH_CENSUS gate is absent from the" \
       "worker. Check that Sources/MLXFastModel/E58DispatchCensus.swift exists" \
       "and that installIfRequested() is called, then rebuild." >&2
  exit 3
fi

exit "${status}"
