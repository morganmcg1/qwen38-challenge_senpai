#!/usr/bin/env bash
# Research-only launcher for one --local-iterate arm.
#
# run_job takes an argv list with no environment field, so the per-arm env has
# to be established inside a script. Everything expensive still goes through
# ./benchmark-qwen-mtp.sh so the run lock, orphan check and 40C cool gate are
# never bypassed.
#
# usage: research/run-arm.sh ARM [--trace] [--tokens N]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: run-arm.sh ARM [--trace] [--tokens N]}"
shift

trace=0
tokens="${MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS:-64}"
while (($#)); do
  case "$1" in
    --trace) trace=1; shift ;;
    --tokens) tokens="$2"; shift 2 ;;
    --head-dir) export MLXFAST_QWEN_MTP_HEAD_DIR="$2"; shift 2 ;;
    --h-vector) export MLX_QWEN_MTP_H_VECTOR="$2"; shift 2 ;;
    *) echo "run-arm.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

out="research/out/${arm}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"

if ((trace)); then
  # The generated worker sandbox denies file-write*, and the parent swallows
  # worker stderr, so a readable phase trace needs the documented local
  # relaxation. Refused on official runs by the trusted CLI.
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/trace.txt"
  export MLXFAST_NO_SANDBOX=1
fi

# A previous job can leave a metallib built from another arm's patched sources
# in every build root, and nothing downstream of here rebuilds it.
tools/build-mlx-metallib.sh --all-build-roots

{
  echo "arm=${arm}"
  echo "tokens=${tokens}"
  echo "trace=${trace}"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "h_vector=${MLX_QWEN_MTP_H_VECTOR:-<measured-default>}"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

if ((trace)); then
  # The session writes trace lines to stderr only; MLX_QWEN_MTP_TRACE_PATH is
  # not read by any source, so the wrapper is what makes that path real.
  ./benchmark-qwen-mtp.sh --local-iterate 2>"${MLX_QWEN_MTP_TRACE_PATH}"
else
  ./benchmark-qwen-mtp.sh --local-iterate
fi
status=$?
echo "exit=${status}" >> "${out}/meta.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${out}/meta.txt"
exit "${status}"
