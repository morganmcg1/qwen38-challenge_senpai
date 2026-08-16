#!/usr/bin/env bash
# Research-only launcher for one --local-iterate arm.
#
# run_job takes an argv list with no environment field, so the per-arm env has
# to be established inside a script. Everything expensive still goes through
# ./benchmark-qwen-mtp.sh so the run lock, orphan check and 40C cool gate are
# never bypassed.
#
# usage: research/run-arm.sh ARM [--force-depth D] [--trace] [--tokens N]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: run-arm.sh ARM [--force-depth D] [--trace] [--tokens N]}"
shift

force_depth=""
trace=0
tokens="${MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS:-64}"
while (($#)); do
  case "$1" in
    --force-depth) force_depth="$2"; shift 2 ;;
    --trace) trace=1; shift ;;
    --tokens) tokens="$2"; shift 2 ;;
    *) echo "run-arm.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

out="research/out/${arm}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"

if [[ -n "${force_depth}" ]]; then
  export MLX_QWEN_MTP_FORCE_DEPTH="${force_depth}"
fi

if ((trace)); then
  # The generated worker sandbox denies file-write*, and the parent swallows
  # worker stderr, so a readable phase trace needs the documented local
  # relaxation. Refused on official runs by the trusted CLI.
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/trace.txt"
  export MLXFAST_NO_SANDBOX=1
fi

{
  echo "arm=${arm}"
  echo "tokens=${tokens}"
  echo "force_depth=${force_depth:-<adaptive>}"
  echo "trace=${trace}"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate
status=$?
echo "exit=${status}" >> "${out}/meta.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${out}/meta.txt"
exit "${status}"
