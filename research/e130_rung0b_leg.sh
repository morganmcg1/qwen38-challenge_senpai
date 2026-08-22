#!/usr/bin/env bash
# One E130 rung 0b census leg: research/e79_trace_leg.sh plus the E58 dispatch
# census and the E80 GPU-time ledger.
#
#   usage: research/e130_rung0b_leg.sh TAG TOKENS [--force-drafts N]
#
# The census counts dispatches and records the dispatch grid, so `ntg.x` is read
# directly from `grid=WxHxD`. It locks on every dispatch, bind and barrier, so
# THIS LEG IS NOT A TIMING LEG. Its wall time is inflated by construction and is
# never reported as candidate time. Only counts and GPU-side intervals are used.
#
# `MLX_E120_QMV_ARM` is deliberately left unset, so Route B runs its shipped
# `sumtable` arm and the census reads the SHIPPED configuration, which is what
# rung 0b asks about.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e130_rung0b_leg.sh TAG TOKENS [--force-drafts N]}"
tokens="${2:?usage: e130_rung0b_leg.sh TAG TOKENS [--force-drafts N]}"
shift 2

while (($#)); do
  case "$1" in
    --force-drafts) export MLX_E80_FORCE_DRAFTS="$2"; shift 2 ;;
    *) echo "e130_rung0b_leg.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

out="research/out/${tag}"
export MLX_E58_DISPATCH_CENSUS=1
export MLX_E58_DISPATCH_CENSUS_SHAPES=1
export MLX_E80_GPU_TIME=1
export MLX_E80_SNAPSHOT_ROUNDS="${MLX_E80_SNAPSHOT_ROUNDS:-16}"
export MLX_E58_DISPATCH_CENSUS_PATH="${PWD}/${out}/census.jsonl"

mkdir -p "${out}"
research/e79_trace_leg.sh "${tag}" "${tokens}"
status=$?

{
  echo "experiment=e130-rung0b"
  echo "instrument=e58-dispatch-census+e80-gputime"
  echo "instrument_is_timing_safe=false"
  echo "qmv_arm=${MLX_E120_QMV_ARM:-sumtable (shipped default, unset)}"
  echo "force_drafts=${MLX_E80_FORCE_DRAFTS:-<unset>}"
  echo "census_lines=$(wc -l < "${out}/census.jsonl" 2>/dev/null | tr -d ' ')"
} >> "${out}/meta.txt"

exit "${status}"
