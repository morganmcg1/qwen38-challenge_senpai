#!/usr/bin/env bash
# One E90 leg: `research/e79_trace_leg.sh` plus the head-chain submission arm
# and, on request, the GPU interval ledger.
#
#   usage: research/e90_leg.sh TAG TOKENS [--arm NAME] [--intervals] [--sync-head]
#
#   --arm NAME    MLX_QWEN_MTP_HEAD_SUBMIT for this leg: ship, everyN or mid.
#                 Default ship, which is the shipped single submission.
#   --intervals   MLX_E90_GPU_INTERVALS=1. Records the GPU execution interval
#                 of every command buffer to research/out/TAG/gpu-intervals.jsonl.
#                 Use it for attribution legs only: it costs one lock and two
#                 clock reads per command buffer, which is small but not zero.
#   --sync-head   drain the head chain before the verify window (attribution).
#
# The leg builds nothing. Build and witness the worker before the session, not
# between legs of one session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e90_leg.sh TAG TOKENS [--arm NAME] [--intervals] [--sync-head]}"
tokens="${2:?usage: e90_leg.sh TAG TOKENS [--arm NAME] [--intervals] [--sync-head]}"
shift 2

arm="ship"
intervals=0
declare -a passthrough=()
while (($#)); do
  case "$1" in
    --arm) arm="$2"; shift 2 ;;
    --intervals) intervals=1; shift ;;
    --sync-head) passthrough+=(--sync-head); shift ;;
    --cool-gate) passthrough+=(--cool-gate); shift ;;
    *) echo "e90_leg.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

out="research/out/${tag}"
export MLX_QWEN_MTP_HEAD_SUBMIT="${arm}"
if ((intervals)); then
  export MLX_E90_GPU_INTERVALS=1
  export MLX_E90_GPU_INTERVALS_PATH="${PWD}/${out}/gpu-intervals.jsonl"
fi

research/e79_trace_leg.sh "${tag}" "${tokens}" "${passthrough[@]+"${passthrough[@]}"}"
status=$?

{
  echo "arm=${arm}"
  echo "gpu_intervals=${intervals}"
  echo "experiment=e90"
} >> "${out}/meta.txt"

exit "${status}"
