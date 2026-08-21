#!/usr/bin/env bash
# One E92 rung 1 leg: the in-session read-bandwidth residency sweep.
#
#   usage: research/e92_bw_leg.sh TAG TOKENS [SIZES_MB] [REPS]
#
# The probe fires once per worker process, right after the seed prefill of the
# first session, and writes one JSON line per repetition to
# research/out/TAG/e92-bandwidth.jsonl. The GPU interval ledger runs in the
# same process, so `research/e92_bandwidth.py` can convert each recorded
# window into device time.
#
# The leg builds nothing. Build and witness the worker before the session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e92_bw_leg.sh TAG TOKENS [SIZES_MB] [REPS]}"
tokens="${2:?usage: e92_bw_leg.sh TAG TOKENS [SIZES_MB] [REPS]}"
sizes="${3:-16,64,157,330,428,1024}"
reps="${4:-7}"

out="research/out/${tag}"
export MLX_E92_BANDWIDTH=1
export MLX_E92_BANDWIDTH_SIZES_MB="${sizes}"
export MLX_E92_BANDWIDTH_REPS="${reps}"
export MLX_E92_BANDWIDTH_PATH="${PWD}/${out}/e92-bandwidth.jsonl"

research/e90_leg.sh "${tag}" "${tokens}" --intervals
status=$?

{
  echo "e92_bandwidth_sizes_mb=${sizes}"
  echo "e92_bandwidth_reps=${reps}"
  echo "experiment=e92-rung1"
} >> "${out}/meta.txt"

exit "${status}"
