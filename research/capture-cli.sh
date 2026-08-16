#!/usr/bin/env bash
# Research-only passthrough for MLXFAST_SWIFT_BIN.
#
# benchmark-qwen-mtp.sh writes each CLI report into a scratch run directory that
# its EXIT trap deletes, so the per-round `block_request_seconds` array and
# `decode_seconds` never survive a run. Pointing MLXFAST_SWIFT_BIN at this
# script keeps a copy of every stdout report without editing the trusted
# harness, the trusted CLI, or the timing window: argv is forwarded unchanged
# and the JSON is emitted once, after the measured phase has already ended.
set -uo pipefail

real="${MLXFAST_CAPTURE_REAL_BIN:?research/capture-cli.sh: set MLXFAST_CAPTURE_REAL_BIN}"
keep="${MLXFAST_CAPTURE_DIR:?research/capture-cli.sh: set MLXFAST_CAPTURE_DIR}"
mkdir -p "${keep}"

verb="${1:-noverb}"
seq_file="${keep}/.seq"
seq=$(( $(cat "${seq_file}" 2>/dev/null || echo 0) + 1 ))
printf '%s' "${seq}" > "${seq_file}"

"${real}" "$@" | tee "${keep}/$(printf '%02d' "${seq}")-${verb//\//_}.json"
exit "${PIPESTATUS[0]}"
