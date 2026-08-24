#!/usr/bin/env bash
# Research-only (qwen38-r1-e194-sdpa-step-removal): one ABBA-counterbalanced
# local session over the two E194 query-layout arms.
#
#   research/e194_session.sh ORDER TOKENS [SESSION_NAME]
#
#   ORDER   comma-separated arm list of 0/1, e.g. 0,1,1,0
#           0 = MLX_E194_SEQ_MAJOR_Q=0, the shipped `today` form
#           1 = MLX_E194_SEQ_MAJOR_Q=1, Route 2' (one sequence-major query
#               buffer, -1 dispatch, -1 barrier, -1 copy per FA layer)
#   TOKENS  decode tokens per leg (--local-iterate window)
#
# ONE BINARY, TWO ARMS. Both arms are compiled into the same worker and are
# selected at run time, so no leg can time a different build from another leg.
# The worker digest is asserted before and after every leg; a leg whose binary
# moved under it measures nothing. `MLX_` is the only prefix the worker
# forwards (QwenRuntimeWorker.swift:2626, FINDING 474 gate 1).
#
# BUILT-IN NULL CONTROL. The arm changes only the wide-decode (6 <= qL <= 9)
# branch, so the depth-0 serial leg of every run must be arm-independent. A
# serial split that tracks the MTP split falsifies the instrument, not the
# mechanism.
#
# PER-ROUND TIMES WITHOUT AN ADDED INSTRUMENT. The trusted CLI report already
# carries `block_request_seconds`, `decode_seconds` and `seed_prefill_seconds`;
# only the wrapper's score.json drops them. research/capture-cli.sh keeps each
# report after the measured phase ends (E181 precedent), so both arms are read
# by identical trusted timing code.
#
# MLXFAST_LOCAL_COOL_GATE=0 is the standing permitted local measurement mode.
# Entry and exit GPU temperature are recorded for every leg, the arms are
# ABBA-counterbalanced inside one session, and both gate fields are written
# verbatim as false. harness=local, no score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

order="${1:?usage: research/e194_session.sh ORDER TOKENS [NAME]}"
tokens="${2:?usage: research/e194_session.sh ORDER TOKENS [NAME]}"
name="${3:-$(date -u +%Y%m%dT%H%M%SZ)}"

repo_root="${PWD}"
session_out="${repo_root}/research/out/e194/session-${name}"
mkdir -p "${session_out}"

worker=".build-worker/release/mlxfast-runtime-worker"
[[ -x "${worker}" ]] || { echo "e194_session: no built worker at ${worker}" >&2; exit 2; }
certified="${MLX_E194_WORKER_SHA256:?e194_session: set MLX_E194_WORKER_SHA256 to the digest asserted by senpai/rebuild-and-assert-worker.sh}"

IFS=',' read -r -a legs <<<"${order}"
for arm in "${legs[@]}"; do
  case "${arm}" in 0|1) ;; *) echo "e194_session: arm must be 0 or 1, got '${arm}'" >&2; exit 2 ;; esac
done

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}
digest_of() { shasum -a 256 "${worker}" | cut -d' ' -f1; }

echo "e194_session: order=${order} tokens=${tokens} worker=$(digest_of)"

status=0
leg=0
for arm in "${legs[@]}"; do
  leg=$((leg + 1))
  out="${session_out}/leg$(printf '%02d' "${leg}")-arm${arm}"
  mkdir -p "${out}"

  before="$(digest_of)"
  if [[ "${before}" != "${certified}" ]]; then
    echo "e194_session: worker ${before} is not the certified build ${certified}" >&2
    status=1; break
  fi

  entry="$(gpu_temp)"
  echo "=== e194: leg ${leg}/${#legs[@]} arm=${arm} entry=${entry}C $(date -u +%H:%M:%SZ) ==="

  (
    export MLXFAST_LOCAL_COOL_GATE=0
    export MLX_E194_SEQ_MAJOR_Q="${arm}"
    unset MLX_QWEN_MTP_TRACE MLX_QWEN_MTP_TRACE_PATH
    export MLXFAST_SCORE_PATH="${out}/score.json"
    export MLXFAST_CAPTURE_DIR="${out}/reports"
    export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
    export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
    export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
    ./benchmark-qwen-mtp.sh --local-iterate
  ) > "${out}/run.log" 2>&1
  rc=$?
  exit_temp="$(gpu_temp)"
  after="$(digest_of)"

  {
    echo "e194_arm=${arm}"
    echo "e194_leg=${leg}"
    echo "e194_session_order=${order}"
    echo "mode=qwen-mtp-local-iterate"
    echo "tokens=${tokens}"
    echo "exit=${rc}"
    echo "gpu_temp_entry=${entry:-unavailable}"
    echo "gpu_temp_exit=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "phase_trace=0"
    echo "added_timing_instrument=none"
    echo "timing_source=trusted-parent-report"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  if [[ "${before}" != "${after}" ]]; then
    echo "e194_session: worker changed DURING leg ${leg}" >&2
    status=1; break
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "e194_session: leg ${leg} (arm ${arm}) exited ${rc}; see ${out}/run.log" >&2
    status=1; break
  fi
  jq -r '"leg '"${leg}"' arm='"${arm}"' mtp_spt=\(.metrics.mtp_seconds_per_token) serial_spt=\(.metrics.serial_seconds_per_token) ratio=\(.metrics.mtp_decode_speedup) edl=\(.metrics.effective_mean_draft_len) matched=\(.metrics.all_tokens_matched)"' \
    "${out}/score.json" 2>/dev/null || echo "leg ${leg}: no score.json summary"
done

echo "e194_session: artifacts in ${session_out}"
exit "${status}"
