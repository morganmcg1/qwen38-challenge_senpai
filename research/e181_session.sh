#!/usr/bin/env bash
# Research-only (qwen38-r1-e181-local-q-pair-jit-channel): run one
# ABBA-counterbalanced local session over the E181 arm trees.
#
#   research/e181_session.sh MODE ORDER TOKENS [SESSION_NAME]
#
#   MODE    submit   ./benchmark-qwen-mtp.sh --local-submit  (serial + MTP legs)
#           iterate  ./benchmark-qwen-mtp.sh --local-iterate (exactness gate)
#   ORDER   comma-separated arm list, e.g. P,PQ,PQ,P
#   TOKENS  decode tokens for the mode's token window
#
# Each arm is a SEPARATE built tree under research/out/e181/arm<ARM>, so one
# fixed binary per arm is timed in every leg it appears in. The worker digest
# recorded by research/e181_build_arm.sh is asserted before and after every
# leg: a leg whose binary moved under it measures nothing.
#
# PREFILL/DECODE SPLIT WITHOUT ANY ADDED INSTRUMENT. The trusted MTP driver
# already brackets the seed request inside the charged window
# (QwenRuntimeMTPDriver.swift:92-100) and the timed CLI report already carries
# `decode_seconds`, `seed_prefill_seconds` and the per-round
# `block_request_seconds` array (MLXFastCLI/main.swift:2008-2046). Only the
# wrapper's score.json drops them. research/capture-cli.sh keeps each report
# after the measured phase ends, so both arms are read by identical TRUSTED
# timing code and neither arm carries an added timing statement. The phase
# trace stays OFF: it perturbs round wall time and it is not needed.
#
#   prefill seconds       = seed_prefill_seconds
#   decode-only seconds   = decode_seconds - seed_prefill_seconds
#   decode-only s/token   = (decode_seconds - seed_prefill_seconds) / tokens
#
# MLXFAST_LOCAL_COOL_GATE=0 is the standing permitted local measurement mode.
# Entry and exit GPU temperature are recorded for every leg, the arms are
# ABBA-counterbalanced inside one session, and both gate fields are written
# verbatim as false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:?usage: research/e181_session.sh MODE ORDER TOKENS [NAME]}"
order="${2:?usage: research/e181_session.sh MODE ORDER TOKENS [NAME]}"
tokens="${3:?usage: research/e181_session.sh MODE ORDER TOKENS [NAME]}"
name="${4:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${mode}" in submit|iterate) ;; *) echo "e181_session: MODE must be submit or iterate" >&2; exit 2 ;; esac

repo_root="${PWD}"
arms_root="${repo_root}/research/out/e181"
session_out="${arms_root}/session-${mode}-${name}"
mkdir -p "${session_out}"

IFS=',' read -r -a legs <<<"${order}"
for arm in "${legs[@]}"; do
  [[ -x "${arms_root}/arm${arm}/benchmark-qwen-mtp.sh" ]] || {
    echo "e181_session: no built arm tree at ${arms_root}/arm${arm}" >&2; exit 2; }
done

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-/opt/homebrew/bin/macmon}"
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}
expected_digest() {
  awk -F= '/^worker_sha256=/ {print $2}' "${arms_root}/arm$1/research/e181-arm-$1-build.txt"
}
digest_of() { shasum -a 256 "${arms_root}/arm$1/.build-worker/release/mlxfast-runtime-worker" | cut -d' ' -f1; }

echo "e181_session: mode=${mode} order=${order} tokens=${tokens}"
for arm in P PQ PQS; do
  [[ -d "${arms_root}/arm${arm}" ]] || continue
  echo "e181_session: arm ${arm} worker $(digest_of "${arm}")"
done

status=0
leg=0
for arm in "${legs[@]}"; do
  leg=$((leg + 1))
  out="${session_out}/leg$(printf '%02d' "${leg}")-${arm}"
  mkdir -p "${out}"

  before="$(digest_of "${arm}")"
  want="$(expected_digest "${arm}")"
  if [[ "${before}" != "${want}" ]]; then
    echo "e181_session: arm ${arm} worker ${before} is not the certified build ${want}" >&2
    status=1; break
  fi

  entry="$(gpu_temp)"
  echo "=== e181: leg ${leg}/${#legs[@]} arm=${arm} entry=${entry}C ==="

  (
    cd "${arms_root}/arm${arm}" || exit 1
    export MLXFAST_LOCAL_COOL_GATE=0
    unset MLX_QWEN_MTP_TRACE MLX_QWEN_MTP_TRACE_PATH
    export MLXFAST_SCORE_PATH="${out}/score.json"
    export MLXFAST_CAPTURE_DIR="${out}/reports"
    export MLXFAST_CAPTURE_REAL_BIN="${arms_root}/arm${arm}/.build/release/mlxfast-swift"
    export MLXFAST_SWIFT_BIN="${arms_root}/arm${arm}/research/capture-cli.sh"
    if [[ "${mode}" == "submit" ]]; then
      export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${tokens}"
      ./benchmark-qwen-mtp.sh --local-submit
    else
      export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
      ./benchmark-qwen-mtp.sh --local-iterate
    fi
  ) > "${out}/run.log" 2>&1
  rc=$?
  exit_temp="$(gpu_temp)"
  after="$(digest_of "${arm}")"

  {
    echo "e181_arm=${arm}"
    echo "e181_leg=${leg}"
    echo "e181_session_order=${order}"
    echo "mode=qwen-mtp-local-${mode}"
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
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  if [[ "${before}" != "${after}" ]]; then
    echo "e181_session: arm ${arm} worker changed DURING leg ${leg}" >&2
    status=1; break
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "e181_session: leg ${leg} (${arm}) exited ${rc}; see ${out}/run.log" >&2
    status=1; break
  fi
done

echo "e181_session: artifacts in ${session_out}"
exit "${status}"
