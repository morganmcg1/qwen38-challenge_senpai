#!/usr/bin/env bash
# Research-only (e198-route1-row-amortized-kernel): run one
# ABBA-counterbalanced local session over the E198 arms.
#
#   research/e198_session.sh MODE ORDER TOKENS [SESSION_NAME]
#
#   MODE    submit   ./benchmark-qwen-mtp.sh --local-submit  (serial + MTP legs)
#           iterate  ./benchmark-qwen-mtp.sh --local-iterate (exactness gate)
#   ORDER   comma-separated arm list, e.g. OFF,FUSED,FUSED,OFF
#   TOKENS  decode tokens for the mode's token window
#
#   OFF     the shipped qL 6...9 split, unchanged
#   FUSED   one fused dispatch for every width in MLXFAST_E198_ROWS (default
#           6,7,8,9)
#
# ONE built tree serves both arms. The arm is an environment variable that
# `FusedRowAmortizedSDPA.enabledRows` reads once per worker process, so the
# binary is byte-identical in every leg. The session asserts that single
# certified worker digest before and after every leg: a leg whose binary moved
# under it measures nothing. RULE 384.
#
# MLXFAST_LOCAL_COOL_GATE=0 is the standing permitted local measurement mode.
# Entry and exit GPU temperature are recorded for every leg, the arms are
# ABBA-counterbalanced inside one session, and both gate fields are written
# verbatim as false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:?usage: research/e198_session.sh MODE ORDER TOKENS [NAME]}"
order="${2:?usage: research/e198_session.sh MODE ORDER TOKENS [NAME]}"
tokens="${3:?usage: research/e198_session.sh MODE ORDER TOKENS [NAME]}"
name="${4:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${mode}" in submit|iterate) ;; *) echo "e198_session: MODE must be submit or iterate" >&2; exit 2 ;; esac

rows="${MLXFAST_E198_ROWS:-6,7,8,9}"
repo_root="${PWD}"
session_out="${repo_root}/research/out/e198/session-${mode}-${name}"
mkdir -p "${session_out}"

build_record="research/e198-build.txt"
[[ -s "${build_record}" ]] || { echo "e198_session: run research/e198_build.sh first" >&2; exit 2; }
worker=".build-worker/release/mlxfast-runtime-worker"
want_digest="$(awk -F= '/^worker_sha256=/ {print $2}' "${build_record}")"
digest_of() { shasum -a 256 "${worker}" | cut -d' ' -f1; }

# HEAD IDENTITY (RULE 389, HARNESS DEFECT 44). `setup-qwen-mtp.sh` only ever
# provisions the ORGANIZER-PINNED head; the ranked workflow hands the head that
# `mtp-head.manifest.json` DECLARES to the candidate leg. Both arms here must
# load that declared head, so resolve it explicitly and never let the wrapper
# fall back to the pinned cache.
export MLXFAST_QWEN_MTP_HEAD_DIR="${MLXFAST_QWEN_MTP_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${MLXFAST_QWEN_MTP_HEAD_DIR}/model.safetensors" ]] || {
  echo "e198_session: declared head missing at ${MLXFAST_QWEN_MTP_HEAD_DIR}; run research/fetch-declared-head.sh" >&2
  exit 2
}
head_provenance_sha256="$(
  python3 research/fb7_head_provenance.py --head-dir "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["tree_digest_sha256"])'
)"

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["temp"]["gpu_temp_avg"])' 2>/dev/null
}

IFS=',' read -r -a legs <<<"${order}"
for arm in "${legs[@]}"; do
  case "${arm}" in OFF|FUSED) ;; *) echo "e198_session: unknown arm ${arm}" >&2; exit 2 ;; esac
done

echo "e198_session: mode=${mode} order=${order} tokens=${tokens} rows=${rows}"
echo "e198_session: worker $(digest_of) (certified ${want_digest})"
echo "e198_session: head_provenance_sha256=${head_provenance_sha256}"

status=0
leg=0
for arm in "${legs[@]}"; do
  leg=$((leg + 1))
  out="${session_out}/leg$(printf '%02d' "${leg}")-${arm}"
  mkdir -p "${out}"

  before="$(digest_of)"
  if [[ "${before}" != "${want_digest}" ]]; then
    echo "e198_session: worker ${before} is not the certified build ${want_digest}" >&2
    status=1; break
  fi

  entry="$(gpu_temp)"
  echo "=== e198: leg ${leg}/${#legs[@]} arm=${arm} entry=${entry}C ==="

  (
    export MLXFAST_LOCAL_COOL_GATE=0
    unset MLX_QWEN_MTP_TRACE MLX_QWEN_MTP_TRACE_PATH
    if [[ "${arm}" == "FUSED" ]]; then
      export MLXFAST_QWEN_FUSED_SDPA_ROWS="${rows}"
    else
      unset MLXFAST_QWEN_FUSED_SDPA_ROWS
    fi
    export MLXFAST_SCORE_PATH="${out}/score.json"
    export MLXFAST_CAPTURE_DIR="${out}/reports"
    export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
    export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
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
  after="$(digest_of)"

  {
    echo "e198_arm=${arm}"
    echo "e198_rows=$([[ "${arm}" == "FUSED" ]] && echo "${rows}" || echo none)"
    echo "e198_leg=${leg}"
    echo "e198_session_order=${order}"
    echo "mode=qwen-mtp-local-${mode}"
    echo "tokens=${tokens}"
    echo "exit=${rc}"
    echo "gpu_temp_entry=${entry:-unavailable}"
    echo "gpu_temp_exit=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "head_provenance_sha256=${head_provenance_sha256}"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "phase_trace=0"
    echo "added_timing_instrument=none"
    echo "timing_source=trusted-parent-report"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  if [[ "${before}" != "${after}" ]]; then
    echo "e198_session: worker changed DURING leg ${leg}" >&2
    status=1; break
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "e198_session: leg ${leg} (${arm}) exited ${rc}; see ${out}/run.log" >&2
    status=1; break
  fi
done

echo "e198_session: artifacts in ${session_out}"
exit "${status}"
