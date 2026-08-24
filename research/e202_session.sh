#!/usr/bin/env bash
# Research-only (E202): run one in-process arm-switched local session over the
# split-cell eval() barrier arms.
#
#   research/e202_session.sh MODE OFFSETS TOKENS [SESSION_NAME]
#
#   MODE     submit   ./benchmark-qwen-mtp.sh --local-submit
#            iterate  ./benchmark-qwen-mtp.sh --local-iterate
#   OFFSETS  comma-separated per-leg arm-rotation offsets, e.g. 0,1,2,2,1,0
#            `-` means "no arm rotation": the shipped branch in every round.
#   TOKENS   decode tokens for the mode's token window
#
# THE ARMS DO NOT SEPARATE THE LEGS. Every leg rotates through all three arms
# at the round boundary (SHIPPED / BARRIER-LAST / BARRIER-ALL), so each leg
# carries the same arm composition and the arms share one thermal state, one
# token stream and one binary (RULE 388). The offset only shifts WHICH round
# index gets which arm; running every offset the same number of times gives a
# Latin square in (round index x arm), so the per-round width sequence — which
# is identical in every leg because `eval()` changes no value — is balanced
# across arms by construction.
#
# A consequence worth keeping: because all legs share the arm composition, the
# leg-to-leg spread of the trusted parent's ms/round is a NULL control in this
# session, and directly re-measures the FINDING 518 artifact floor.
#
# ONE built tree serves every leg; the session asserts that single certified
# worker digest before and after every leg (RULE 384).
#
# MLXFAST_LOCAL_COOL_GATE=0 is the standing permitted local measurement mode.
# Entry and exit GPU temperature are recorded for every leg and both gate
# fields are written verbatim as false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:?usage: research/e202_session.sh MODE OFFSETS TOKENS [NAME]}"
offsets="${2:?usage: research/e202_session.sh MODE OFFSETS TOKENS [NAME]}"
tokens="${3:?usage: research/e202_session.sh MODE OFFSETS TOKENS [NAME]}"
name="${4:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "${mode}" in submit|iterate) ;; *) echo "e202_session: MODE must be submit or iterate" >&2; exit 2 ;; esac

repo_root="${PWD}"
session_out="${repo_root}/research/out/e202/session-${mode}-${name}"
mkdir -p "${session_out}"

build_record="research/e202-build.txt"
[[ -s "${build_record}" ]] || { echo "e202_session: run research/e202_build.sh first" >&2; exit 2; }
worker=".build-worker/release/mlxfast-runtime-worker"
want_digest="$(awk -F= '/^worker_sha256=/ {print $2}' "${build_record}")"
digest_of() { shasum -a 256 "${worker}" | cut -d' ' -f1; }

# HEAD IDENTITY (RULE 389, HARNESS DEFECT 44): resolve the DECLARED head
# explicitly; never let the wrapper fall back to the organizer-pinned cache.
export MLXFAST_QWEN_MTP_HEAD_DIR="${MLXFAST_QWEN_MTP_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${MLXFAST_QWEN_MTP_HEAD_DIR}/model.safetensors" ]] || {
  echo "e202_session: declared head missing at ${MLXFAST_QWEN_MTP_HEAD_DIR}; run research/fetch-declared-head.sh" >&2
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

IFS=',' read -r -a legs <<<"${offsets}"
for off in "${legs[@]}"; do
  case "${off}" in 0|1|2|-) ;; *) echo "e202_session: offset ${off} must be 0, 1, 2 or -" >&2; exit 2 ;; esac
done

echo "e202_session: mode=${mode} offsets=${offsets} tokens=${tokens}"
echo "e202_session: worker $(digest_of) (certified ${want_digest})"
echo "e202_session: head_provenance_sha256=${head_provenance_sha256}"

status=0
leg=0
for off in "${legs[@]}"; do
  leg=$((leg + 1))
  out="${session_out}/leg$(printf '%02d' "${leg}")-off${off}"
  mkdir -p "${out}"

  before="$(digest_of)"
  if [[ "${before}" != "${want_digest}" ]]; then
    echo "e202_session: worker ${before} is not the certified build ${want_digest}" >&2
    status=1; break
  fi

  entry="$(gpu_temp)"
  echo "=== e202: leg ${leg}/${#legs[@]} offset=${off} entry=${entry}C ==="

  (
    export MLXFAST_LOCAL_COOL_GATE=0
    # ARM WITNESS (RULE 391b) and the per-round timing instrument in one: the
    # round trace carries `e202_arm`, `e202_calls` and `e202_barriers` for
    # every round, so a leg proves which arm each round executed. The trace
    # needs the sandbox lifted (the worker profile denies every write except
    # /dev/null and the timed parent discards worker stderr). The instrument is
    # identical in all three arms (RULE 391c) — it runs once per round, outside
    # the split-cell branch — but it makes the session incomparable with
    # sandboxed, untraced history, which meta.txt records.
    export MLXFAST_NO_SANDBOX=1
    export MLX_QWEN_MTP_TRACE=1
    export MLX_QWEN_MTP_TRACE_PATH="${out}/trace.txt"   # RULE 386: one path per leg
    unset MLX_QWEN_MTP_TRACE_SYNC_HEAD
    if [[ "${off}" == "-" ]]; then
      unset DARKBLOOM_E202_BARRIER_OFFSET
    else
      export DARKBLOOM_E202_BARRIER_OFFSET="${off}"
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
    echo "e202_offset=${off}"
    echo "e202_leg=${leg}"
    echo "e202_session_offsets=${offsets}"
    echo "mode=qwen-mtp-local-${mode}"
    echo "tokens=${tokens}"
    echo "exit=${rc}"
    echo "gpu_temp_entry=${entry:-unavailable}"
    echo "gpu_temp_exit=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "head_provenance_sha256=${head_provenance_sha256}"
    echo "head_class=declared"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "worker_sandbox=disabled"
    echo "comparable_with_sandboxed_history=false"
    echo "phase_trace=1"
    echo "added_timing_instrument=e202-per-round-trace-and-arm-witness-all-arms"
    echo "timing_source=per-round-trace-round_us"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  if [[ "${before}" != "${after}" ]]; then
    echo "e202_session: worker changed DURING leg ${leg}" >&2
    status=1; break
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "e202_session: leg ${leg} (offset ${off}) exited ${rc}; see ${out}/run.log" >&2
    status=1; break
  fi
done

echo "e202_session: artifacts in ${session_out}"
exit "${status}"
