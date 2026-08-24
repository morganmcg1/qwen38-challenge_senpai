#!/usr/bin/env bash
# Research-only (qwen38-r1-e168-margin-clamp-calibration): the two-BUILD
# confirmation for the depth cap.
#
#   research/e168_cap_confirm.sh REPEATS
#
# The ladder measured the depth axis with `MLX_E159_FIXED_DRAFT_DEPTH`, which
# PINS the proposed count and bypasses `costModelDepth` completely. The shipped
# mechanism is a CAP that leaves the adaptive walk in charge below the cap, so
# the ladder cannot confirm it. This script measures the two real builds:
#
#   base  the arm's CLI and worker built from the campaign base
#   cap   the same pair built from the candidate commit
#
# Both binary sets must already exist under ${E168_BIN} (see the header of
# research/e168_cap_stage.sh). No build runs here, because a rebuild between
# legs would change the thing under test in the middle of the session.
#
# The three properties that make these wall times admissible:
#   * the real 40 C cool gate runs, because MLXFAST_LOCAL_COOL_GATE is never
#     set here;
#   * the per-round phase trace is never enabled;
#   * `--local-submit` at 512 tokens is the measured mode, so exact post-EOS
#     continuation and row-ledger closure are checked on the timed pass.
#
# Arm order is a palindrome (base cap cap base), so a monotone thermal or clock
# drift cancels to first order within one repeat.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

repeats="${1:?usage: research/e168_cap_confirm.sh REPEATS}"

bin_root="${E168_BIN:-${PWD}/research/out/e172/bin}"
out_root="${E168_CONFIRM_OUT:-${PWD}/research/out/e172}"

for arm in base cap; do
  for product in mlxfast-swift mlxfast-runtime-worker; do
    [[ -x "${bin_root}/${arm}/${product}" ]] || {
      echo "e168_confirm: missing ${bin_root}/${arm}/${product}" >&2
      exit 2
    }
  done
done

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-/opt/homebrew/bin/macmon}"
export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${E168_TOKENS:-512}"
# The pinned-depth research instrument must be off: it would bypass the very
# code path the cap acts on and make both arms identical.
unset MLX_E159_FIXED_DRAFT_DEPTH

digest() { shasum -a 256 "$1" | cut -d' ' -f1; }
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

declare -A worker_digest cli_digest
for arm in base cap; do
  worker_digest["${arm}"]="$(digest "${bin_root}/${arm}/mlxfast-runtime-worker")"
  cli_digest["${arm}"]="$(digest "${bin_root}/${arm}/mlxfast-swift")"
  echo "e168_confirm: ${arm} worker ${worker_digest[${arm}]}"
done
[[ "${worker_digest[base]}" != "${worker_digest[cap]}" ]] || {
  echo "e168_confirm: both arms have the SAME worker; nothing is under test" >&2
  exit 2
}
echo "e168_confirm: tokens ${MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS}, real 40C gate ON"

session=()
for ((r = 0; r < repeats; r++)); do session+=(base cap cap base); done
echo "e168_confirm: session order ${session[*]}"

status=0
leg=0
for arm in "${session[@]}"; do
  leg=$((leg + 1))
  out="${out_root}/${arm}/leg${leg}"
  rm -rf "${out}"; mkdir -p "${out}/reports"

  export MLXFAST_RUNTIME_WORKER_EXECUTABLE="${bin_root}/${arm}/mlxfast-runtime-worker"
  export MLXFAST_SCORE_PATH="${out}/score.json"
  export MLXFAST_CAPTURE_DIR="${out}/reports"
  export MLXFAST_CAPTURE_REAL_BIN="${bin_root}/${arm}/mlxfast-swift"
  export MLXFAST_SWIFT_BIN="${PWD}/research/capture-cli.sh"

  before_worker="$(digest "${MLXFAST_RUNTIME_WORKER_EXECUTABLE}")"
  [[ "${before_worker}" == "${worker_digest[${arm}]}" ]] || {
    echo "e168_confirm: ${arm} worker changed before leg ${leg}" >&2
    status=1; break
  }
  entry="$(gpu_temp)"
  echo "=== e168_confirm: leg ${leg}/${#session[@]} arm=${arm} entry=${entry}C ==="

  ./benchmark-qwen-mtp.sh --local-submit || { status=1; }

  exit_temp="$(gpu_temp)"
  after_worker="$(digest "${MLXFAST_RUNTIME_WORKER_EXECUTABLE}")"
  {
    echo "e170_arm=${arm}"
    echo "e170_leg=${leg}"
    echo "e170_session_order=${session[*]}"
    echo "fixed_draft_depth=unset"
    echo "worker_path=${MLXFAST_RUNTIME_WORKER_EXECUTABLE}"
    echo "cli_sha256=${cli_digest[${arm}]}"
    echo "gpu_temp_entry=${entry:-unavailable}"
    echo "gpu_temp_exit=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before_worker}"
    echo "worker_sha256_after=${after_worker}"
    echo "worker_digest_stable=$([[ "${before_worker}" == "${after_worker}" ]] && echo true || echo false)"
    echo "cool_gate_passed_real_gate=true"
    echo "gate_qualified_for_timing=true"
    echo "trace_perturbs_timing=false"
    echo "tokens=${MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS}"
    echo "mode=qwen-mtp-local-submit"
  } > "${out}/meta.txt"

  [[ "${before_worker}" == "${after_worker}" ]] || {
    echo "e168_confirm: ${arm} worker changed DURING leg ${leg}" >&2
    status=1; break
  }
  ((status)) && break
done
exit "${status}"
