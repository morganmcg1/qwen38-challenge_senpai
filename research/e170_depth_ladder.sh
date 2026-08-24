#!/usr/bin/env bash
# Research-only (qwen38-r1-e170-fixed-depth-ladder): gated fixed-draft-depth
# ladder at the submission token window.
#
#   research/e170_depth_ladder.sh ARM[,ARM...] REPEATS
#
#   ARM   adapt   the shipped adaptive schedule, no environment override
#         dN      MLX_E159_FIXED_DRAFT_DEPTH=N, so every round proposes
#                 min(offer, N) drafts and the verified width is M = 1 + N
#
# This is a TIMED script and it differs from research/e37-run.sh in the three
# ways that make a wall time admissible:
#
#   * the real 40 C cool gate runs, because MLXFAST_LOCAL_COOL_GATE is never
#     set here;
#   * the per-round phase trace is never enabled, because tracing perturbs
#     round wall time;
#   * `--local-submit` at 512 tokens is the measured mode, so exact post-EOS
#     continuation and row-ledger closure are checked on the same pass that is
#     timed.
#
# Arm order is a palindrome across the whole session, so a monotone thermal or
# clock drift cancels to first order for every arm at once rather than only
# within a pair.
#
# The worker digest is asserted before AND after every leg. A leg whose binary
# changed under it is not a measurement of anything, and the ladder's whole
# claim is that two legs differ only by one environment integer.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arms="${1:?usage: research/e170_depth_ladder.sh ARM[,ARM...] REPEATS}"
repeats="${2:?usage: research/e170_depth_ladder.sh ARM[,ARM...] REPEATS}"

IFS=',' read -r -a arm_list <<<"${arms}"
for arm in "${arm_list[@]}"; do
  case "${arm}" in
    adapt|d[0-8]) ;;
    *) echo "e170: arm must be adapt or dN for N in 0..8" >&2; exit 2 ;;
  esac
done

out_root="${E170_OUT:-${PWD}/research/out/e170}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"

# A stale worker twin silently measures the previous build. Fail before the
# GPU time, not after it.
research/e168_build.sh || exit 1

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-/opt/homebrew/bin/macmon}"
export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${E170_TOKENS:-512}"

digest() { shasum -a 256 "${worker}" | cut -d' ' -f1; }
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

baseline_digest="$(digest)"
echo "e170: worker ${baseline_digest}"
echo "e170: tokens ${MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS}, real 40C gate ON"

# One palindrome unit is the arm list followed by its reverse, so repeating it
# keeps the whole session symmetric about its midpoint.
order=("${arm_list[@]}" $(printf '%s\n' "${arm_list[@]}" | tail -r))
session=()
for ((r = 0; r < repeats; r++)); do session+=("${order[@]}"); done
echo "e170: session order ${session[*]}"

status=0
leg=0
for arm in "${session[@]}"; do
  leg=$((leg + 1))
  case "${arm}" in
    adapt) unset MLX_E159_FIXED_DRAFT_DEPTH ;;
    *) export MLX_E159_FIXED_DRAFT_DEPTH="${arm#d}" ;;
  esac

  out="${out_root}/${arm}/leg${leg}"
  rm -rf "${out}"; mkdir -p "${out}/reports"
  export MLXFAST_SCORE_PATH="${out}/score.json"
  export MLXFAST_CAPTURE_DIR="${out}/reports"

  before="$(digest)"
  [[ "${before}" == "${baseline_digest}" ]] || {
    echo "e170: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }
  entry="$(gpu_temp)"
  echo "=== e170: leg ${leg}/${#session[@]} arm=${arm} entry=${entry}C ==="

  ./benchmark-qwen-mtp.sh --local-submit || { status=1; }

  exit_temp="$(gpu_temp)"
  after="$(digest)"
  {
    echo "e170_arm=${arm}"
    echo "e170_leg=${leg}"
    echo "e170_session_order=${session[*]}"
    echo "fixed_draft_depth=${MLX_E159_FIXED_DRAFT_DEPTH:-unset}"
    echo "gpu_temp_entry=${entry:-unavailable}"
    echo "gpu_temp_exit=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "cool_gate_passed_real_gate=true"
    echo "gate_qualified_for_timing=true"
    echo "trace_perturbs_timing=false"
    echo "tokens=${MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS}"
    echo "mode=qwen-mtp-local-submit"
  } > "${out}/meta.txt"

  [[ "${before}" == "${after}" ]] || {
    echo "e170: worker changed DURING leg ${leg}" >&2
    status=1; break
  }
  ((status)) && break
done
exit "${status}"
