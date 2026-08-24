#!/usr/bin/env bash
# Research-only (qwen38-r1-e189-single-pass-qmv-screen): end-to-end pinned-row
# ladder that prices the single-pass QMV width plan against the shipped staged
# plan at the widths where they differ.
#
#   research/e189_rows_ladder.sh REPEATS
#
# One binary, two arms selected at process start:
#
#   sN   MLX_E189_QMV_WIDTH_PLAN=staged     (shipped plan, G(m)=2 at m=6,7,8)
#   pN   MLX_E189_QMV_WIDTH_PLAN=singlepass (G(m)=1 at m<=8)
#
# N is MLX_E159_FIXED_DRAFT_DEPTH, so the verified width is M = 1 + N and every
# round of a leg has the same M. M = 5 is the control: both plans compile the
# same IPG = 5 case there, so any measured difference at M = 5 is session noise.
#
# `MLX_QWEN_MTP_TRACE=1` alone is E182's `t` arm: the shipped overlap survives
# and `round_us` keeps its usual meaning. The primary read is nevertheless the
# ABSOLUTE candidate figure (`mtp_seconds_per_token` and the round_us median),
# never the local serial-to-MTP ratio, because both local legs run the same
# candidate build.
#
# NOT GATE-QUALIFIED. MLXFAST_LOCAL_COOL_GATE=0 is set for every leg under the
# standing three conditions: the arm order is a palindrome so monotone thermal
# drift cancels to first order, entry and exit GPU temperature are recorded per
# leg, and meta.txt keeps cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false verbatim.
#
# Exactness still runs on every leg: --local-iterate checks both timed legs
# against reference rows this build generates for the full window, and the
# trusted parent closes the row ledger.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

repeats="${1:?usage: research/e189_rows_ladder.sh REPEATS}"

out_root="${E189_OUT:-${PWD}/research/out/e189/ladder}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"
tokens="${E189_TOKENS:-256}"
depths="${E189_DEPTHS:-4 5 6 7}"

research/e168_build.sh || exit 1

head_dir="${E189_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e189: declared head tree missing at ${head_dir}" >&2
  exit 2
}
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
echo "e189: head $(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-/opt/homebrew/bin/macmon}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_LOCAL_COOL_GATE=0
export MLXFAST_NO_SANDBOX=1

digest() { shasum -a 256 "${worker}" | cut -d' ' -f1; }
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

baseline_digest="$(digest)"
echo "e189: worker ${baseline_digest}"
echo "e189: tokens ${tokens}, cool gate OFF (ungated, counterbalanced)"

order=()
for d in ${depths}; do order+=("s${d}"); done
for d in ${depths}; do order+=("p${d}"); done
reverse=($(printf '%s\n' "${order[@]}" | tail -r))
unit=("${order[@]}" "${reverse[@]}")
session=()
for ((r = 0; r < repeats; r++)); do session+=("${unit[@]}"); done
echo "e189: ${#session[@]} legs, order ${session[*]}"

status=0
leg=0
for arm in "${session[@]}"; do
  leg=$((leg + 1))
  export MLX_E159_FIXED_DRAFT_DEPTH="${arm#?}"
  case "${arm}" in
    s*) export MLX_E189_QMV_WIDTH_PLAN=staged ;;
    p*) export MLX_E189_QMV_WIDTH_PLAN=singlepass ;;
    *) echo "e189: unknown arm ${arm}" >&2; status=2; break ;;
  esac

  out="${out_root}/${arm}/leg${leg}"
  rm -rf "${out}"; mkdir -p "${out}/reports"
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="${out}/trace.txt"
  export MLXFAST_SCORE_PATH="${out}/score.json"
  export MLXFAST_CAPTURE_DIR="${out}/reports"
  export MLXFAST_CAPTURE_REAL_BIN="${PWD}/.build/release/mlxfast-swift"
  export MLXFAST_SWIFT_BIN="${PWD}/research/capture-cli.sh"

  before="$(digest)"
  [[ "${before}" == "${baseline_digest}" ]] || {
    echo "e189: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }
  entry="$(gpu_temp)"
  echo "=== e189: leg ${leg}/${#session[@]} arm=${arm} entry=${entry}C ==="
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  ./benchmark-qwen-mtp.sh --local-iterate
  rc=$?

  exit_temp="$(gpu_temp)"
  after="$(digest)"
  {
    echo "e189_arm=${arm}"
    echo "e189_leg=${leg}"
    echo "e189_session_order=${session[*]}"
    echo "width_plan=${MLX_E189_QMV_WIDTH_PLAN}"
    echo "fixed_draft_depth=${MLX_E159_FIXED_DRAFT_DEPTH}"
    echo "verify_width_m=$((MLX_E159_FIXED_DRAFT_DEPTH + 1))"
    echo "gpu_temp_entry=${entry:-unavailable}"
    echo "gpu_temp_exit=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "head_sha=$(git rev-parse HEAD)"
    echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "official_or_ranked_score=false"
    echo "tokens=${tokens}"
    echo "mode=qwen-mtp-local-iterate"
    echo "exit=${rc}"
    echo "rounds_traced=$(grep -c '^mtp-trace: round=' "${out}/trace.txt" 2>/dev/null || echo 0)"
    echo "started=${started}"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e189: leg ${leg} (${arm}) exited ${rc}" >&2
    status=1; break
  fi
  [[ "${before}" == "${after}" ]] || {
    echo "e189: worker changed DURING leg ${leg}" >&2
    status=1; break
  }
done
exit "${status}"
