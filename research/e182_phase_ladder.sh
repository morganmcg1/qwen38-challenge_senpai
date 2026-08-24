#!/usr/bin/env bash
# Research-only (qwen38-r1-e182-round-cost-phase-decomposition): pinned-depth
# ladder that resolves WHICH PHASE of a drafting round carries the round
# cost's dependence on verify width M.
#
#   research/e182_phase_ladder.sh REPEATS
#
# Two instrument arms at every width, inside ONE palindromic session:
#
#   tN   MLX_QWEN_MTP_TRACE=1 only. The shipped overlap survives, so the
#        coarse phase table (draft_build / verify_build+eval_wall / readout /
#        commit / upkeep) keeps its usual meaning and round_us(M) is the
#        reference shape the band arm must reproduce.
#   bN   MLX_QWEN_MTP_TRACE=1 + MLX_QWEN_MTP_TRACE_SYNC_HEAD=1
#        + MLX_E182_BAND_SYNC=1. The head chain is drained before the verify
#        build, and the verify forward drains at every mixer and MLP boundary,
#        so `d_submit2_us` is head device time and `band_*_us` are per-layer
#        family device times. 129 extra syncs per forward: the arm is
#        ATTRIBUTION ONLY and its absolute round time is not a round cost.
#
# N is MLX_E159_FIXED_DRAFT_DEPTH, so the verified width is M = 1 + N and
# every round of a leg has the same M. N = 0 exercises the serial/skip body,
# which is the M = 1 anchor.
#
# NOT GATE-QUALIFIED. MLXFAST_LOCAL_COOL_GATE=0 is set for every leg under the
# standing three conditions: the arm order is a palindrome so monotone thermal
# drift cancels to first order, entry and exit GPU temperature are recorded per
# leg, and meta.txt keeps cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false verbatim. Tracing perturbs the round on top of
# that, so no number from this script is a candidate speed claim; only the
# SHAPE of a phase against M is read.
#
# Exactness still runs on every leg: --local-iterate checks both timed legs
# against reference rows this build generates for the full window, and the
# trusted parent closes the row ledger, so a leg that decodes a different
# stream fails here instead of contributing a phase number.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

repeats="${1:?usage: research/e182_phase_ladder.sh REPEATS}"

out_root="${E182_OUT:-${PWD}/research/out/e182}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"
tokens="${E182_TOKENS:-256}"
depths="${E182_DEPTHS:-0 1 2 3 4 5 6 7 8}"

research/e168_build.sh || exit 1

# The ranked candidate leg runs the head that mtp-head.manifest.json declares.
# setup-qwen-mtp.sh only provisions the organizer-pinned head, so an
# unmodified --local-iterate would measure the drafting phase of a head the
# ranked run never executes. Point at the resolved declared tree, as E37 does.
head_dir="${E182_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e182: declared head tree missing at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 2
}
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
echo "e182: head $(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"

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
echo "e182: worker ${baseline_digest}"
echo "e182: tokens ${tokens}, cool gate OFF (ungated, counterbalanced)"

order=()
for d in ${depths}; do order+=("t${d}"); done
for d in ${depths}; do order+=("b${d}"); done
reverse=($(printf '%s\n' "${order[@]}" | tail -r))
unit=("${order[@]}" "${reverse[@]}")
session=()
for ((r = 0; r < repeats; r++)); do session+=("${unit[@]}"); done
echo "e182: ${#session[@]} legs, order ${session[*]}"

status=0
leg=0
for arm in "${session[@]}"; do
  leg=$((leg + 1))
  export MLX_E159_FIXED_DRAFT_DEPTH="${arm#?}"
  case "${arm}" in
    t*)
      unset MLX_QWEN_MTP_TRACE_SYNC_HEAD MLX_E182_BAND_SYNC
      ;;
    b*)
      export MLX_QWEN_MTP_TRACE_SYNC_HEAD=1
      export MLX_E182_BAND_SYNC=1
      ;;
    *) echo "e182: unknown arm ${arm}" >&2; status=2; break ;;
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
    echo "e182: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }
  entry="$(gpu_temp)"
  echo "=== e182: leg ${leg}/${#session[@]} arm=${arm} entry=${entry}C ==="
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  ./benchmark-qwen-mtp.sh --local-iterate
  rc=$?

  exit_temp="$(gpu_temp)"
  after="$(digest)"
  {
    echo "e182_arm=${arm}"
    echo "e182_leg=${leg}"
    echo "e182_session_order=${session[*]}"
    echo "fixed_draft_depth=${MLX_E159_FIXED_DRAFT_DEPTH}"
    echo "verify_width_m=$((MLX_E159_FIXED_DRAFT_DEPTH + 1))"
    echo "band_sync=${MLX_E182_BAND_SYNC:-0}"
    echo "sync_head=${MLX_QWEN_MTP_TRACE_SYNC_HEAD:-0}"
    echo "gpu_temp_entry=${entry:-unavailable}"
    echo "gpu_temp_exit=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "head_sha=$(git rev-parse HEAD)"
    echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "timing_claims_permitted=false"
    echo "trace_perturbs_timing=true"
    echo "tokens=${tokens}"
    echo "mode=qwen-mtp-local-iterate"
    echo "exit=${rc}"
    echo "rounds_traced=$(grep -c '^mtp-trace: round=' "${out}/trace.txt" 2>/dev/null || echo 0)"
    echo "started=${started}"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e182: leg ${leg} (${arm}) exited ${rc}" >&2
    status=1; break
  fi
  [[ "${before}" == "${after}" ]] || {
    echo "e182: worker changed DURING leg ${leg}" >&2
    status=1; break
  }
done
exit "${status}"
