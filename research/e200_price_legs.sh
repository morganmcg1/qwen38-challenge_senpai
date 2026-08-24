#!/usr/bin/env bash
# Research-only (qwen38-r1-e200-step-aware-depth-price): traced adaptive legs
# that (a) prove the shipped uniform depth price is live inside costModelDepth
# and (b) close the EMA feedback loop the offline replay cannot see.
#
#   research/e200_price_legs.sh REPEATS
#
# One binary, three arms selected at process start by MLX_E200_DEPTH_PRICE:
#
#   ship      makeUniformDepthPrice(0.18)          -- shipped default
#   rstep     E197 ranked smooth-step dR(m), level rescaled to the shipped 1.26
#   rstepnat  E197 ranked smooth-step dR(m), measured level, no rescale
#
# The schedule IS the mechanism here, so MLX_E159_FIXED_DRAFT_DEPTH must stay
# unset: a fixed depth would pin the very decision under test. The script
# refuses to run if it is set.
#
# THIS IS NOT A PUBLISHED DEPTH-PRICE TIMING CONTRAST. RULE 79 forbids deciding
# a depth-price or schedule-policy question from a local timing leg in either
# direction, and the E200 decision is taken from the offline replay carve-out.
# Wall time here is recorded for completeness and for exactness/row-ledger
# coverage only.
#
# NOT GATE-QUALIFIED. MLXFAST_LOCAL_COOL_GATE=0 is set for every leg under the
# standing three conditions: the arm order is a palindrome so monotone thermal
# drift cancels to first order, entry and exit GPU temperature are recorded per
# leg, and meta.txt keeps cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false verbatim.
#
# Tracing is ON for every leg. It perturbs wall time, which is another reason
# no timing claim is published from this script.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

repeats="${1:?usage: research/e200_price_legs.sh REPEATS}"

out_root="${E200_OUT:-${PWD}/research/out/e200/legs}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"
tokens="${E200_TOKENS:-256}"
unit_spec="${E200_UNIT:-ship rstep rstep ship}"

if [[ -n "${MLX_E159_FIXED_DRAFT_DEPTH:-}" ]]; then
  echo "e200: MLX_E159_FIXED_DRAFT_DEPTH is set; the adaptive schedule is the mechanism" >&2
  exit 2
fi

research/e168_build.sh || exit 1

head_dir="${E200_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e200: declared head tree missing at ${head_dir}" >&2
  exit 2
}
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
head_file_sha="$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
echo "e200: head class=declared model.safetensors ${head_file_sha}"

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
echo "e200: worker ${baseline_digest}"
echo "e200: tokens ${tokens}, cool gate OFF (ungated, counterbalanced), trace ON"

read -r -a unit <<< "${unit_spec}"
session=()
for ((r = 0; r < repeats; r++)); do session+=("${unit[@]}"); done
echo "e200: ${#session[@]} legs, order ${session[*]}"

status=0
leg=0
for arm in "${session[@]}"; do
  leg=$((leg + 1))
  case "${arm}" in
    ship|rstep|rstepnat) export MLX_E200_DEPTH_PRICE="${arm}" ;;
    *) echo "e200: unknown arm ${arm}" >&2; status=2; break ;;
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
    echo "e200: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }
  entry="$(gpu_temp)"
  echo "=== e200: leg ${leg}/${#session[@]} arm=${arm} entry=${entry}C ==="
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  ./benchmark-qwen-mtp.sh --local-iterate
  rc=$?

  exit_temp="$(gpu_temp)"
  after="$(digest)"
  {
    echo "e200_arm=${arm}"
    echo "e200_leg=${leg}"
    echo "e200_session_order=${session[*]}"
    echo "depth_price_arm=${MLX_E200_DEPTH_PRICE}"
    echo "fixed_draft_depth=${MLX_E159_FIXED_DRAFT_DEPTH:-unset}"
    echo "head_class=declared"
    echo "head_model_safetensors_sha256=${head_file_sha}"
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
    echo "trace_perturbs_timing=true"
    echo "published_depth_price_timing_contrast=false"
    echo "tokens=${tokens}"
    echo "mode=qwen-mtp-local-iterate"
    echo "exit=${rc}"
    echo "rounds_traced=$(grep -c '^mtp-trace: round=' "${out}/trace.txt" 2>/dev/null || echo 0)"
    echo "started=${started}"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e200: leg ${leg} (${arm}) exited ${rc}" >&2
    status=1; break
  fi
  [[ "${before}" == "${after}" ]] || {
    echo "e200: worker changed DURING leg ${leg}" >&2
    status=1; break
  }
done
exit "${status}"
