#!/usr/bin/env bash
# Run one ABBA-counterbalanced, ungated E85 timing session.
#
#   usage: research/e85_abba.sh SESSION_TAG TOKENS REPEATS ARM_A ARM_B
#
#   ARM is one of: base a b ab
#     base  MLX_E85_FUSED_EMBED=0 MLX_E85_GATHER_QMM=0
#     a     fused quantized-embedding dual-norm-concat only
#     b     gather_qmm rerank only
#     ab    both
#
# One binary serves every arm; the arms differ only in two MLX_-prefixed
# environment variables, so no leg can time a different build. The MLXFAST_
# prefix would be dropped by the worker's environment sanitizer.
#
# The cool gate is DISABLED. That is legal for a local timed arm only under the
# three standing conditions, all of which this script enforces or records:
#   1. arms are ABBA-counterbalanced inside one session;
#   2. entry and exit GPU temperature are recorded for every arm;
#   3. cool_gate_passed_real_gate=false and gate_qualified_for_timing=false are
#      preserved verbatim.
# A leg from this script is directional causal evidence inside its session. It
# is never a gate-qualified or ranked score.
#
# The session aborts if any leg exits non-zero or reports
# all_tokens_matched != true, so a correctness break cannot burn the allocation.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="${1:?usage: e85_abba.sh SESSION_TAG TOKENS REPEATS ARM_A ARM_B}"
tokens="${2:?usage: e85_abba.sh SESSION_TAG TOKENS REPEATS ARM_A ARM_B}"
repeats="${3:?usage: e85_abba.sh SESSION_TAG TOKENS REPEATS ARM_A ARM_B}"
arm_a="${4:?usage: e85_abba.sh SESSION_TAG TOKENS REPEATS ARM_A ARM_B}"
arm_b="${5:?usage: e85_abba.sh SESSION_TAG TOKENS REPEATS ARM_A ARM_B}"

head_dir="${E85_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e85_abba.sh: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 1
fi

root="research/out/${session}"
rm -rf "${root}"
mkdir -p "${root}"

gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

arm_env() {
  case "$1" in
    base) echo "0 0" ;;
    a)    echo "1 0" ;;
    b)    echo "0 1" ;;
    ab)   echo "1 1" ;;
    *) echo "e85_abba.sh: unknown arm $1" >&2; exit 2 ;;
  esac
}

worker_sha="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
cli_sha="$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"

{
  echo "session=${session}"
  echo "experiment=e85-materialised-intermediate-elimination"
  echo "tokens=${tokens}"
  echo "repeats=${repeats}"
  echo "order=ABBA x ${repeats} (A=${arm_a}, B=${arm_b})"
  echo "local_mode=--local-iterate"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "head_dir=${head_dir}"
  echo "worker_sha256=${worker_sha}"
  echo "cli_sha256=${cli_sha}"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${root}/session.txt"

printf 'leg\tarm\tfused_embed\tgather_qmm\tmtp_s_per_tok\tserial_s_per_tok\tratio\tmean_draft_len\taccepted_rate\tmatched\ttemp_in\ttemp_out\tseconds\n' \
  > "${root}/legs.tsv"

leg=0
status=0
for ((r = 0; r < repeats; ++r)); do
  for arm in "${arm_a}" "${arm_b}" "${arm_b}" "${arm_a}"; do
    leg=$((leg + 1))
    read -r fused gather <<<"$(arm_env "${arm}")"
    out="${root}/leg$(printf '%02d' "${leg}")-${arm}"
    mkdir -p "${out}"

    temp_in="$(gpu_temp)"
    started_s=$(date +%s)
    (
      export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
      export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
      export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
      export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
      export MLXFAST_NO_SANDBOX=1
      export MLXFAST_LOCAL_COOL_GATE=0
      export MLX_E85_FUSED_EMBED="${fused}"
      export MLX_E85_GATHER_QMM="${gather}"
      # Per-round records. The advisor requires a median over paired per-round
      # deltas, because leg totals sum the rare multi-millisecond OS scheduling
      # spikes that a median rejects. `snapshotScheduleSignal` only formats
      # host values that the round already materialised, so the trace adds no
      # GPU synchronisation, and it is enabled identically for both arms.
      export MLX_QWEN_MTP_TRACE=1
      export MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/rounds.txt"
      ./benchmark-qwen-mtp.sh --local-iterate \
        > "${out}/wrapper.out" 2> "${out}/wrapper.err"
    )
    leg_status=$?
    elapsed=$(( $(date +%s) - started_s ))
    temp_out="$(gpu_temp)"

    if ((leg_status != 0)); then
      echo "e85_abba.sh: leg ${leg} (${arm}) exited ${leg_status}" >&2
      status="${leg_status}"
      break
    fi

    row="$(python3 - "${out}/score.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
print("\t".join(str(m[k]) for k in (
    "mtp_seconds_per_token", "serial_seconds_per_token", "mtp_decode_speedup",
    "effective_mean_draft_len", "accepted_draft_rate", "all_tokens_matched")))
PY
)"
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n' \
      "${leg}" "${arm}" "${fused}" "${gather}" "${row}" \
      "${temp_in}" "${temp_out}" "${elapsed}" >> "${root}/legs.tsv"

    if [[ "${row}" != *"True"* ]]; then
      echo "e85_abba.sh: leg ${leg} (${arm}) reported all_tokens_matched != true" >&2
      status=3
      break
    fi
  done
  ((status == 0)) || break
done

{
  echo "legs_completed=${leg}"
  echo "post_run_worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${root}/session.txt"

exit "${status}"
