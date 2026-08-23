#!/usr/bin/env bash
# E154 R2 -- is the scored round GPU-bound or dispatch-bound?
#
#   usage: research/e154_r2_session.sh SEQUENCE [PROMPT_ID]
#
# SEQUENCE is a comma list of arm names. Each arm runs one `mtp-timed` leg in
# its own process. Arms are named in `arm_env()` below.
#
# WHAT EACH LEG MEASURES. Two independent instruments run together:
#
#   1. Direct accounting, no model. Every round records wall time, this
#      thread's CPU time and the WHOLE PROCESS's CPU time. `wall - process_cpu`
#      is the host's idle share of the round. MLX encodes command buffers on
#      its own scheduler thread, so the process clock is the one that can
#      falsify "the host is saturated"; a calling-thread clock cannot.
#   2. Delay injection, causal. A pure-CPU busy-wait of `delta` us or a
#      dependent dummy matmul chain of `gamma` steps is added to the round from
#      exactly one side. Levels CYCLE BY DRAFTING-ROUND INDEX inside one leg, so
#      the zero level is interleaved by construction and monotone thermal drift
#      is shared by every level instead of being confounded with it.
#
# TOKEN NEUTRALITY. The instrument reads nothing and feeds nothing back. The
# dummy chain is joined into the round's existing blocking eval so its device
# work cannot outlive the round that ordered it. The witness is the report:
# identical `round_count`, identical `effective_draft_lengths` and identical
# `all_tokens_matched` across every arm of a session.
#
# THERMAL. `MLXFAST_LOCAL_COOL_GATE` is not taken. Every leg records entry and
# exit GPU temperature and writes `cool_gate_passed_real_gate=false` and
# `gate_qualified_for_timing=false` verbatim. Arms must be counterbalanced
# inside one session; that is the caller's job through SEQUENCE.
#
# HARNESS DEFECT 28: only MLX_, DARKBLOOM_, METAL_, MTL_, DYLD_ and LC_ names
# survive `sanitizedRuntimeWorkerEnvironment`, which is why the instrument is
# spelled `MLX_E154_*`.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(($#)) || { echo "usage: research/e154_r2_session.sh SEQUENCE [PROMPT_ID]" >&2; exit 2; }
sequence="$1"
prompt_id="${2:-benchfixture}"

tokens="${E154_TOKENS:-512}"
depth="${E154_DEPTH:-8}"
label="${E154_LABEL:-s1}"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
root="${E154_ROOT:-.mlxfast-private/e154}"
goldens_dir="${root}/goldens"
runs_root="${root}/${E154_RUNS_DIR:-runs}"
bench_fixture="correctness_prompts/public_longcopy_gate_english_512_256.json"

# Sweep levels. Log spaced so one leg brackets three orders of magnitude, and
# the top level is larger than a whole local round so it MUST move wall time.
# That top level is positive control 1 (CPU) and 2 (GPU); a sweep whose slope
# never reaches 1 has not proved slack, it has proved nothing.
cpu_sweep="${E154_CPU_SWEEP:-0,1000,4000,16000,64000,256000}"
gpu_sweep="${E154_GPU_SWEEP:-0,1,4,16,64,256}"
gpu_width="${E154_GPU_WIDTH:-1024}"

arm_env() {
  case "$1" in
    zero)      echo "" ;;
    cpu_pre)   echo "MLX_E154_CPU_DELAY_US=${cpu_sweep} MLX_E154_CPU_SITE=preverify" ;;
    cpu_eval)  echo "MLX_E154_CPU_DELAY_US=${cpu_sweep} MLX_E154_CPU_SITE=preeval" ;;
    gpu)       echo "MLX_E154_GPU_CHAIN=${gpu_sweep} MLX_E154_GPU_WIDTH=${gpu_width}" ;;
    *)         echo "UNKNOWN" ;;
  esac
}

for arm in ${sequence//,/ }; do
  [[ "$(arm_env "${arm}")" == "UNKNOWN" ]] && {
    echo "e154_r2: unknown arm ${arm}" >&2; exit 2; }
done

prompt_file_for() {
  case "$1" in
    benchfixture) echo "${bench_fixture}" ;;
    english) echo "research/e11_prose_gate_english_512.txt" ;;
    *) if [[ -s "research/e124_prose_hi_$1_512.txt" ]]; then
         echo "research/e124_prose_hi_$1_512.txt"
       else
         echo "research/e17_prose_$1_512.txt"
       fi ;;
  esac
}
prompt="$(prompt_file_for "${prompt_id}")"
[[ -s "${prompt}" ]] || { echo "e154_r2: no prompt for ${prompt_id}" >&2; exit 2; }

for tool in jq python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "e154_r2: missing ${tool}" >&2; exit 1; }
done
[[ -x "${swift_bin}" ]] || { echo "e154_r2: missing ${swift_bin}" >&2; exit 1; }
[[ -f "${weights}/config.json" ]] || {
  echo "e154_r2: missing ${weights}/config.json" >&2; exit 1; }

# A timing session over uncommitted candidate code cannot be reproduced from
# the recorded commit, so refuse instead of recording a commit that is not what
# ran. `E154_ALLOW_DIRTY=1` is for the debug leg only.
dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" && "${E154_ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "e154_r2: candidate surface is dirty; refusing to time over it" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

head_dir="${E154_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
head_verification="$(research/fetch-declared-head.sh 2>&1)" || {
  echo "e154_r2: declared head verification failed" >&2
  echo "${head_verification}" >&2; exit 1; }
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e154_r2: no run-tree MTP head at ${head_dir}" >&2; exit 1; }

LOCAL_ITERATE=1
LOCAL_SUBMIT=0
lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_guard_enabled\(\) \{/,/^\}/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
eval "${lock_definitions}" || {
  echo "e154_r2: could not reuse benchmark.sh's run-lock definitions" >&2
  exit 1; }
trap 'release_local_run_lock' EXIT
acquire_local_run_lock
abort_if_model_already_resident

mkdir -p "${goldens_dir}" "${runs_root}"

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

worker=".build-worker/release/mlxfast-runtime-worker"
worker_sha="$(shasum -a 256 "${worker}" | awk '{print $1}')"
echo "e154_r2: session_commit=${session_commit}"
echo "e154_r2: session_worker_sha256=${worker_sha}"

# --- reference rows, regenerated on the CURRENT base --------------------------
# A golden carried over from an older base reports as a candidate exactness
# failure, so the session owns its own goldens directory keyed by base.
golden="${goldens_dir}/${prompt_id}-rows-$((tokens + 1))-${session_commit:0:8}.json"
if ! jq -e --argjson n "$((tokens + 1))" '
      .reference_self_consistent == true and (.rows | length) >= $n
    ' "${golden}" >/dev/null 2>&1; then
  gen="${runs_root}/.golden-${prompt_id}"
  rm -rf "${gen}"; mkdir -p "${gen}"
  echo "=== e154_r2: reference rows for ${prompt_id} on ${session_commit:0:8} ==="
  if [[ "${prompt_id}" == "benchfixture" ]]; then
    jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
      "${bench_fixture}" > "${gen}/plan.json"
  else
    MLXFAST_NO_SANDBOX=1 "${swift_bin}" generate-golden \
      --prompt-file "${prompt}" --weights "${weights}" \
      --output "${gen}/seed.json" --name "e154_${prompt_id}" --steps 64 \
      > "${gen}/seed.log" 2>&1 || {
        echo "e154_r2: tokenizer pass failed" >&2
        tail -5 "${gen}/seed.log" >&2; exit 1; }
    jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
      "${gen}/seed.json" > "${gen}/plan.json"
  fi
  MLXFAST_NO_SANDBOX=1 "${swift_bin}" mtp-verify \
    --weights "${weights}" --mtp-head "${head_dir}" \
    --emitted "${gen}/plan.json" --generate "$((tokens + 1))" \
    --mtp-depth "${depth}" --output "${golden}" \
    --plan-output "${gen}/generated-plan.json" \
    > "${gen}/verify.log" 2>&1 || {
      echo "e154_r2: reference generation failed" >&2
      tail -20 "${gen}/verify.log" >&2; exit 1; }
  jq -e --argjson n "$((tokens + 1))" '
      . as $g
      | $g.reference_self_consistent == true and ($g.rows | length) >= $n
        and ([range(0; ($g.rows | length))
              | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)]
             | length) == 0
    ' "${golden}" >/dev/null || {
      echo "e154_r2: reference rows unusable" >&2; exit 1; }
fi
golden_sha="$(shasum -a 256 "${golden}" | cut -d' ' -f1)"

# --- the arm legs -------------------------------------------------------------
status=0
position=0
for arm in ${sequence//,/ }; do
  position=$((position + 1))
  slot="${label}p${position}${arm}"
  out="${runs_root}/${slot}"
  rm -rf "${out}"; mkdir -p "${out}"
  records="${PWD}/${out}/rounds.txt"

  entry_c="$(gpu_temp)"
  start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=== e154_r2 ${slot}: arm=${arm} position=${position} prompt=${prompt_id}" \
       "tokens=${tokens} depth=${depth} entry_c=${entry_c:-na} ==="

  # shellcheck disable=SC2046
  env MLXFAST_NO_SANDBOX=1 "MLX_E154_OUT=${records}" $(arm_env "${arm}") \
    "${swift_bin}" mtp-timed \
      --weights "${weights}" --mtp-head "${head_dir}" \
      --golden "${golden}" --tokens "${tokens}" --mtp-depth "${depth}" \
      --output "${out}/report.json" \
    > "${out}/stdout.json" 2> "${out}/stderr.log"
  rc=$?
  exit_c="$(gpu_temp)"

  {
    echo "experiment=e154-r2-round-boundedness"
    echo "harness=local"
    echo "arm=${arm}"
    echo "arm_env=$(arm_env "${arm}")"
    echo "label=${label}"
    echo "position=${position}"
    echo "sequence=${sequence}"
    echo "prompt_id=${prompt_id}"
    echo "prompt_file=${prompt}"
    echo "tokens=${tokens}"
    echo "offered_depth=${depth}"
    echo "golden=${golden}"
    echo "golden_sha256=${golden_sha}"
    echo "phase_trace=0"
    echo "timing_valid=true"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "official_or_ranked_score=false"
    echo "gpu_temp_entry_c=${entry_c}"
    echo "gpu_temp_exit_c=${exit_c}"
    echo "base_sha=${session_commit}"
    echo "dirty_candidate_paths=$(git status --porcelain -- Sources Vendor \
      Package.swift Package.resolved mtp-head.manifest.json | wc -l | tr -d ' ')"
    echo "head_dir=${head_dir}"
    echo "head_manifest_tree_sha256=$(jq -r .sha256 mtp-head.manifest.json)"
    echo "worker_sha256=${worker_sha}"
    echo "cli_sha256=$(shasum -a 256 "${swift_bin}" | awk '{print $1}')"
    echo "host=$(hostname)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string)"
    echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
    echo "started_utc=${start_iso}"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "instrument_records=$(grep -c '^e154-arm: ' "${records}" 2>/dev/null || echo 0)"
    if [[ -s "${out}/report.json" ]]; then
      jq -r '"all_tokens_matched=\(.all_tokens_matched)",
             "residual_divergence_count=\(.residual_divergence_count)",
             "round_count=\(.round_count)",
             "seconds_per_token=\(.parent_measured_seconds_per_token)",
             "decode_seconds=\(.decode_seconds // "na")",
             "seed_prefill_seconds=\(.seed_prefill_seconds // "na")",
             "effective_mean_draft_len=\(.effective_mean_draft_len)",
             "accepted_draft_rate=\(.accepted_draft_rate)",
             "head_provenance_sha256=\(.head_provenance.sha256 // "none")"' \
        "${out}/report.json"
    fi
    echo "exit=${rc}"
  } > "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e154_r2: ${slot} exited ${rc}" >&2
    tail -5 "${out}/stderr.log" >&2
    status=1
    continue
  fi
  jq -r '"  matched=\(.all_tokens_matched) rounds=\(.round_count)"
    + " spt=\(.parent_measured_seconds_per_token)"
    + " edl=\(.effective_mean_draft_len)"' "${out}/report.json"
done

echo "e154_r2: sequence complete, status=${status}"
exit "${status}"
