#!/usr/bin/env bash
# E146 R-C -- the local fixed-binary noise floor.
#
#   usage: research/e146_rc_session.sh [TAG] [ORDER]
#
# WHAT THIS MEASURES. The run-to-run sd of a gate-qualified 512-token local leg
# when the binary, the fixture, the host, the session and the arm are all held
# fixed. There is NO arm here and no source change: every MTP leg is the same
# unchanged advisor base tree, run again. The whole quantity is nuisance.
#
# WHY THE REAL GATE. This is a floor measurement for gate-qualified legs, so
# MLXFAST_LOCAL_COOL_GATE=0 is not available: an ungated session measures
# thermal drift as well as leg noise and the two cannot be separated afterwards.
# Every leg runs `./benchmark.sh --local-cool-gate-only`, which is the wrapper's
# own 40 C gate (benchmark.sh:28 COOL_GATE_TEMP_C=40, dispatched at
# benchmark.sh:1701-1709). Entry temperature, exit temperature and the gate wait
# are recorded per leg so drift is auditable rather than assumed absent.
#
# NO TRACE. `MLX_QWEN_MTP_TRACE` writes a line per round and invalidates the
# timing, so it is not set. The behavioural channel here is `round_count`, which
# `mtp-timed` reports without the trace.
#
# LOCK. `mtp-timed` does not run benchmark.sh's resident-model guard, so this
# session reuses benchmark.sh's own lock definitions. The gate child cannot
# deadlock against them: `--local-cool-gate-only` exits at benchmark.sh:1709,
# before `acquire_local_run_lock` at benchmark.sh:1938.
#
# HOME is exported for the same reason as research/e142_rung0.sh: run_job starts
# a supervised process with HOME=/Users/ec2-user, while the declared head cache
# lives under the role home.
#
# This is a LOCAL harness: one public-corpus fixture, reference rows generated
# by the candidate itself. It is never an official or ranked score.
set -uo pipefail
export HOME=/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-askeladd/home
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:-rc}"
# 8 MTP legs and 3 serial legs, interleaved so the serial legs cannot occupy
# only the cold or only the warm end of the session.
order="${2:-mtp,serial,mtp,mtp,mtp,serial,mtp,mtp,mtp,serial,mtp}"
tokens="${E146_TOKENS:-512}"
depth="${E146_DEPTH:-8}"
prompt_id="${E146_PROMPT:-beagle_a}"
prompt="research/e124_prose_hi_${prompt_id}_512.txt"

swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
worker_bin=".build-worker/release/mlxfast-runtime-worker"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
head_dir="${E146_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
root="${E146_ROOT:-.mlxfast-private/e146}"
out_dir="${root}/${tag}"
goldens_dir="${root}/goldens"
golden="${goldens_dir}/${prompt_id}-rows-$((tokens + 1)).json"

for tool in jq python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "e146_rc_session: missing ${tool}" >&2; exit 1; }
done
for path in "${swift_bin}" "${worker_bin}" ./benchmark.sh; do
  [[ -x "${path}" ]] || { echo "e146_rc_session: missing ${path}" >&2; exit 1; }
done
[[ -s "${prompt}" ]] || { echo "e146_rc_session: no prompt ${prompt}" >&2; exit 1; }
[[ -f "${weights}/config.json" ]] || {
  echo "e146_rc_session: missing ${weights}/config.json" >&2; exit 1; }

# THE MEASURED TREE MUST BE THE ASSIGNMENT BASE TREE. A dirty candidate path
# would make this a measurement of an edit, not of the floor.
dirty="$(git status --porcelain -- Sources Vendor Package.swift Package.resolved \
         mtp-head.manifest.json | wc -l | tr -d ' ')"
[[ "${dirty}" == "0" ]] || {
  echo "e146_rc_session: candidate paths are dirty; refusing to measure" >&2
  git status --porcelain -- Sources Vendor Package.swift Package.resolved \
    mtp-head.manifest.json >&2
  exit 1; }

head_verification="$(research/fetch-declared-head.sh 2>&1)" || {
  echo "e146_rc_session: declared head verification failed" >&2
  echo "${head_verification}" >&2; exit 1; }
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e146_rc_session: no run-tree MTP head at ${head_dir}" >&2; exit 1; }

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
  echo "e146_rc_session: could not reuse benchmark.sh's run-lock definitions" >&2
  exit 1; }
trap 'release_local_run_lock' EXIT
acquire_local_run_lock
abort_if_model_already_resident

mkdir -p "${out_dir}" "${goldens_dir}"

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

# The gate's own "waited Ns" line is the only honest record of how long the leg
# queued, so it is parsed rather than timed from outside.
gate_log="${out_dir}/cool-gate.log"
cool_gate() {
  local label="$1" tmp="${out_dir}/.gate.$$"
  echo "--- cool gate before ${label}" >&2
  ./benchmark.sh --local-cool-gate-only 2> "${tmp}"
  local rc=$?
  cat "${tmp}" >> "${gate_log}"
  gate_waited_s="$(sed -n 's/.*gate passed .*waited \([0-9][0-9]*\)s.*/\1/p' "${tmp}" | tail -1)"
  gate_passed="$(grep -c 'GPU cool-down gate passed' "${tmp}")"
  rm -f "${tmp}"
  return "${rc}"
}

# --- reference rows, once ----------------------------------------------------
# Generated by this same unchanged tree. That makes `all_tokens_matched` a
# determinism check across the eleven legs, not a fidelity claim against the
# hidden reference.
if ! jq -e --argjson n "$((tokens + 1))" '
      . as $g
      | $g.reference_self_consistent == true
        and ($g.rows | length) >= $n
        and ([range(0; ($g.rows | length))
              | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)]
             | length) == 0
    ' "${golden}" >/dev/null 2>&1; then
  echo "=== e146: reference rows for ${prompt_id} ($((tokens + 1)) rows) ==="
  MLXFAST_NO_SANDBOX=1 "${swift_bin}" generate-golden \
    --prompt-file "${prompt}" \
    --weights "${weights}" \
    --output "${out_dir}/seed.json" \
    --name "e146_${prompt_id}" \
    --steps 64 > "${out_dir}/seed.log" 2>&1 || {
      echo "e146_rc_session: tokenizer pass failed" >&2
      tail -5 "${out_dir}/seed.log" >&2; exit 1; }
  jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
    "${out_dir}/seed.json" > "${out_dir}/plan.json"
  MLXFAST_NO_SANDBOX=1 "${swift_bin}" mtp-verify \
    --weights "${weights}" \
    --mtp-head "${head_dir}" \
    --emitted "${out_dir}/plan.json" \
    --generate "$((tokens + 1))" \
    --mtp-depth "${depth}" \
    --output "${golden}" \
    --plan-output "${out_dir}/generated-plan.json" \
    > "${out_dir}/verify.log" 2>&1 || {
      echo "e146_rc_session: reference generation failed" >&2
      tail -20 "${out_dir}/verify.log" >&2; exit 1; }
  jq -e --argjson n "$((tokens + 1))" '
      . as $g
      | $g.reference_self_consistent == true
        and ($g.rows | length) >= $n
        and ([range(0; ($g.rows | length))
              | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)]
             | length) == 0
    ' "${golden}" >/dev/null || {
      echo "e146_rc_session: reference rows unusable" >&2; exit 1; }
fi

# --- legs --------------------------------------------------------------------
legs="${out_dir}/legs.jsonl"
: > "${legs}"
: > "${gate_log}"
index=0
status=0
worker_sha_start="$(shasum -a 256 "${worker_bin}" | cut -d' ' -f1)"
session_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

IFS=',' read -r -a kinds <<< "${order}"
for kind in "${kinds[@]}"; do
  index=$((index + 1))
  label="$(printf '%02d-%s' "${index}" "${kind}")"
  report="${out_dir}/leg.${label}.json"
  case "${kind}" in
    mtp)    leg_depth="${depth}" ;;
    serial) leg_depth=0 ;;
    *) echo "e146_rc_session: unknown leg kind ${kind}" >&2; exit 2 ;;
  esac

  gate_waited_s=""
  gate_passed=0
  cool_gate "leg ${label}" || { echo "e146_rc_session: cool gate failed" >&2; exit 1; }
  entry_c="$(gpu_temp)"
  leg_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  worker_before="$(shasum -a 256 "${worker_bin}" | cut -d' ' -f1)"

  echo "=== leg ${label}: depth=${leg_depth} tokens=${tokens} entry=${entry_c}C waited=${gate_waited_s}s"
  MLXFAST_NO_SANDBOX=1 "${swift_bin}" mtp-timed \
    --weights "${weights}" \
    --mtp-head "${head_dir}" \
    --golden "${golden}" \
    --tokens "${tokens}" \
    --mtp-depth "${leg_depth}" \
    --output "${report}" \
    > "${out_dir}/stdout.${label}.json" 2> "${out_dir}/stderr.${label}.log"
  rc=$?
  ((rc == 0)) || status="${rc}"

  exit_c="$(gpu_temp)"
  worker_after="$(shasum -a 256 "${worker_bin}" | cut -d' ' -f1)"
  leg_end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  jq -c -n --arg kind "${kind}" --arg label "${label}" --arg report "${report}" \
        --arg entry "${entry_c}" --arg exitc "${exit_c}" \
        --arg started "${leg_start}" --arg finished "${leg_end}" \
        --arg wbefore "${worker_before}" --arg wafter "${worker_after}" \
        --arg waited "${gate_waited_s}" --argjson gate "${gate_passed}" \
        --argjson depth "${leg_depth}" --argjson rc "${rc}" \
        --slurpfile r "${report}" \
    '{kind: $kind, label: $label, report: $report, leg_depth: $depth,
      exit_code: $rc,
      gpu_temp_entry_c: ($entry | tonumber? // null),
      gpu_temp_exit_c: ($exitc | tonumber? // null),
      cool_gate_wait_seconds: ($waited | tonumber? // null),
      cool_gate_passed_real_gate: ($gate > 0),
      gate_qualified_for_timing: ($gate > 0 and $rc == 0),
      started_utc: $started, finished_utc: $finished,
      worker_sha256_before: $wbefore, worker_sha256_after: $wafter,
      worker_unchanged: ($wbefore == $wafter),
      metrics: (if ($r | length) > 0 then
        ($r[0] | {parent_measured_seconds_per_token, decode_seconds,
                  decode_token_count, seed_prefill_seconds,
                  prefill_seconds_per_token, first_block_seconds,
                  p50_block_request_seconds, round_count,
                  non_drafting_round_count, effective_mean_draft_len,
                  effective_max_draft_len, accepted_draft_rate,
                  accepted_draft_total, rejected_draft_total,
                  all_tokens_matched, residual_divergence_count,
                  emitted_token_total, declared_rows_total,
                  reference_checked_row_total, is_serial_control,
                  mtp_depth, parity_all_ok,
                  blocks_only_seconds: ([$r[0].block_request_seconds[]] | add),
                  block_count: ($r[0].block_request_seconds | length)})
        else null end)}' >> "${legs}"

  if ((rc != 0)); then
    echo "e146_rc_session: leg ${label} exited ${rc}" >&2
    tail -5 "${out_dir}/stderr.${label}.log" >&2
  fi
  [[ "${worker_before}" == "${worker_after}" ]] || {
    echo "e146_rc_session: the worker moved during leg ${label}" >&2; status=1; }
done

{
  echo "experiment=qwen38-r1-e146-nuisance-floor-census"
  echo "rung=R-C"
  echo "harness=local"
  echo "instrument=mtp-timed"
  echo "arm=unchanged-advisor-base"
  echo "order=${order}"
  echo "tokens=${tokens}"
  echo "offered_draft_depth=${depth}"
  echo "prompt_id=${prompt_id}"
  echo "prompt_file=${prompt}"
  echo "prompt_sha256=$(shasum -a 256 "${prompt}" | cut -d' ' -f1)"
  echo "golden=${golden}"
  echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)"
  echo "cool_gate_passed_real_gate=true"
  echo "gate_qualified_for_timing=true"
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=${dirty}"
  echo "head_dir=${head_dir}"
  echo "head_safetensors_sha256=$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
  echo "head_manifest_tree_sha256=$(jq -r .sha256 mtp-head.manifest.json)"
  echo "worker_sha256_start=${worker_sha_start}"
  echo "worker_sha256_end=$(shasum -a 256 "${worker_bin}" | cut -d' ' -f1)"
  echo "cli_sha256=$(shasum -a 256 "${swift_bin}" | cut -d' ' -f1)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "toolchain=$(swift --version 2>&1 | head -1)"
  echo "started_utc=${session_start}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit=${status}"
} > "${out_dir}/meta.txt"

cat "${out_dir}/meta.txt"
exit "${status}"
