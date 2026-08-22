#!/usr/bin/env bash
# E122 rung 0 -- collect the per-round schedule signal on several prompts.
#
#   usage: research/e122_rung0_session.sh PROMPT_ID [PROMPT_ID ...]
#
# NOT A TIMING SESSION. Every leg runs with the per-round phase trace on, which
# writes to a file inside the round, so no seconds figure produced here is a
# measurement of anything. meta.txt records `timing_valid=false` verbatim.
#
# WHAT IT COLLECTS. The shipped session already snapshots, before it proposes
# anything, the target top-2 margin of the pending primary token
# (`snapshotScheduleSignal`, Qwen36MTPBlockSession.swift). Each round's trace
# line therefore already carries
#
#     m=<target top-1 minus top-2 at the previous round's accepted frontier>
#     streak=<full-accept streak>  cap=<width cap in force>
#     ema=<eight per-position acceptance EMAs>
#     sched=<the extension walk: position:p/reach/threshold;...>
#     d=<depth actually drafted>   acc=<drafts accepted>
#
# so rung 0 needs NO source change at all. It is a read of an instrument that
# is already committed and already runs under `MLX_QWEN_MTP_TRACE=1`.
#
# WHY `mtp-timed` AND NOT `--local-iterate` (HARNESS DEFECT 20). The wrapper
# runs a serial control leg and an MTP leg in one process against one trace
# file, so a realised width or depth distribution read from a wrapper trace
# mixes two legs. This session drives the trusted `mtp-timed` verb once per
# prompt, one leg per process, one trace file per leg.
#
# PROMPT IDS. `benchfixture` is the benchmark's own public copy fixture; every
# other id names research/e17_prose_<id>_512.txt, except `english`, which names
# research/e11_prose_gate_english_512.txt. beagle and botany are HIDDEN ranked
# prompts and cannot be run locally; see research/e122-results.md for the
# local-proxy mapping used instead.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(($#)) || { echo "usage: research/e122_rung0_session.sh PROMPT_ID [...]" >&2; exit 2; }

tokens="${E122_TOKENS:-512}"
depth="${E122_DEPTH:-8}"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
root="${E122_ROOT:-.mlxfast-private/e122}"
goldens_dir="${root}/goldens"
runs_root="${root}/runs"
bench_fixture="correctness_prompts/public_longcopy_gate_english_512_256.json"

prompt_file_for() {
  case "$1" in
    benchfixture) echo "${bench_fixture}" ;;
    english) echo "research/e11_prose_gate_english_512.txt" ;;
    *) echo "research/e17_prose_$1_512.txt" ;;
  esac
}

for id in "$@"; do
  [[ -s "$(prompt_file_for "${id}")" ]] || {
    echo "e122_rung0_session: no prompt text for ${id}" >&2; exit 2; }
done

for tool in jq python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "e122_rung0_session: missing ${tool}" >&2; exit 1; }
done
[[ -x "${swift_bin}" ]] || {
  echo "e122_rung0_session: missing ${swift_bin}" >&2; exit 1; }
[[ -f "${weights}/config.json" ]] || {
  echo "e122_rung0_session: missing ${weights}/config.json" >&2; exit 1; }

# The DECLARED head, not the organizer-pinned one: the ranked candidate leg
# runs the head named by mtp-head.manifest.json, so the schedule signal has to
# be collected against that head. The digest check makes the provenance an
# assertion rather than a directory-name convention.
head_dir="${E122_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" ]] || {
  echo "e122_rung0_session: no MTP head at ${head_dir}" >&2; exit 1; }
declared_sha="$(jq -r .sha256 mtp-head.manifest.json)"
head_sha="$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
[[ "${head_sha}" == "${declared_sha}" ]] || {
  echo "e122_rung0_session: head digest ${head_sha} != declared ${declared_sha}" >&2
  exit 1; }

# One model-holding process at a time, reusing benchmark.sh's own lock and
# orphan scan so this session and a local benchmark exclude each other.
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
  echo "e122_rung0_session: could not reuse benchmark.sh's run-lock definitions" >&2
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
status=0

for id in "$@"; do
  prompt="$(prompt_file_for "${id}")"
  out="${runs_root}/${id}"
  rm -rf "${out}"; mkdir -p "${out}"
  golden="${goldens_dir}/${id}-rows-$((tokens + 1)).json"

  # Reference rows: the SERIAL reference trajectory this leg is checked
  # against. Generated once per prompt and cached; they depend only on the
  # target, the head and the decode chain, none of which rung 0 changes.
  if ! jq -e --argjson n "$((tokens + 1))" '
        .reference_self_consistent == true and (.rows | length) >= $n
      ' "${golden}" >/dev/null 2>&1; then
    echo "=== e122 rung 0: reference rows for ${id} ==="
    if [[ "${id}" == "benchfixture" ]]; then
      jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
        "${bench_fixture}" > "${out}/plan.json"
    else
      # One serial step, used only to reach the trusted tokenizer: the
      # prompt's token ids come back as `cases[0].prompt_tokens`.
      MLXFAST_NO_SANDBOX=1 "${swift_bin}" generate-golden \
        --prompt-file "${prompt}" \
        --weights "${weights}" \
        --output "${out}/seed.json" \
        --name "e122_${id}" \
        --steps 1 > "${out}/seed.log" 2>&1 || {
          echo "e122_rung0_session: ${id}: tokenizer pass failed" >&2
          tail -5 "${out}/seed.log" >&2; status=1; continue; }
      jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
        "${out}/seed.json" > "${out}/plan.json"
    fi
    MLXFAST_NO_SANDBOX=1 "${swift_bin}" mtp-verify \
      --weights "${weights}" \
      --mtp-head "${head_dir}" \
      --emitted "${out}/plan.json" \
      --generate "$((tokens + 1))" \
      --mtp-depth "${depth}" \
      --output "${golden}" \
      --plan-output "${out}/generated-plan.json" \
      > "${out}/verify.log" 2>&1 || {
        echo "e122_rung0_session: ${id}: reference generation failed" >&2
        tail -20 "${out}/verify.log" >&2; status=1; continue; }
    jq -e --argjson n "$((tokens + 1))" '
        . as $g
        | $g.reference_self_consistent == true
          and ($g.rows | length) >= $n
          and ([range(0; ($g.rows | length))
                | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)]
               | length) == 0
      ' "${golden}" >/dev/null || {
        echo "e122_rung0_session: ${id}: reference rows unusable" >&2
        status=1; continue; }
  fi

  trace_path="${PWD}/${out}/trace.txt"
  : > "${trace_path}"
  entry_c="$(gpu_temp)"
  start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "=== e122 rung 0: traced leg ${id} (${tokens} tokens, offer ${depth}) ==="
  env MLXFAST_NO_SANDBOX=1 \
      MLX_QWEN_MTP_TRACE=1 \
      MLX_QWEN_MTP_TRACE_PATH="${trace_path}" \
    "${swift_bin}" mtp-timed \
      --weights "${weights}" \
      --mtp-head "${head_dir}" \
      --golden "${golden}" \
      --tokens "${tokens}" \
      --mtp-depth "${depth}" \
      --output "${out}/report.json" \
    > "${out}/stdout.json" 2> "${out}/stderr.log"
  rc=$?
  exit_c="$(gpu_temp)"

  {
    echo "experiment=e122-target-margin-conditioned-draft-depth"
    echo "leg_kind=e122-rung0-traced-signal"
    echo "harness=local"
    echo "prompt_id=${id}"
    echo "prompt_file=${prompt}"
    echo "prompt_sha256=$(shasum -a 256 "${prompt}" | cut -d' ' -f1)"
    echo "golden=${golden}"
    echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)"
    echo "tokens=${tokens}"
    echo "offered_depth=${depth}"
    echo "phase_trace=1"
    echo "timing_valid=false"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "official_or_ranked_score=false"
    echo "gpu_temp_entry_c=${entry_c}"
    echo "gpu_temp_exit_c=${exit_c}"
    echo "base_sha=$(git rev-parse HEAD)"
    echo "dirty_candidate_paths=$(git status --porcelain -- Sources Vendor \
      Package.swift Package.resolved mtp-head.manifest.json | wc -l | tr -d ' ')"
    echo "head_dir=${head_dir}"
    echo "head_safetensors_sha256=$(
      shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
    echo "worker_sha256=$(shasum -a 256 "${worker}" | awk '{print $1}')"
    echo "cli_sha256=$(shasum -a 256 "${swift_bin}" | awk '{print $1}')"
    echo "host=$(hostname)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string)"
    echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
    echo "started_utc=${start_iso}"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}")"
    echo "exit=${rc}"
  } > "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e122_rung0_session: ${id}: mtp-timed exited ${rc}" >&2
    tail -5 "${out}/stderr.log" >&2
    status=1
    continue
  fi
  jq -r '"  matched=\(.all_tokens_matched) rounds=\(.round_count)"
    + " draft=\(.effective_mean_draft_len)"' "${out}/report.json"
done

exit "${status}"
