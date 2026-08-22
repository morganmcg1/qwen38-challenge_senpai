#!/usr/bin/env bash
# E139 -- one leg of the ZERO-NOISE ACCEPTANCE channel.
#
#   usage: research/e139_session.sh ARM PROMPT_ID [PROMPT_ID ...]
#
# NOT A TIMING SESSION. Every leg runs the per-round phase trace, which writes
# a line per round, so `timing_valid=false` is recorded verbatim on every leg.
# The channel this session reads is `effective_mean_draft_len`,
# `accepted_draft_rate` and `round_count`, which are functions of the arm, the
# fixture and the token window alone. They carry no thermal or scheduling
# noise, so one leg per arm resolves them exactly.
#
# ARMS. All arms are the SAME worker binary under environment gates that only
# reach the PROPOSAL path.
#
#   ship      shipped default: bf16-narrowed rerank store, probe fraction 0.25
#   fp32      MLX_E139_FP32_TIEBREAK=1
#   p0<D>     MLX_E139_PROBE_FRACTION=0.<D>
#   fp32p0<D> both riders, probe fraction 0.<D>
#
# The probe arm name carries the fraction: `p0` stands for the leading `0.`
# and the rest are the decimal digits, so `p015` is 0.15, `p002` is 0.02 and
# `p0075` is 0.075.
#
# WITNESS (CAMPAIGN RULE 114). The round trace carries
# `sel_env=<top32 gate>+e139fp32:<gate>:<drafts>+e139p:<gate>:<probes>`. A leg
# whose witnessed gates or probe count disagree with the arm it was asked for
# is discarded, not interpreted. `probes = ceil(fraction * 12292)`, so the
# probe arm is witnessed by an integer the run derived, never by the string it
# was given.
#
#   E139_TOKENS=512   decode tokens per leg
#   E139_REP=1        replicate index, for the same-arm determinism pair
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(($# >= 2)) || { echo "usage: research/e139_session.sh ARM PROMPT_ID [...]" >&2; exit 2; }

arm="$1"; shift
tokens="${E139_TOKENS:-512}"
rep="${E139_REP:-1}"
depth="${E139_DEPTH:-8}"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
root="${E139_ROOT:-.mlxfast-private/e139}"
goldens_dir="${root}/goldens"
runs_root="${root}/runs"
bench_fixture="correctness_prompts/public_longcopy_gate_english_512_256.json"
leaves=12292

case "${arm}" in
  ship)     arm_env=();                        want_fp32=unset; want_p=unset ;;
  fp32)     arm_env=(MLX_E139_FP32_TIEBREAK=1); want_fp32=1;    want_p=unset ;;
  p0[0-9]*) want_p="0.${arm#p0}"
            arm_env=(MLX_E139_PROBE_FRACTION="${want_p}"); want_fp32=unset ;;
  fp32p0[0-9]*)
            want_p="0.${arm#fp32p0}"
            arm_env=(MLX_E139_FP32_TIEBREAK=1
                     MLX_E139_PROBE_FRACTION="${want_p}")
            want_fp32=1 ;;
  *) echo "e139_session: unknown arm ${arm}" >&2; exit 2 ;;
esac

if [[ "${want_p}" == "unset" ]]; then
  want_probes=$(python3 -c "import math;print(max(1,math.ceil(0.25*${leaves})))")
else
  want_probes=$(python3 -c "import math;print(max(1,math.ceil(${want_p}*${leaves})))")
fi

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

for id in "$@"; do
  [[ -s "$(prompt_file_for "${id}")" ]] || {
    echo "e139_session: no prompt text for ${id}" >&2; exit 2; }
done

for tool in jq python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "e139_session: missing ${tool}" >&2; exit 1; }
done
[[ -x "${swift_bin}" ]] || { echo "e139_session: missing ${swift_bin}" >&2; exit 1; }
[[ -f "${weights}/config.json" ]] || {
  echo "e139_session: missing ${weights}/config.json" >&2; exit 1; }

# STALENESS GUARD (HARNESS DEFECT 36). Neither local mode rebuilds the worker,
# so assert that the binary about to run carries both E139 gate strings before
# spending GPU on it. `grep -c`, not `grep -q`: under `set -o pipefail` a
# matching `grep -q` closes the pipe, `strings` dies of SIGPIPE with 141, and
# the guard then rejects exactly the fresh worker it should accept.
worker=".build-worker/release/mlxfast-runtime-worker"
for needle in MLX_E139_FP32_TIEBREAK MLX_E139_PROBE_FRACTION; do
  n="$(strings -a "${worker}" | grep -cF -- "${needle}")"
  ((n > 0)) || {
    echo "e139_session: ${worker} carries no ${needle}; rebuild it" >&2; exit 1; }
done

head_dir="${E139_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
head_verification="$(research/fetch-declared-head.sh 2>&1)" || {
  echo "e139_session: declared head verification failed" >&2
  echo "${head_verification}" >&2; exit 1; }
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e139_session: no run-tree MTP head at ${head_dir}" >&2; exit 1; }

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
  echo "e139_session: could not reuse benchmark.sh's run-lock definitions" >&2
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

status=0

for id in "$@"; do
  prompt="$(prompt_file_for "${id}")"
  out="${runs_root}/${arm}-${id}-r${rep}-t${tokens}"
  rm -rf "${out}"; mkdir -p "${out}"
  golden="${goldens_dir}/${id}-rows-$((tokens + 1)).json"

  # Reference rows on the CURRENT base, generated under the SHIPPED default
  # with no E139 gate exported. They are the serial argmax trajectory every
  # arm must reproduce, so no arm may generate its own reference.
  if ! jq -e --argjson n "$((tokens + 1))" '
        .reference_self_consistent == true and (.rows | length) >= $n
      ' "${golden}" >/dev/null 2>&1; then
    echo "=== e139: reference rows for ${id} (shipped default) ==="
    if [[ "${id}" == "benchfixture" ]]; then
      jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
        "${bench_fixture}" > "${out}/plan.json"
    else
      MLXFAST_NO_SANDBOX=1 "${swift_bin}" generate-golden \
        --prompt-file "${prompt}" \
        --weights "${weights}" \
        --output "${out}/seed.json" \
        --name "e139_${id}" \
        --steps 64 > "${out}/seed.log" 2>&1 || {
          echo "e139_session: ${id}: tokenizer pass failed" >&2
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
        echo "e139_session: ${id}: reference generation failed" >&2
        tail -20 "${out}/verify.log" >&2; status=1; continue; }
    jq -e --argjson n "$((tokens + 1))" '
        . as $g
        | $g.reference_self_consistent == true
          and ($g.rows | length) >= $n
          and ([range(0; ($g.rows | length))
                | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)]
               | length) == 0
      ' "${golden}" >/dev/null || {
        echo "e139_session: ${id}: reference rows unusable" >&2
        status=1; continue; }
  fi

  trace_path="${PWD}/${out}/trace.txt"
  : > "${trace_path}"
  entry_c="$(gpu_temp)"
  start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  worker_before="$(shasum -a 256 "${worker}" | awk '{print $1}')"

  echo "=== e139: ${arm} ${id} rep ${rep} (${tokens} tokens, offer ${depth}) ==="
  env "${arm_env[@]+"${arm_env[@]}"}" \
      MLXFAST_NO_SANDBOX=1 \
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
  worker_after="$(shasum -a 256 "${worker}" | awk '{print $1}')"

  sel_env="$(sed -n 's/.* sel_env=\([^ ]*\) .*/\1/p' "${trace_path}" | tail -1)"
  got_fp32="$(printf '%s' "${sel_env}" | sed -n 's/.*+e139fp32:\([^:+]*\):.*/\1/p')"
  got_fp32_drafts="$(printf '%s' "${sel_env}" | sed -n 's/.*+e139fp32:[^:+]*:\([0-9]*\).*/\1/p')"
  got_p="$(printf '%s' "${sel_env}" | sed -n 's/.*+e139p:\([^:+]*\):.*/\1/p')"
  got_probes="$(printf '%s' "${sel_env}" | sed -n 's/.*+e139p:[^:+]*:\([0-9]*\).*/\1/p')"

  {
    echo "experiment=e139-zero-noise-acceptance-instrument"
    echo "leg_kind=e139-acceptance-channel"
    echo "harness=local"
    echo "arm=${arm}"
    echo "rep=${rep}"
    echo "prompt_id=${id}"
    echo "prompt_file=${prompt}"
    echo "prompt_sha256=$(shasum -a 256 "${prompt}" | cut -d' ' -f1)"
    echo "golden=${golden}"
    echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)"
    echo "tokens=${tokens}"
    echo "offered_depth=${depth}"
    echo "asked_env=${arm_env[*]-none}"
    echo "witness_sel_env=${sel_env}"
    echo "witness_fp32_gate=${got_fp32}"
    echo "witness_fp32_rerank_drafts=${got_fp32_drafts}"
    echo "witness_probe_gate=${got_p}"
    echo "witness_probes=${got_probes}"
    echo "expected_fp32_gate=${want_fp32}"
    echo "expected_probe_gate=${want_p}"
    echo "expected_probes=${want_probes}"
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
    echo "head_manifest_tree_sha256=$(jq -r .sha256 mtp-head.manifest.json)"
    echo "worker_sha256=${worker_before}"
    echo "worker_sha256_after_leg=${worker_after}"
    echo "cli_sha256=$(shasum -a 256 "${swift_bin}" | awk '{print $1}')"
    echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
    echo "host=$(hostname)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string)"
    echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
    echo "started_utc=${start_iso}"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}")"
    if [[ -s "${out}/report.json" ]]; then
      jq -r '"all_tokens_matched=\(.all_tokens_matched)",
             "residual_divergence_count=\(.residual_divergence_count)",
             "round_count=\(.round_count)",
             "effective_mean_draft_len=\(.effective_mean_draft_len)",
             "effective_max_draft_len=\(.effective_max_draft_len)",
             "accepted_draft_rate=\(.accepted_draft_rate)",
             "accepted_draft_total=\(.accepted_draft_total)",
             "rejected_draft_total=\(.rejected_draft_total)",
             "decode_tokens=\(.decode_tokens // "none")",
             "head_provenance_sha256=\(.head_provenance.sha256 // "none")"' \
        "${out}/report.json"
    fi
    echo "exit=${rc}"
  } > "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e139_session: ${id}: mtp-timed exited ${rc}" >&2
    tail -5 "${out}/stderr.log" >&2
    status=1
    continue
  fi
  if [[ "${worker_before}" != "${worker_after}" ]]; then
    echo "e139_session: ${arm}/${id}: the worker moved during the leg" >&2
    status=1
  fi
  if [[ "${got_fp32}" != "${want_fp32}" || "${got_p}" != "${want_p}" \
        || "${got_probes}" != "${want_probes}" ]]; then
    echo "e139_session: ${arm}/${id}: witness disagrees with the asked arm:" \
         "fp32=${got_fp32} (want ${want_fp32})" \
         "p=${got_p} (want ${want_p})" \
         "probes=${got_probes} (want ${want_probes})" >&2
    status=1
  fi
  if [[ "${got_fp32_drafts:-0}" == "0" ]]; then
    echo "e139_session: ${arm}/${id}: no draft reached the rerank kernel" >&2
    status=1
  fi
  jq -r '"  matched=\(.all_tokens_matched) divergence=\(.residual_divergence_count)"
    + " rounds=\(.round_count) draft=\(.effective_mean_draft_len)"
    + " accept=\(.accepted_draft_rate)"' "${out}/report.json"
done

exit "${status}"
