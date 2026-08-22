#!/usr/bin/env bash
# E134 item 1 -- warm-refill arms, measured on the local 48 GiB host.
#
#   usage: research/e134_warm_session.sh ARM PROMPT_ID [PROMPT_ID ...]
#
# ARMS. Each arm is the SAME binary; only two DARKBLOOM_ switches differ.
#
#   base         refill off, no emulated clear   -- the shipped local path
#   refill       refill ON,  no emulated clear   -- the new default, locally
#   clear        refill off, emulated clear ON   -- the ranked allocator state
#   clearrefill  refill ON,  emulated clear ON   -- ranked state plus the fix
#
# WHY THE EMULATION EXISTS. `wireResidentWeightsIfEnabled()` is gated on
# `physicalMemory >= 96 GiB` and calls `Memory.clearCache()` after the shape
# warm. This host has 48 GiB, so that path returns early and the refill has
# nothing to repair here. `DARKBLOOM_QWEN_MTP_EMULATE_RESIDENCY_CLEAR=1`
# reproduces only the allocator side effect, never the wired ticket, which the
# campaign has established has no local instrument (ledger items 140, 141,
# 200(H)).
#
# NOT A TIMING SESSION for score purposes: every leg runs the per-round phase
# trace and no cool gate. `timing_valid=false` is recorded verbatim. The
# instruments are within-leg or arm-differenced:
#
#   seed_prefill_seconds   the first timed forward after warm
#   first_block_seconds    the first scored round
#   round-1 excess         from trace.txt, against a width-matched tail
#
# HARNESS DEFECT 28: only MLX_, DARKBLOOM_, METAL_, MTL_, DYLD_ and LC_
# prefixed names survive `sanitizedRuntimeWorkerEnvironment`, which is why both
# switches carry the DARKBLOOM_ prefix.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(($# >= 2)) || {
  echo "usage: research/e134_warm_session.sh ARM PROMPT_ID [...]" >&2; exit 2; }

arm="$1"; shift
case "${arm}" in
  base)        arm_env=(DARKBLOOM_QWEN_MTP_WARM_REFILL=0) ;;
  refill)      arm_env=(DARKBLOOM_QWEN_MTP_WARM_REFILL=1) ;;
  clear)       arm_env=(DARKBLOOM_QWEN_MTP_WARM_REFILL=0
                        DARKBLOOM_QWEN_MTP_EMULATE_RESIDENCY_CLEAR=1) ;;
  clearrefill) arm_env=(DARKBLOOM_QWEN_MTP_WARM_REFILL=1
                        DARKBLOOM_QWEN_MTP_EMULATE_RESIDENCY_CLEAR=1) ;;
  *) echo "e134_warm_session: unknown arm ${arm}" >&2; exit 2 ;;
esac

tokens="${E134_TOKENS:-512}"
depth="${E134_DEPTH:-8}"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
root="${E134_ROOT:-.mlxfast-private/e134}"
goldens_dir="${E134_GOLDENS_DIR:-.mlxfast-private/e128/goldens}"
runs_root="${root}/runs-${arm}"
bench_fixture="correctness_prompts/public_longcopy_gate_english_512_256.json"

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
    echo "e134_warm_session: no prompt text for ${id}" >&2; exit 2; }
  [[ -s "${goldens_dir}/${id}-rows-$((tokens + 1)).json" ]] || {
    echo "e134_warm_session: no golden for ${id}; run e128_session first" >&2
    exit 2; }
done

for tool in jq python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "e134_warm_session: missing ${tool}" >&2; exit 1; }
done
[[ -x "${swift_bin}" ]] || {
  echo "e134_warm_session: missing ${swift_bin}" >&2; exit 1; }
[[ -f "${weights}/config.json" ]] || {
  echo "e134_warm_session: missing ${weights}/config.json" >&2; exit 1; }

head_dir="${E134_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
head_verification="$(research/fetch-declared-head.sh 2>&1)" || {
  echo "e134_warm_session: declared head verification failed" >&2
  echo "${head_verification}" >&2
  exit 1; }
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e134_warm_session: no run-tree MTP head at ${head_dir}" >&2; exit 1; }

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
  echo "e134_warm_session: could not reuse benchmark.sh's run-lock definitions" >&2
  exit 1; }
trap 'release_local_run_lock' EXIT
acquire_local_run_lock
abort_if_model_already_resident

mkdir -p "${runs_root}"

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
  if [[ -s "${out}/report.json" ]] && grep -qx 'exit=0' "${out}/meta.txt" 2>/dev/null \
     && [[ "${E134_FORCE:-0}" != "1" ]]; then
    echo "=== e134: ${arm}/${id} already collected, keeping it ==="
    continue
  fi
  rm -rf "${out}"; mkdir -p "${out}"
  golden="${goldens_dir}/${id}-rows-$((tokens + 1)).json"

  trace_path="${PWD}/${out}/trace.txt"
  : > "${trace_path}"
  entry_c="$(gpu_temp)"
  start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  leg_env=(MLXFAST_NO_SANDBOX=1
           MLX_QWEN_MTP_TRACE=1
           MLX_QWEN_MTP_TRACE_PATH="${trace_path}"
           "${arm_env[@]}")

  echo "=== e134: ${arm} leg ${id} (${tokens} tokens, offer ${depth}) ==="
  env "${leg_env[@]}" \
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
    echo "experiment=e134-oracle-discrimination-at-the-m6-cliff"
    echo "leg_kind=e134-item1-warm-refill-${arm}"
    echo "arm=${arm}"
    echo "arm_env=${arm_env[*]}"
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
    echo "head_manifest_tree_sha256=$(jq -r .sha256 mtp-head.manifest.json)"
    echo "worker_sha256=$(shasum -a 256 "${worker}" | awk '{print $1}')"
    echo "cli_sha256=$(shasum -a 256 "${swift_bin}" | awk '{print $1}')"
    echo "host=$(hostname)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string)"
    echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
    echo "started_utc=${start_iso}"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}")"
    grep -m1 '^mlxfast: qwen-mtp warm ' "${out}/stderr.log" \
      | sed 's/^mlxfast: qwen-mtp warm /warm_/' \
      | tr ' ' '\n' | sed '/^$/d' | sed 's/^shapes_ms/warm_shapes_ms/' \
      | grep -E '^(warm_|refill|emulated_clear|cache_|active_)' || true
    if [[ -s "${out}/report.json" ]]; then
      jq -r '"all_tokens_matched=\(.all_tokens_matched)",
             "residual_divergence_count=\(.residual_divergence_count)",
             "seed_prefill_seconds=\(.seed_prefill_seconds)",
             "first_block_seconds=\(.first_block_seconds)",
             "decode_seconds=\(.decode_seconds)",
             "parent_measured_seconds_per_token=\(.parent_measured_seconds_per_token)",
             "p50_block_request_seconds=\(.p50_block_request_seconds)",
             "round_count=\(.round_count)",
             "effective_mean_draft_len=\(.effective_mean_draft_len)",
             "accepted_draft_rate=\(.accepted_draft_rate)",
             "head_provenance_sha256=\(.head_provenance.sha256 // "none")"' \
        "${out}/report.json"
    fi
    echo "exit=${rc}"
  } > "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e134_warm_session: ${arm}/${id}: mtp-timed exited ${rc}" >&2
    tail -5 "${out}/stderr.log" >&2
    status=1
    continue
  fi
  jq -r '"  matched=\(.all_tokens_matched) prefill=\(.seed_prefill_seconds)"
    + " first_block=\(.first_block_seconds) rounds=\(.round_count)"
    + " draft=\(.effective_mean_draft_len)"' "${out}/report.json"
done

exit "${status}"
