#!/usr/bin/env bash
# Run one E83 seed-prefill decomposition session.
#
#   usage: research/e83_prefill.sh TAG [smoke|full]
#
# The session is a Tests/-only instrument. It holds the real checkpoint, so it
# takes the same local run lock benchmark.sh takes: two overlapping resident
# models out-of-memory this host.
#
# MLXFAST_LOCAL_COOL_GATE has no effect here, because this session never calls
# benchmark.sh. The session is instead ABBA-counterbalanced inside the test and
# records entry and exit GPU temperature per block. Every result carries
# cool_gate_passed_real_gate=false, gate_qualified_for_timing=false and
# official_or_ranked_score=false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e83_prefill.sh TAG [PROFILE]}"
profile="${2:-full}"

run_prefill=1
run_head=0
case "${profile}" in
  smoke)
    # Every arm installer and both probe kinds run once. This proves the module
    # surgery and the swizzle on the real checkpoint without spending a session.
    : "${MLXFAST_E83_REPS:=1}"
    : "${MLXFAST_E83_WARMUP:=1}"
    : "${MLXFAST_E83_LADDER_REPS:=1}"
    : "${MLXFAST_E83_ARMS:=null,mlp_all,gdn_in_qkv,all_interceptable}"
    ;;
  full) : "${MLXFAST_E83_LADDER_REPS:=6}" ;;
  followup)
    # Re-scores the positive control against the stall the host actually
    # delivers, and prices one pinned proposal-head step. The full session
    # already measured the arms, the ladder and the roofline.
    : "${MLXFAST_E83_REPS:=3}"
    : "${MLXFAST_E83_WARMUP:=1}"
    : "${MLXFAST_E83_LADDER_REPS:=0}"
    : "${MLXFAST_E83_ARMS:=none}"
    : "${MLXFAST_E83_ISOLATED:=0}"
    run_head=1
    ;;
  gates)
    # Rung 3 ran the two prefill-width fusion gates from this profile. The arms
    # need two switches inside the vendored model, and that instrument is
    # reverted, so the profile cannot run on this tree. The recorded session is
    # research/results/e83/gates.json and research/e83_report.py still reads it.
    echo "e83_prefill.sh: the gates profile needs the reverted rung-3 instrument." >&2
    echo "  results:  research/e83_report.py research/results/e83/gates.json" >&2
    echo "  replay:   check out the instrument commit named in" >&2
    echo "            research/results/qwen38-r1-e83-prefill-decomposition.md" >&2
    exit 2
    ;;
  head) run_prefill=0; run_head=1 ;;
  *) echo "e83_prefill.sh: unknown profile ${profile}" >&2; exit 2 ;;
esac
export MLXFAST_E83_REPS MLXFAST_E83_WARMUP MLXFAST_E83_ARMS MLXFAST_E83_LADDER_REPS
export MLXFAST_E83_ISOLATED

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

# --- local run guard: reuse benchmark.sh's definitions verbatim ---------------
local_run_guard_enabled() {
  [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]
}
run_lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
if ! eval "${run_lock_definitions}"; then
  echo "e83_prefill.sh: could not evaluate benchmark.sh's local run guard;" >&2
  echo "e83_prefill.sh: refusing to run unguarded" >&2
  exit 1
fi
for reused in local_run_lock_path acquire_local_run_lock release_local_run_lock \
              list_resident_model_processes abort_if_model_already_resident; do
  if ! declare -F "${reused}" >/dev/null 2>&1; then
    echo "e83_prefill.sh: could not reuse benchmark.sh's ${reused}();" >&2
    echo "e83_prefill.sh: refusing to run unguarded" >&2
    exit 1
  fi
done

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
LOCAL_RUN_LOCK_OWNED=""
cleanup() { release_local_run_lock; }
trap cleanup EXIT
abort_if_model_already_resident || exit 1
acquire_local_run_lock || exit 1

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

export MLXFAST_RUN_E83_PREFILL="${run_prefill}"
export MLXFAST_RUN_E83_HEAD="${run_head}"
export MLXFAST_E83_OUT="${PWD}/${out}/decomposition.json"
export MLXFAST_E83_HEAD_OUT="${PWD}/${out}/head_step.json"

experiment="${MLXFAST_CENSUS_EXPERIMENT:-e83-prefill-decomposition}"
group="${MLXFAST_CENSUS_GROUP:-e83-prefill-decomposition}"

{
  echo "tag=${tag}"
  echo "experiment=${experiment}"
  echo "harness=local"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "gpu_cores=$(ioreg -l 2>/dev/null \
    | LC_ALL=C sed -n 's/.*"gpu-core-count" = \([0-9][0-9]*\).*/\1/p' | head -1)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "os=$(sw_vers -productVersion)"
  echo "swift=$(swift --version 2>&1 | head -1)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "seed_length=${MLXFAST_E83_SEED_LEN:-512}"
  echo "reps=${MLXFAST_E83_REPS:-5}"
  echo "warmup=${MLXFAST_E83_WARMUP:-2}"
  echo "arms=${MLXFAST_E83_ARMS:-<test-default>}"
  echo "profile=${profile}"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

set -o pipefail
swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
  --filter E83PrefillDecompositionTests 2>&1 \
  | python3 research/e83_wandb_stream.py \
      --tag "${tag}" --meta "${out}/meta.txt" --log "${out}/session.log" \
      --experiment "${experiment}" --group "${group}"
status=${PIPESTATUS[0]}

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
