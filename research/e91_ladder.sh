#!/usr/bin/env bash
# Run one E91 seed-prefill session.
#
#   usage: research/e91_ladder.sh TAG [smoke|ladder|ceiling]
#
#   smoke    two arms, one rep, plus the boundary census. Proves the knob and
#            the harness on the real checkpoint without spending a session.
#   ladder   rung 1. Nine schedules, ABBA, three reps, plus the census.
#   ceiling  rung 2. Synthetic weights only, so the 15 GB checkpoint is never
#            resident and the probe is cheap.
#
# Tests/-only instrument. The ladder profiles hold the real checkpoint, so they
# take the same local run lock benchmark.sh takes: two overlapping resident
# models out-of-memory this host.
#
# MLXFAST_LOCAL_COOL_GATE has no effect here, because this session never calls
# benchmark.sh. The session is instead ABBA-counterbalanced inside the test and
# records entry and exit GPU temperature per block. Every result carries
# cool_gate_passed_real_gate=false, gate_qualified_for_timing=false and
# official_or_ranked_score=false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e91_ladder.sh TAG [PROFILE]}"
profile="${2:-ladder}"

run_ladder=1
run_ceiling=0
case "${profile}" in
  smoke)
    : "${MLXFAST_E91_REPS:=1}"
    : "${MLXFAST_E91_WARMUP:=1}"
    : "${MLXFAST_E91_ARMS:=s1,off}"
    ;;
  ladder)
    : "${MLXFAST_E91_REPS:=3}"
    : "${MLXFAST_E91_WARMUP:=2}"
    ;;
  ceiling)
    run_ladder=0
    run_ceiling=1
    ;;
  *) echo "e91_ladder.sh: unknown profile ${profile}" >&2; exit 2 ;;
esac
export MLXFAST_E91_REPS MLXFAST_E91_WARMUP MLXFAST_E91_ARMS MLXFAST_E91_CENSUS

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

# --- local run guard: reuse benchmark.sh's definitions verbatim ---------------
run_lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
if ! eval "${run_lock_definitions}"; then
  echo "e91_ladder.sh: could not evaluate benchmark.sh's local run guard;" >&2
  echo "e91_ladder.sh: refusing to run unguarded" >&2
  exit 1
fi
for reused in local_run_lock_path acquire_local_run_lock release_local_run_lock \
              list_resident_model_processes abort_if_model_already_resident; do
  if ! declare -F "${reused}" >/dev/null 2>&1; then
    echo "e91_ladder.sh: could not reuse benchmark.sh's ${reused}();" >&2
    echo "e91_ladder.sh: refusing to run unguarded" >&2
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

export MLXFAST_RUN_E91_LADDER="${run_ladder}"
export MLXFAST_RUN_E91_CEILING="${run_ceiling}"
export MLXFAST_E91_OUT="${PWD}/${out}/ladder.json"
export MLXFAST_E91_CEILING_OUT="${PWD}/${out}/ceiling.json"

experiment="${MLXFAST_CENSUS_EXPERIMENT:-e91-prefill-ladder}"
group="${MLXFAST_CENSUS_GROUP:-e91-prefill-ladder}"

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
  echo "seed_length=${MLXFAST_E91_SEED_LEN:-512}"
  echo "reps=${MLXFAST_E91_REPS:-3}"
  echo "warmup=${MLXFAST_E91_WARMUP:-2}"
  echo "arms=${MLXFAST_E91_ARMS:-<test-default>}"
  echo "profile=${profile}"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

set -o pipefail
swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
  --filter E91PrefillLadderTests 2>&1 \
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
