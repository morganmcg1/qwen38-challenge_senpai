#!/usr/bin/env bash
# Run one E71 in-situ width-tax census session.
#
#   usage: research/e71_census.sh TAG [env overrides via the environment]
#
# The census is a Tests/-only instrument. It holds the real checkpoint, so it
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

tag="${1:?usage: e71_census.sh TAG [PROFILE]}"
profile="${2:-full}"

# Profiles set defaults only; an explicit environment variable always wins, so
# the same driver serves the cheap plumbing check and the real census.
case "${profile}" in
  smoke)
    # Every arm installer runs once at the headline width. This proves the
    # module surgery on the real checkpoint without spending a census.
    : "${MLXFAST_E71_CURVE_WIDTHS:=1,6}"
    : "${MLXFAST_E71_ARM_WIDTHS:=6}"
    : "${MLXFAST_E71_REPS:=2}"
    : "${MLXFAST_E71_WARMUP:=1}"
    ;;
  full) ;;
  *) echo "e71_census.sh: unknown profile ${profile}" >&2; exit 2 ;;
esac
export MLXFAST_E71_CURVE_WIDTHS MLXFAST_E71_ARM_WIDTHS \
       MLXFAST_E71_REPS MLXFAST_E71_WARMUP

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

# --- local run guard: reuse benchmark.sh's definitions verbatim ---------------
LOCAL_RUN_LOCK_OWNED=""
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
  echo "e71_census.sh: could not evaluate benchmark.sh's local run guard;" >&2
  echo "e71_census.sh: refusing to run unguarded" >&2
  exit 1
fi
for reused in local_run_lock_path acquire_local_run_lock release_local_run_lock \
              list_resident_model_processes abort_if_model_already_resident; do
  if ! declare -F "${reused}" >/dev/null 2>&1; then
    echo "e71_census.sh: could not reuse benchmark.sh's ${reused}();" >&2
    echo "e71_census.sh: refusing to run unguarded" >&2
    exit 1
  fi
done

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
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

export MLXFAST_RUN_E71_WIDTH_TAX=1
export MLXFAST_E71_OUT="${PWD}/${out}/census.json"

experiment="${MLXFAST_CENSUS_EXPERIMENT:-e71-in-situ-width-tax-census}"
group="${MLXFAST_CENSUS_GROUP:-e71-width-tax-census}"

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
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "seed_length=${MLXFAST_E71_SEED_LEN:-768}"
  echo "reps=${MLXFAST_E71_REPS:-12}"
  echo "warmup=${MLXFAST_E71_WARMUP:-3}"
  echo "curve_widths=${MLXFAST_E71_CURVE_WIDTHS:-1,2,3,4,5,6,7,8,9}"
  echo "arm_widths=${MLXFAST_E71_ARM_WIDTHS:-4,5,6,9}"
  echo "arms=${MLXFAST_E71_ARMS:-null,lm_head,mlp_all,mlp_down,fa_o_proj,gdn_out_proj,all_interceptable}"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

set -o pipefail
swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
  --filter E71WidthTaxCensusTests 2>&1 \
  | python3 research/e71_wandb_stream.py \
      --tag "${tag}" --meta "${out}/meta.txt" --log "${out}/session.log" \
      --experiment "${experiment}" --group "${group}"
status=${PIPESTATUS[0]}

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
