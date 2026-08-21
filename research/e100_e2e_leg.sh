#!/usr/bin/env bash
# E100 -- run one end-to-end --local-iterate leg and record the identity tuple
# around it.
#
#   usage: research/e100_e2e_leg.sh TAG ARM
#
#   TAG   output directory name under research/out/
#   ARM   `base` or `collapse`, recorded verbatim.
#
# The two arms are two BUILDS, because the dispatch table is a compile-time
# template selection. The session is counterbalanced A B B A across four legs.
#
# MLXFAST_LOCAL_COOL_GATE=0 is the standing permitted local measurement mode:
# the arms are ABBA-counterbalanced within one session, entry and exit GPU
# temperature are recorded for every leg, and the honesty flags below stay
# false. This leg is directional causal evidence, never a ranked score.
#
# Set MLXFAST_E100_TRACE=1 for a DIAGNOSTIC leg. It emits one `mtp-trace:` line
# per round to a file, which gives the verify width histogram that decides
# whether the M = 5 dispatch entry is reached at all. The round trace adds host
# work, so a traced leg is never a timed leg of the ABBA session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e100_e2e_leg.sh TAG ARM}"
arm="${2:?usage: e100_e2e_leg.sh TAG ARM}"
out_dir="research/out/${tag}"
mkdir -p "${out_dir}"

twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

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

entry_c="$(gpu_temp)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

trace_path=""
if [[ "${MLXFAST_E100_TRACE:-0}" == "1" ]]; then
  trace_path="${PWD}/${out_dir}/rounds.trace"
  : > "${trace_path}"
fi

MLXFAST_LOCAL_COOL_GATE=0 \
MLX_QWEN_MTP_TRACE="${MLXFAST_E100_TRACE:-0}" \
MLX_QWEN_MTP_TRACE_PATH="${trace_path}" \
MLXFAST_SCORE_PATH="${PWD}/${out_dir}/score.json" \
./benchmark-qwen-mtp.sh --local-iterate 2>&1 | tee "${out_dir}/run.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

# Verify width histogram. `d` is the draft count, so the verify width is d + 1.
if [[ -s "${trace_path:-/dev/null}" ]]; then
  grep -o ' d=[0-9]*' "${trace_path}" | tr -d ' d=' | sort -n | uniq -c \
    > "${out_dir}/draft_hist.txt"
fi

{
  echo "experiment=e100-fewer-weight-streams-per-round"
  echo "leg_kind=end-to-end-local-iterate"
  echo "arm=${arm}"
  echo "harness=local"
  echo "started_utc=${start_iso}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "twin_m5=$(grep -c 'qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>' "${twin}")"
  echo "twin_m9=$(grep -c 'qmv_fast_crossrow_affine4_g64_m<T, 9, 5, true>' "${twin}")"
  echo "twin_na_bound=$(grep -o 'wide multi-row QMV supports NA in \[2, [0-9]\]' "${twin}" | head -1)"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "toolchain=$(swift --version 2>&1 | head -1)"
  echo "gpu_temp_entry_c=${entry_c}"
  echo "gpu_temp_exit_c=${exit_c}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "timing_valid=false"
  echo "traced=${MLXFAST_E100_TRACE:-0}"
  echo "status=${status}"
} > "${out_dir}/meta.txt"

cat "${out_dir}/meta.txt"
if [[ -s "${out_dir}/draft_hist.txt" ]]; then
  echo "--- draft-count histogram (count, d); verify width is d + 1 ---"
  cat "${out_dir}/draft_hist.txt"
fi
exit "${status}"
