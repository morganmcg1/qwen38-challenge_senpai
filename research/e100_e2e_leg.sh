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
# MLXFAST_QWEN_MTP_DEPTH is the per-round draft ceiling the parent offers, and
# it is recorded per leg. Two sessions answer two different questions:
#
#   depth 8   the shipped schedule. Does the change pay where the solver
#             actually runs? `segmentedVerifyDepthCap` is 7, so the verify
#             width is 1..8 and M = 5 is one width among several.
#   depth 4   the reach-free control. The offer caps draftCount at 4, so the
#             verify width is at most 5 and M = 5 dominates. This isolates how
#             well the kernel gain CONVERTS from how often it is REACHED.
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

depth="${MLXFAST_QWEN_MTP_DEPTH:-8}"
tokens="${MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS:-64}"

MLXFAST_LOCAL_COOL_GATE=0 \
MLXFAST_QWEN_MTP_DEPTH="${depth}" \
MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}" \
MLXFAST_SCORE_PATH="${PWD}/${out_dir}/score.json" \
./benchmark-qwen-mtp.sh --local-iterate 2>&1 | tee "${out_dir}/run.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e100-fewer-weight-streams-per-round"
  echo "leg_kind=end-to-end-local-iterate"
  echo "arm=${arm}"
  echo "harness=local"
  echo "started_utc=${start_iso}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  # Only these paths reach the built worker. A research-only file left
  # uncommitted while a leg runs is recorded but does not invalidate the leg.
  echo "git_dirty_build=$(git status --porcelain -- Sources Vendor Package.swift \
    Package.resolved tools mtp-head.manifest.json | wc -l | tr -d ' ')"
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
  echo "offered_depth=${depth}"
  echo "decode_tokens=${tokens}"
  echo "status=${status}"
} > "${out_dir}/meta.txt"

cat "${out_dir}/meta.txt"
jq -r '.metrics | "score=\(.mtp_decode_speedup) mtp_spt=\(.mtp_seconds_per_token) serial_spt=\(.serial_seconds_per_token) mean_draft=\(.effective_mean_draft_len) accept=\(.accepted_draft_rate) matched=\(.all_tokens_matched)"' \
  "${out_dir}/score.json" 2>/dev/null || true
exit "${status}"
