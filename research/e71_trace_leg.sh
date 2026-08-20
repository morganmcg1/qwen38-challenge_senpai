#!/usr/bin/env bash
# Run one traced, UNCHANGED-base leg of ./benchmark-qwen-mtp.sh for E71 rung 1.
#
#   usage: research/e71_trace_leg.sh TAG TOKENS
#
# This leg stages nothing and builds nothing of its own. Its only purpose is to
# produce a same-host, same-day in-situ V(M) = verify_build_us + eval_wall_us
# per verify width, so the Tests/-only census harness can be gated against a
# real scored leg instead of only against a historical ledger row.
#
# MLXFAST_LOCAL_COOL_GATE=0 is set. Entry and exit GPU temperature are recorded.
# Every result carries cool_gate_passed_real_gate=false,
# gate_qualified_for_timing=false and official_or_ranked_score=false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e71_trace_leg.sh TAG TOKENS}"
tokens="${2:?usage: e71_trace_leg.sh TAG TOKENS}"

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export MLXFAST_LOCAL_COOL_GATE=0
export MLXFAST_NO_SANDBOX=1

trace_path="${PWD}/${out}/trace.txt"
: > "${trace_path}"
export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"

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

{
  echo "tag=${tag}"
  echo "experiment=e71-in-situ-width-tax-census"
  echo "rung=1"
  echo "harness=local"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "tokens=${tokens}"
  echo "local_mode=--local-iterate"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate \
  > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}" || true)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
