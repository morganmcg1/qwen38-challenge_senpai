#!/usr/bin/env bash
# Price one materialised intermediate by ADDING intermediates to the per-draft
# proposal path instead of removing them.
#
#   usage: research/e85_tax_session.sh SESSION_TAG TOKENS REPEATS K1 K2 K3
#
# Each repeat runs the tax levels forward and then in reverse, so the order is
# palindromic (K1 K2 K3 K3 K2 K1). A linear thermal drift cancels for every
# level, exactly as ABBA does for two arms.
#
# Removing the six intermediates this experiment targets moves decode time by
# less than one third of one leg's noise. Adding K of them multiplies the same
# per-buffer coefficient by K. At K=128 the campaign's claimed 13-16 us per
# buffer predicts +5.1 to +6.2 percent of candidate time, while the E85 census
# predicts +0.6 to +1.1 percent. Both are far above the 0.43 percent per-leg
# residual, and they differ from each other by a factor of eight.
#
# The slope must also exceed the 0.66-1.55 us host dispatch cost measured in
# E80. A slope near zero would mean MLX elided the tax rather than that buffers
# are free, and would have to be settled with a census leg before any claim.
#
# The cool gate is DISABLED, which is legal for a local timed arm under the
# three standing conditions. This script counterbalances inside one session,
# records entry and exit GPU temperature for every leg, and preserves
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false verbatim.
# A leg from this script is never a gate-qualified or ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="${1:?usage: e85_tax_session.sh SESSION_TAG TOKENS REPEATS K1 K2 K3}"
tokens="${2:?usage: e85_tax_session.sh SESSION_TAG TOKENS REPEATS K1 K2 K3}"
repeats="${3:?usage: e85_tax_session.sh SESSION_TAG TOKENS REPEATS K1 K2 K3}"
shift 3
levels=("$@")
((${#levels[@]} >= 2)) || { echo "e85_tax_session.sh: need >= 2 tax levels" >&2; exit 2; }

head_dir="${E85_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e85_tax_session.sh: no head at ${head_dir}" >&2
  exit 1
fi

root="research/out/${session}"
rm -rf "${root}"
mkdir -p "${root}"

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

reversed=()
for ((i = ${#levels[@]} - 1; i >= 0; --i)); do reversed+=("${levels[i]}"); done
order=("${levels[@]}" "${reversed[@]}")

worker_sha="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

{
  echo "session=${session}"
  echo "experiment=e85-buffer-tax-slope"
  echo "tokens=${tokens}"
  echo "repeats=${repeats}"
  echo "tax_levels=${levels[*]}"
  echo "order=palindrome ${order[*]} x ${repeats}"
  echo "arms=MLX_E85_FUSED_EMBED=1 MLX_E85_GATHER_QMM=1 (fixed for every leg)"
  echo "local_mode=--local-iterate"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "head_dir=${head_dir}"
  echo "worker_sha256=${worker_sha}"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${root}/session.txt"

printf 'leg\ttax\tmtp_s_per_tok\tserial_s_per_tok\tratio\tmean_draft_len\taccepted_rate\tmatched\ttemp_in\ttemp_out\tseconds\n' \
  > "${root}/legs.tsv"

leg=0
status=0
for ((r = 0; r < repeats; ++r)); do
  for tax in "${order[@]}"; do
    leg=$((leg + 1))
    out="${root}/leg$(printf '%02d' "${leg}")-k${tax}"
    mkdir -p "${out}"

    temp_in="$(gpu_temp)"
    started_s=$(date +%s)
    (
      export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
      export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
      export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
      export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
      export MLXFAST_NO_SANDBOX=1
      export MLXFAST_LOCAL_COOL_GATE=0
      export MLX_E85_FUSED_EMBED=1
      export MLX_E85_GATHER_QMM=1
      export MLX_E85_BUFFER_TAX="${tax}"
      ./benchmark-qwen-mtp.sh --local-iterate \
        > "${out}/wrapper.out" 2> "${out}/wrapper.err"
    )
    leg_status=$?
    elapsed=$(( $(date +%s) - started_s ))
    temp_out="$(gpu_temp)"

    if ((leg_status != 0)); then
      echo "e85_tax_session.sh: leg ${leg} (K=${tax}) exited ${leg_status}" >&2
      status="${leg_status}"
      break
    fi

    row="$(python3 - "${out}/score.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
print("\t".join(str(m[k]) for k in (
    "mtp_seconds_per_token", "serial_seconds_per_token", "mtp_decode_speedup",
    "effective_mean_draft_len", "accepted_draft_rate", "all_tokens_matched")))
PY
)"
    printf '%d\t%s\t%s\t%s\t%s\t%d\n' \
      "${leg}" "${tax}" "${row}" "${temp_in}" "${temp_out}" "${elapsed}" \
      >> "${root}/legs.tsv"

    if [[ "${row}" != *"True"* ]]; then
      echo "e85_tax_session.sh: leg ${leg} (K=${tax}) reported all_tokens_matched != true" >&2
      status=3
      break
    fi
  done
  ((status == 0)) || break
done

{
  echo "legs_completed=${leg}"
  echo "post_run_worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${root}/session.txt"

exit "${status}"
