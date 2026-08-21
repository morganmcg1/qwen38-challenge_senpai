#!/usr/bin/env bash
# E109 rung 0 -- ONE timed native-MTP decode leg of the resolution protocol.
#
#   usage: research/e109_ab_leg.sh OUT_DIR ARM_LABEL [ENV=VALUE ...]
#
# This is the protocol's unit of measurement. It drives the SAME trusted verb
# the local runner and the ranked workflow drive -- `mlxfast-swift mtp-timed`
# -- against an already-generated reference row file, and nothing else. The
# tripwire, the reference pass and the serial control run ONCE per session in
# research/e109_ab_session.sh instead of once per leg, because none of them is
# part of the measured quantity and each of them costs about as much as the
# leg itself.
#
# WHY NOT `benchmark-qwen-mtp.sh --local-iterate` PER ARM. That wrapper spends
# about 145 s to produce about 1.3 s of decode-only MTP signal at its default
# 64 tokens: four model-loading processes, of which only the fourth is the
# number we read. At 512 tokens this leg spends about 30 s to produce about
# 10.6 s of decode-only signal. Same trusted timing path, same worker, ~26x
# more measured GPU time per wall-clock second.
#
# WHAT IT RECORDS. `mtp-timed --output` writes the parent's own report, which
# already carries `block_request_seconds`: one wall-clock seconds entry per
# decode ROUND, measured by the trusted parent. That array, not the leg's
# seconds-per-token scalar, is the protocol's endpoint. It is a decode-only
# round time by construction, so campaign rule 34 is satisfied by the
# instrument rather than by a two-point seed model.
#
# THERMAL. MLXFAST_LOCAL_COOL_GATE is not consulted by `mtp-timed`; the gate
# lives in the wrapper. This leg therefore always runs ungated, records entry
# and exit GPU temperature, and keeps `cool_gate_passed_real_gate=false` and
# `gate_qualified_for_timing=false`. Directional causal evidence inside one
# counterbalanced session only, never a ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out_dir="${1:?usage: e109_ab_leg.sh OUT_DIR ARM_LABEL [ENV=VALUE ...]}"
arm_label="${2:?usage: e109_ab_leg.sh OUT_DIR ARM_LABEL [ENV=VALUE ...]}"
shift 2

: "${E109_GOLDEN:?e109_ab_leg.sh: E109_GOLDEN must name the reference row file}"
: "${E109_HEAD_DIR:?e109_ab_leg.sh: E109_HEAD_DIR must name the MTP head}"
tokens="${E109_TOKENS:-512}"
depth="${E109_DEPTH:-8}"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"

mkdir -p "${out_dir}"

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
start_epoch="$(python3 -c 'import time; print(time.time())')"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

env "$@" \
  MLXFAST_NO_SANDBOX=1 \
  "${swift_bin}" mtp-timed \
    --weights "${weights}" \
    --mtp-head "${E109_HEAD_DIR}" \
    --golden "${E109_GOLDEN}" \
    --tokens "${tokens}" \
    --mtp-depth "${depth}" \
    --output "${out_dir}/report.json" \
  > "${out_dir}/stdout.json" 2> "${out_dir}/stderr.log"
status=$?

end_epoch="$(python3 -c 'import time; print(time.time())')"
exit_c="$(gpu_temp)"
worker=".build-worker/release/mlxfast-runtime-worker"

{
  echo "experiment=e109-resolve-the-bar-then-latency-residual"
  echo "leg_kind=e109-ab-protocol-mtp-timed"
  echo "harness=local"
  echo "arm_label=${arm_label}"
  echo "arm_env=$*"
  echo "tokens=${tokens}"
  echo "offered_depth=${depth}"
  echo "started_utc=${start_iso}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "leg_wall_seconds=$(python3 -c "print(f'{${end_epoch}-${start_epoch}:.3f}')")"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty_build=$(git status --porcelain -- Sources Vendor Package.swift \
    Package.resolved tools mtp-head.manifest.json | wc -l | tr -d ' ')"
  echo "worker_sha256=$(shasum -a 256 "${worker}" | awk '{print $1}')"
  echo "worker_mtime=$(date -u -r "${worker}" +%Y-%m-%dT%H:%M:%SZ)"
  echo "cli_sha256=$(shasum -a 256 "${swift_bin}" | awk '{print $1}')"
  echo "golden=${E109_GOLDEN}"
  echo "golden_sha256=$(shasum -a 256 "${E109_GOLDEN}" | awk '{print $1}')"
  echo "head_dir=${E109_HEAD_DIR}"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "gpu_cores=20"
  echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "gpu_temp_entry_c=${entry_c}"
  echo "gpu_temp_exit_c=${exit_c}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "status=${status}"
} > "${out_dir}/meta.txt"

if [[ "${status}" -ne 0 ]]; then
  echo "e109_ab_leg: ${arm_label} FAILED (${status})" >&2
  tail -5 "${out_dir}/stderr.log" >&2
  exit "${status}"
fi

jq -r '"  matched=\(.all_tokens_matched) rounds=\(.round_count)"
  + " spt=\(.parent_measured_seconds_per_token)"
  + " draft=\(.effective_mean_draft_len)"' "${out_dir}/report.json"
