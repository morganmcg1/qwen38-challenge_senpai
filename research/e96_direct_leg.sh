#!/usr/bin/env bash
# One E96 leg that runs the trusted CLI directly, without the wrapper's public
# drift tripwire.
#
#   usage: research/e96_direct_leg.sh TAG TOKENS STEP_MODE [TG_Y] [FORCE_DRAFTS]
#
# WHY THIS EXISTS. `benchmark-qwen-mtp.sh` step 1 runs `mlxfast-swift
# correctness` against the public M5 golden and exits 1 before it measures
# anything. An ablation arm emits different tokens by construction, so that gate
# fails and no timing is ever produced (measured: `e96smoke-off-y4`,
# `e96smoke-t1-y4`). This script therefore reproduces the wrapper's steps 2 and
# 4 with the same trusted binary and the same arguments, and skips step 1.
#
#   2. reference rows   mlxfast-swift mtp-verify --generate (this arm's build)
#   4. native-MTP decode mlxfast-swift mtp-timed --mtp-depth 8
#
# Every arm, including the unablated controls, runs this identical path, so the
# comparison across arms stays matched. A leg here is NEVER a fidelity result
# and never a score: it carries no public gate, no cool gate, and an ablation
# arm's tokens are not verified.
#
# The serial control leg is skipped: the local ratio cancels a change that
# moves both legs, and the quantity this experiment needs is the absolute
# candidate round cost. Pass WITH_SERIAL=1 to add it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e96_direct_leg.sh TAG TOKENS STEP_MODE [TG_Y] [FORCE_DRAFTS]}"
tokens="${2:?usage: e96_direct_leg.sh TAG TOKENS STEP_MODE [TG_Y] [FORCE_DRAFTS]}"
step="${3:?usage: e96_direct_leg.sh TAG TOKENS STEP_MODE [TG_Y] [FORCE_DRAFTS]}"
tg_y="${4:-}"
force_drafts="${5:-}"
with_serial="${WITH_SERIAL:-0}"

swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
public_golden_path="correctness_prompts/public_longcopy_gate_english_512_256.json"
depth="${MLXFAST_QWEN_MTP_DEPTH:-8}"

head_dir="${E96_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e96_direct_leg.sh: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

# One model-holding process at a time. The wrapper owns the shared lock; this
# script does not take it, so it refuses to start beside a live worker instead.
if pgrep -f "mlxfast-runtime-worker" >/dev/null 2>&1; then
  echo "e96_direct_leg.sh: a runtime worker is already resident:" >&2
  pgrep -lf "mlxfast-runtime-worker" >&2
  exit 1
fi

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_NO_SANDBOX=1
export MLX_E96_STEP="${step}"
[[ -n "${tg_y}" ]] && export MLX_E96_TG_Y="${tg_y}"
[[ -n "${force_drafts}" ]] && export MLX_E96_FORCE_DRAFTS="${force_drafts}"

# The sink is opened O_APPEND so every worker sharing one path writes into the
# same file. The reference pass decodes at the same depth as the timed pass, so
# one shared path would mix reference rounds into the measured record. Each
# phase therefore gets its own file.
trace_path="${PWD}/${out}/trace.txt"
verify_trace_path="${PWD}/${out}/trace-verify.txt"
: > "${trace_path}"
: > "${verify_trace_path}"
export MLX_QWEN_MTP_TRACE=1

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
  echo "experiment=e96-gated-deltanet-recurrent-step"
  echo "harness=local"
  echo "leg_path=direct-cli-no-tripwire"
  echo "tokens=${tokens}"
  echo "local_mode=direct-mtp-timed"
  echo "step_mode=${step}"
  echo "tg_y=${MLX_E96_TG_Y:-4}"
  echo "force_drafts=${MLX_E96_FORCE_DRAFTS:-<schedule>}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "public_drift_tripwire_run=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
  echo "worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
  "${public_golden_path}" > "${out}/seed-plan.json"

MLX_QWEN_MTP_TRACE_PATH="${verify_trace_path}" \
"${swift_bin}" mtp-verify \
  --weights "${weights_path}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
  --emitted "${out}/seed-plan.json" \
  --generate "$(( tokens + 1 ))" \
  --mtp-depth "${depth}" \
  --output "${out}/golden-rows.json" \
  --plan-output "${out}/generated-plan.json" \
  > "${out}/verify.out" 2> "${out}/verify.err"
generate_status=$?
echo "gpu_temp_after_reference_c=$(gpu_temp)" >> "${out}/meta.txt"
echo "generate_exit=${generate_status}" >> "${out}/meta.txt"

serial_status="skipped"
if [[ "${with_serial}" == "1" && "${generate_status}" == "0" ]]; then
  MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/trace-serial.txt" \
  "${swift_bin}" mtp-timed \
    --weights "${weights_path}" \
    --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
    --golden "${out}/golden-rows.json" \
    --tokens "${tokens}" \
    --mtp-depth 0 \
    > "${out}/serial-control.json" 2> "${out}/serial.err"
  serial_status=$?
fi
echo "serial_exit=${serial_status}" >> "${out}/meta.txt"

timed_status=1
if [[ "${generate_status}" == "0" ]]; then
  # The trace file is written round by round DURING the decode, so a leg whose
  # post-window reference audit throws still leaves a complete round record.
  MLX_QWEN_MTP_TRACE_PATH="${trace_path}" \
  "${swift_bin}" mtp-timed \
    --weights "${weights_path}" \
    --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
    --golden "${out}/golden-rows.json" \
    --tokens "${tokens}" \
    --mtp-depth "${depth}" \
    > "${out}/mtp-decode.json" 2> "${out}/mtp.err"
  timed_status=$?
fi

{
  echo "timed_exit=${timed_status}"
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}" || true)"
  echo "post_run_worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${generate_status}"
