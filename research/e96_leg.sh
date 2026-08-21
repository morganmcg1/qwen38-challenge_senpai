#!/usr/bin/env bash
# One E96 leg: a traced --local-iterate run with the GDN recurrent-step arm
# selected by the environment.
#
#   usage: research/e96_leg.sh TAG TOKENS STEP_MODE [TG_Y] [FORCE_DRAFTS]
#
#   STEP_MODE     vendor | clone | t1 | off  (MLX_E96_STEP)
#   TG_Y          threadgroup y extent for the clone arms (MLX_E96_TG_Y)
#   FORCE_DRAFTS  constant draft width for every round (MLX_E96_FORCE_DRAFTS)
#
# The `off` and `t1` arms change the tokens the model emits. That is the point
# of the ablation, so the leg sets MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1 and the
# reported tokens are NOT verified. Never read such a leg as a fidelity result.
#
# The leg builds nothing. Build and witness the worker before the session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e96_leg.sh TAG TOKENS STEP_MODE [TG_Y] [FORCE_DRAFTS]}"
tokens="${2:?usage: e96_leg.sh TAG TOKENS STEP_MODE [TG_Y] [FORCE_DRAFTS]}"
step="${3:?usage: e96_leg.sh TAG TOKENS STEP_MODE [TG_Y] [FORCE_DRAFTS]}"
tg_y="${4:-}"
force_drafts="${5:-}"

head_dir="${E96_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e96_leg.sh: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export MLXFAST_NO_SANDBOX=1
export MLXFAST_LOCAL_COOL_GATE=0
export MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1

export MLX_E96_STEP="${step}"
[[ -n "${tg_y}" ]] && export MLX_E96_TG_Y="${tg_y}"
[[ -n "${force_drafts}" ]] && export MLX_E96_FORCE_DRAFTS="${force_drafts}"

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
  echo "experiment=e96-gated-deltanet-recurrent-step"
  echo "harness=local"
  echo "tokens=${tokens}"
  echo "local_mode=--local-iterate"
  echo "step_mode=${step}"
  echo "tg_y=${MLX_E96_TG_Y:-4}"
  echo "force_drafts=${MLX_E96_FORCE_DRAFTS:-<schedule>}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "tokens_verified=false"
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

./benchmark-qwen-mtp.sh --local-iterate \
  > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}" || true)"
  echo "post_run_worker_sha256=$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

exit "${status}"
