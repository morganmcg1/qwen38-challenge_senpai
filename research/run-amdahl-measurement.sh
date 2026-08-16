#!/usr/bin/env bash
# Research-only driver: run one gated local Qwen-MTP benchmark, keep both timed
# reports, and decompose them into the charged seed prefill and decode work.
#
#   research/run-amdahl-measurement.sh TAG [MODE] [TOKENS] [BASE_SHA] [NOTES]
#
# Every gate benchmark-qwen-mtp.sh owns (drift tripwire, orphan scan, run lock,
# 40C cool gate, report seals) still runs unmodified; MLXFAST_SWIFT_BIN only
# points at a passthrough that keeps a copy of each stdout report.
set -euo pipefail

tag="${1:?usage: run-amdahl-measurement.sh TAG [MODE] [TOKENS] [BASE_SHA] [NOTES]}"
mode="${2:---local-iterate}"
tokens="${3:-}"
base_sha="${4:-}"
notes="${5:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

out_dir="${repo_root}/.mlxfast-private/amdahl/${tag}"
rm -rf -- "${out_dir}"
mkdir -p "${out_dir}"

export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
export MLXFAST_CAPTURE_DIR="${out_dir}/reports"
export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
export MLXFAST_SCORE_PATH="${out_dir}/score.json"
if [[ -n "${tokens}" ]]; then
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${tokens}"
fi

{
  echo "run-amdahl-measurement: tag=${tag} mode=${mode} tokens=${tokens:-default}"
  echo "run-amdahl-measurement: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  date -u '+run-amdahl-measurement: started_utc=%Y-%m-%dT%H:%M:%SZ'
} >&2

./benchmark-qwen-mtp.sh "${mode}"

date -u '+run-amdahl-measurement: benchmark_finished_utc=%Y-%m-%dT%H:%M:%SZ' >&2

python_bin="${MLXFAST_PYTHON_BIN:-/Users/ec2-user/.senpai/aws-mac-runners/qwen38-mlx-senpai-r1/venv/bin/python3}"
[[ -x "${python_bin}" ]] || python_bin="$(command -v python3)"

wandb_flag=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb_flag=(--wandb)
else
  echo "run-amdahl-measurement: WANDB_API_KEY unset; skipping W&B logging" >&2
fi

WANDB_PROJECT="${WANDB_PROJECT:-qwen38-mlx-challenge-senpai}" \
WANDB_ENTITY="${WANDB_ENTITY:-wandb-applied-ai-team}" \
"${python_bin}" research/prefill_amdahl.py "${MLXFAST_CAPTURE_DIR}" \
  --tag "${tag}" \
  --mode "${mode}" \
  --base-sha "${base_sha}" \
  --head-sha "$(git rev-parse HEAD)" \
  --score-json "${MLXFAST_SCORE_PATH}" \
  --notes "${notes}" \
  "${wandb_flag[@]}" \
  | tee "${out_dir}/amdahl.json"
