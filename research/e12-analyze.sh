#!/bin/bash
# e12 seed-prefill charge: analysis entry point (argv only, no env plumbing).
#
#   research/e12-analyze.sh TAG [SHORT_WINDOW_TAG]
#
# Reads the per-arm reports research/e12-run.sh captured for TAG and logs the
# seed-prefill decomposition to W&B. SHORT_WINDOW_TAG adds a second capture at a
# different decode window so the fixed/variable solve can cross-check the
# directly measured charge. No GPU work happens here.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

tag="${1:?tag}"
short_tag="${2:-}"

export WANDB_RUN_GROUP="qwen38-r1-e12-seed-prefill-charge"
export WANDB_PROJECT="qwen38-mlx-challenge-senpai"
export WANDB_ENTITY="wandb-applied-ai-team"

args=(
  "research/capture-e12-${tag}"
  --tag "e12-${tag}"
  --base-sha fe38ecc21e4084e4d17dac3aa76264bb5897a614
  --head-sha "$(git rev-parse HEAD)"
  --score-json "research/score-e12-${tag}.json"
  --wandb
  --notes "e12 seed-prefill charge, matched --local-iterate pair, tag=${tag}"
)
[[ -n "${short_tag}" ]] && args+=(--short-window-capture-dir "research/capture-e12-${short_tag}")

out="research/analysis-e12-${tag}.json"
python3 research/prefill_amdahl.py "${args[@]}" | tee "${out}"
status="${PIPESTATUS[0]}"
echo "e12: analysis_exit=${status} out=${out}"
exit "${status}"
