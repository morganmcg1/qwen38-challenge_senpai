#!/usr/bin/env bash
# E143 R0 stage B: the target's exact logit and rank of the proposed token at
# every first-divergence row. One streamed pass over the affine-4 lm_head.
#
# HOME is set here because `run_job` starts a supervised process with
# HOME=/Users/ec2-user, while the E142 capture this reads lives under the role
# home. Same reason as research/e142_rung0.sh.
set -uo pipefail
export HOME=/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-askeladd/home
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 research/e143_r0.py --stage b "$@"
