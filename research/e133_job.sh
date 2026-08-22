#!/usr/bin/env bash
# E133: run one command with THIS role's own HOME.
#
# `run_job` starts a supervised process with HOME=/Users/ec2-user, while the
# model head, the hidden-state corpus and the thermal state this campaign uses
# live under the role home. research/e87_job.sh does the same job for
# student-qwen-thorfinn and hard-codes his home, which does not exist on this
# machine, so E133 needs its own wrapper rather than a shared one.
set -uo pipefail
export HOME=/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-askeladd/home
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "$@"
