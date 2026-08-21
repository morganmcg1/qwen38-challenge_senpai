#!/usr/bin/env bash
# E87: run one command with the role's own HOME.
#
# `run_job` starts a supervised process with HOME=/Users/ec2-user, while the
# interactive shell and every cached artifact this campaign uses live under the
# role home. Without this wrapper a supervised leg would look for the model
# head, the corpus and the thermal state in the wrong tree.
set -uo pipefail
export HOME=/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-thorfinn/home
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "$@"
