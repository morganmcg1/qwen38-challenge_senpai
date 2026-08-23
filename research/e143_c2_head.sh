#!/usr/bin/env bash
# E143 C2 rung 0: head provenance and pack size. HOME is set for the same reason as
# research/e142_rung0.sh -- run_job starts with HOME=/Users/ec2-user while the
# capture and the declared head live under the role home.
set -uo pipefail
export HOME=/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-askeladd/home
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 research/e143_c2_head.py "$@"
