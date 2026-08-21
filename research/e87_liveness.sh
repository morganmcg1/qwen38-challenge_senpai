#!/usr/bin/env bash
# E87 arm-C liveness positive control.
#
# Builds a damaged clone of the arm-C head whose `draft_cluster.perm` is
# reversed, then runs one short leg with it. A live cluster path must collapse
# the accepted-draft rate; a dead one (silent fall-back to the dense readout)
# must leave it at the declared value.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

built="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/e87/built"
/opt/homebrew/bin/python3 research/e87_damage_head.py \
  "${built}/e87-armC-plain-k12292-p25-run" "${built}/e87-armC-damaged-run" || exit $?

E87_TOKENS="${E87_TOKENS:-64}" research/e87_timing_session.sh e87live armc-damaged
