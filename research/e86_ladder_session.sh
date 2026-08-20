#!/usr/bin/env bash
# E86: sweep the decode asyncEval ladder rung set, env only, one built worker.
#
#   usage: research/e86_ladder_session.sh PREFIX [--no-sync-head] ARM ...
#
# ARM is `name=ladder`, where `ladder` is the literal value of
# MLX_QWEN_MTP_LADDER for that leg (`default`, `off`, `front`, `dense`, or a
# comma separated rung list). `name` becomes the arm label in the tag, so it
# must not contain `=`. Use `default=default` first so the mirrored order puts
# the session null at both ends of the palindrome.
#
# The ladder is a pure enqueue-timing control at
# Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:2371-2385. It adds no
# op and moves no reduction, so every arm must return the identical round count
# and the identical rows_per_token. A round-count move means the arm is not
# bit-exact and the session must stop.
#
# Every leg loads the DECLARED head, because the ranked candidate leg runs the
# head that mtp-head.manifest.json declares, not the pinned head that
# setup-qwen-mtp.sh provisions.
#
# --sync-head is the default, so the phase split is comparable with the E82
# baseline table: the head chain is drained before the verify window and its
# GPU time lands in draft_build_us instead of hiding inside verify_build_us.
# --no-sync-head measures the PRODUCTION overlap instead, which is the
# configuration a promoted rung set actually runs in.
#
# These legs are UNGATED on purpose (program.md permits an ungated,
# ABBA-counterbalanced local timed arm). e79_trace_leg.sh records entry and
# exit GPU temperature per leg and keeps cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false. The result is directional causal evidence
# inside this session, never a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e86_ladder_session.sh PREFIX [--no-sync-head] ARM ...}"
shift

sync_head=--sync-head
if [[ "${1:-}" == "--no-sync-head" ]]; then
  sync_head=""
  shift
fi

(($#)) || { echo "e86_ladder_session.sh: no arms given" >&2; exit 2; }

declare -a half=("$@")
declare -a order=("${half[@]}")
declare -a rep=()
for _ in "${half[@]}"; do rep+=(1); done
for ((i = ${#half[@]} - 1; i >= 0; i--)); do
  order+=("${half[$i]}")
  rep+=(2)
done

export MLXFAST_QWEN_MTP_HEAD_DIR="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"

for i in "${!order[@]}"; do
  spec="${order[$i]}"
  name="${spec%%=*}"
  ladder="${spec#*=}"
  [[ "${name}" != "${spec}" ]] || {
    echo "e86_ladder_session.sh: arm '${spec}' is not name=ladder" >&2
    exit 2
  }
  MLX_QWEN_MTP_LADDER="${ladder}" \
    research/e79_trace_leg.sh "${prefix}-${name}-${rep[$i]}" 512 ${sync_head}
  status=$?
  echo "leg ${prefix}-${name}-${rep[$i]} ladder=${ladder} exit=${status}"
  ((status == 0)) || exit "${status}"
done
