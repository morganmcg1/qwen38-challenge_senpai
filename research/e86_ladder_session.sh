#!/usr/bin/env bash
# E86: sweep the decode asyncEval ladder rung set, env only, one built worker.
#
#   usage: research/e86_ladder_session.sh PREFIX [--no-sync-head] [--dry-run] ARM ...
#
# ARM is `name=ladder`, where `ladder` is the literal value of
# MLX_QWEN_MTP_LADDER for that leg (`default`, `off`, `front`, `dense`, or a
# comma separated rung list). `name` becomes the arm label in the tag, so it
# must not contain `=`. A repeated `name` gets the next repeat index, so one
# arm may occupy several positions in the half.
#
# POSITION IS A CONFOUND. E86 rung 0 and rung 1 both put the reference arm
# first, so the mirror also put it last. Host phases OUTSIDE the verify window
# cost about 760-820 us/round in every interior leg, but 1372-3607 us/round in
# the four legs that held position 0 or the last position. That inflated every
# arm-versus-reference difference by several hundred us/round.
#
# Give position 0 and the last position to a throwaway `warm` arm, and repeat
# each compared arm so its MEAN position equals the mean position of the
# reference. A palindrome alone does not do this.
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
dry_run=0
while :; do
  case "${1:-}" in
    --no-sync-head) sync_head=""; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) break ;;
  esac
done

(($#)) || { echo "e86_ladder_session.sh: no arms given" >&2; exit 2; }

declare -a half=("$@")
declare -a order=("${half[@]}")
for ((i = ${#half[@]} - 1; i >= 0; i--)); do order+=("${half[$i]}"); done

# macOS ships bash 3.2, which has no associative arrays. Count earlier
# occurrences of the arm name to get its repeat index.
declare -a tags=()
for i in "${!order[@]}"; do
  spec="${order[$i]}"
  name="${spec%%=*}"
  [[ "${name}" != "${spec}" ]] || {
    echo "e86_ladder_session.sh: arm '${spec}' is not name=ladder" >&2
    exit 2
  }
  n=1
  for ((j = 0; j < i; j++)); do
    [[ "${order[$j]%%=*}" == "${name}" ]] && n=$((n + 1))
  done
  tags+=("${prefix}-${name}-${n}")
done

echo "plan: ${#order[@]} legs, sync_head='${sync_head:-off}'"
for i in "${!order[@]}"; do
  echo "  pos=${i} tag=${tags[$i]} ladder=${order[$i]#*=}"
done
((dry_run)) && exit 0

export MLXFAST_QWEN_MTP_HEAD_DIR="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"

for i in "${!order[@]}"; do
  MLX_QWEN_MTP_LADDER="${order[$i]#*=}" \
    research/e79_trace_leg.sh "${tags[$i]}" 512 ${sync_head}
  status=$?
  echo "leg ${tags[$i]} pos=${i} ladder=${order[$i]#*=} exit=${status}"
  ((status == 0)) || exit "${status}"
done

echo "session ${prefix}: ${#order[@]} legs complete"
