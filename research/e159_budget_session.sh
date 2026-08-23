#!/usr/bin/env bash
# E159 R1 -- measure the decode round's cost law R(D) and its budget.
#
#   usage: research/e159_budget_session.sh timing|trace [TAG]
#
# WHAT IS MEASURED. `MLX_E159_FIXED_DRAFT_DEPTH=D` pins the PROPOSED draft
# count of every round, so a leg is one point of
#
#   R(D) = (s + t0) + (h + t1)*D
#
#     s   MTP session overhead per round (snapshots, rollback, ledger)
#     t0  target cost at batch 1
#     h   proposal head cost per draft step
#     t1  target marginal cost per extra verify row
#
# The adaptive schedule varies D round to round, so it cannot identify either
# term; pinning D can. `adapt` is the unmodified shipped schedule and gives the
# current operating point.
#
# TWO SESSIONS, DIFFERENT PURPOSES.
#
#   timing  Untraced legs. These carry the fit. Nothing instruments the round,
#           so the leg time is the ordinary candidate-leg time.
#   trace   MLX_QWEN_MTP_TRACE=1 plus MLX_QWEN_MTP_TRACE_SYNC_HEAD=1. The sync
#           drains the head chain before the verify graph is built, so the
#           head's own GPU wall lands inside `draft_build_us` instead of
#           overlapping the verify submission. The slope of `draft_build_us` in
#           D is therefore h, measured in situ. These legs are NOT used for the
#           R(D) fit: the forced sync and the trace writes both change the
#           round total.
#
# HEAD. Every leg loads the head that `mtp-head.manifest.json` DECLARES, which
# is the head the ranked candidate leg runs. `setup-qwen-mtp.sh` provisions the
# organizer-pinned head instead, and FINDING 302 measured that head at about
# 2.05x the declared head's price per drafted row -- fitting h against it would
# overstate the drafting share of the ranked round.
#
# THERMAL. Ungated by construction (research/e109_ab_leg.sh keeps no cool
# gate), counterbalanced inside one session by the rotation-plus-mirror
# schedule, entry and exit GPU temperature recorded per leg, and every leg
# keeps cool_gate_passed_real_gate=false and gate_qualified_for_timing=false.
# The effect being measured is tens of percent; the thermal term is tenths.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kind="${1:?usage: e159_budget_session.sh timing|trace [TAG]}"
tag="${2:-e159-${kind}}"

head_dir="${E159_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e159_budget_session: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e159_budget_session: candidate sources are dirty; commit before timing" >&2
  exit 1
fi

case "${kind}" in
  timing)
    blocks="${E159_BLOCKS:-2}"
    specs=(
      "d0=MLX_E159_FIXED_DRAFT_DEPTH=0"
      "d1=MLX_E159_FIXED_DRAFT_DEPTH=1"
      "d2=MLX_E159_FIXED_DRAFT_DEPTH=2"
      "d4=MLX_E159_FIXED_DRAFT_DEPTH=4"
      "d8=MLX_E159_FIXED_DRAFT_DEPTH=8"
      "adapt="
    )
    ;;
  kink)
    # The {0,1,2,4,8} sweep left two gaps that hide where the price actually
    # steps: rows 2-3 cost about 0.6 ms each, rows 4-5 about 8.5 ms each and
    # rows 6-9 about 22 ms each. `sdpaWidthWallDepthCap = 5` predicts one step
    # entering verify width 6, which is D = 5. These four arms bracket it.
    blocks="${E159_BLOCKS:-2}"
    specs=(
      "d3=MLX_E159_FIXED_DRAFT_DEPTH=3"
      "d5=MLX_E159_FIXED_DRAFT_DEPTH=5"
      "d6=MLX_E159_FIXED_DRAFT_DEPTH=6"
      "d7=MLX_E159_FIXED_DRAFT_DEPTH=7"
    )
    ;;
  trace)
    blocks="${E159_BLOCKS:-1}"
    trace_env="MLX_QWEN_MTP_TRACE=1,MLX_QWEN_MTP_TRACE_PATH=@LEG@/trace.txt"
    trace_env="${trace_env},MLX_QWEN_MTP_TRACE_SYNC_HEAD=1"
    specs=(
      "t0=MLX_E159_FIXED_DRAFT_DEPTH=0,${trace_env}"
      "t1=MLX_E159_FIXED_DRAFT_DEPTH=1,${trace_env}"
      "t2=MLX_E159_FIXED_DRAFT_DEPTH=2,${trace_env}"
      "t4=MLX_E159_FIXED_DRAFT_DEPTH=4,${trace_env}"
      "t8=MLX_E159_FIXED_DRAFT_DEPTH=8,${trace_env}"
    )
    ;;
  *)
    echo "e159_budget_session: kind must be timing, kink or trace" >&2
    exit 2
    ;;
esac

E109_BLOCKS="${blocks}" \
E109_TOKENS="${E159_TOKENS:-512}" \
E109_DEPTH=8 \
E109_HEAD_DIR="${head_dir}" \
E109_GOLDEN="${E159_GOLDEN:-research/out/e159-golden-512.json}" \
  research/e109_ab_session.sh "${tag}" "${specs[@]}"
