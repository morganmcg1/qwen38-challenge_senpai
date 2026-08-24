#!/usr/bin/env bash
# E165 Stage 0: close the budget of one decode round and measure GPU idle.
#
#   usage: research/e165_census_session.sh [TOKENS]
#
# THE QUESTION. Is the round idle or slow? A schedule fix can only recover
# time in which the device owns no work. The round's blocking eval is a full
# barrier, so the window from that barrier to the next `asyncEval` is PROVABLY
# idle and everything else has a command buffer enqueued. Measuring the first
# quantity bounds the whole host-reordering family of mechanisms.
#
# WHY PINNED. The adaptive schedule changes the proposed draft count round to
# round, so one leg cannot separate the fixed part of a component from its
# row-scaling part. `MLX_E159_FIXED_DRAFT_DEPTH` pins the proposal, so legs at
# d = 1, 4, 7 give three points of every component and the intercept falls out
# by construction. The pin never exceeds 7: 8 bypasses both `costModelDepth`
# and the shipped width cap.
#
# THE LEGS.
#
#   c1 c4 c7   traced, shipped ladder. These carry the component fit.
#   L4         traced, MLX_QWEN_MTP_LADDER=off. The ladder hides host encode
#              of the 64-layer verify graph inside the eval wait, so under the
#              shipped schedule `verify_build_us` is about 97 % GPU wait and
#              must never be read as host work (E86). With no rungs the two
#              separate: `verify_window` IS the host encode and
#              `gpu_verify_eval` IS the verify GPU wall.
#   c4b        traced, shipped ladder. Brackets L4 so the ladder contrast is
#              read against the mean of two neighbours rather than one, which
#              cancels monotone drift to first order.
#   s4 s7      traced, sync-head. Attribution only. Draining the head chain
#              moves head GPU execute into `submit2`, which is the only way to
#              price the GPU work a cross-round prefetch could move.
#   n4         untraced. The instrumentation-neutrality control: it shares
#              every identity field with c4 except the trace, so the two
#              seconds-per-token values bound what the trace itself costs.
#
# DIAGNOSTIC, NOT A TIMED ARM. No leg here is a candidate-versus-base contrast
# and none of them is an official or ranked score. The legs run UNGATED, as
# the assignment allows for the census, so `cool_gate_passed_real_gate=false`
# and `gate_qualified_for_timing=false` stay in every meta file. Entry and
# exit temperature are recorded per leg and the ladder contrast is bracketed.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e165_census_session: candidate sources are dirty; commit first" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

failures=0
run_leg() {
  local tag="$1" depth="$2" ladder="$3"
  shift 3
  echo "=== ${tag}: pinned d=${depth} ladder=${ladder} tokens=${tokens} $* ==="
  export MLX_E159_FIXED_DRAFT_DEPTH="${depth}"
  if [[ "${ladder}" != "default" ]]; then
    export MLX_QWEN_MTP_LADDER="${ladder}"
  fi
  research/e79_trace_leg.sh "${tag}" "${tokens}" "$@"
  local status=$?
  unset MLX_E159_FIXED_DRAFT_DEPTH
  unset MLX_QWEN_MTP_LADDER
  {
    echo "e165_pinned_depth=${depth}"
    echo "e165_ladder=${ladder}"
    echo "e165_session_commit=${session_commit}"
    echo "e165_session_worker_sha256=${session_worker}"
    echo "e165_stage=0-census"
    echo "e165_leg_role=diagnostic"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e165_census_session: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
}

run_leg e165c1  1 default
run_leg e165c4  4 default
run_leg e165L4  4 off
run_leg e165c4b 4 default
run_leg e165c7  7 default
run_leg e165s4  4 default --sync-head
run_leg e165s7  7 default --sync-head
run_leg e165n4  4 default --no-trace

post_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
if [[ "${post_worker}" != "${session_worker}" ]]; then
  echo "e165_census_session: the worker digest changed during the session" >&2
  failures=$((failures + 1))
fi

echo "e165_census_session: ${failures} failed legs"
python3 research/e165_census.py \
  e165c1 e165c4 e165c4b e165L4 e165c7 e165s4 e165s7 e165n4 \
  --json research/e165-census.json
exit $((failures > 0))
