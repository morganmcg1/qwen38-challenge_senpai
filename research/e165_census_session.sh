#!/usr/bin/env bash
# E165 Stage 0: census of the per-round fixed cost at three pinned depths.
#
#   usage: research/e165_census_session.sh [TOKENS]
#
# WHY PINNED. The adaptive schedule changes the proposed draft count round to
# round, so one leg cannot separate the fixed part of a component from its
# row-scaling part. `MLX_E159_FIXED_DRAFT_DEPTH` pins the proposal, so three
# legs at d = 1, 4, 7 give three points of every component and the intercept
# falls out by construction. The pin never exceeds 7: 8 bypasses both
# `costModelDepth` and the shipped width cap.
#
# THE LEGS.
#
#   c1 c4 c7   traced, GATED. These carry the component fit.
#   n4         untraced, GATED. The instrumentation-neutrality control: it
#              shares every identity field with c4 except the trace, so the
#              two seconds-per-token values bound what the trace itself costs.
#   s1 s7      traced, sync-head, ungated. Attribution only. Draining the head
#              chain moves head GPU execute into `submit2`, which is the only
#              way to price the GPU work that could be moved into the idle
#              window.
#
# DIAGNOSTIC, NOT A TIMED ARM. No leg here is a candidate-versus-base contrast
# and none of them is an official or ranked score. The gated legs still pass
# the real 40 C gate, so their absolute component values are comparable with
# each other; the sync-head legs are labelled ungated and perturbing.
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
  local tag="$1" depth="$2"
  shift 2
  echo "=== ${tag}: pinned d=${depth} tokens=${tokens} $* ==="
  export MLX_E159_FIXED_DRAFT_DEPTH="${depth}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" "$@"
  local status=$?
  unset MLX_E159_FIXED_DRAFT_DEPTH
  {
    echo "e165_pinned_depth=${depth}"
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

run_leg e165c1 1 --cool-gate
run_leg e165c4 4 --cool-gate
run_leg e165c7 7 --cool-gate
run_leg e165n4 4 --cool-gate --no-trace
run_leg e165s1 1 --sync-head
run_leg e165s7 7 --sync-head

post_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
if [[ "${post_worker}" != "${session_worker}" ]]; then
  echo "e165_census_session: the worker digest changed during the session" >&2
  failures=$((failures + 1))
fi

echo "e165_census_session: ${failures} failed legs"
python3 research/e165_census.py e165c1 e165c4 e165c7 e165s1 e165s7 \
  --json research/e165-census.json
exit $((failures > 0))
