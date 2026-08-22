#!/usr/bin/env bash
# E136 rung 2: does the C1 sketch shortlist pay for itself end to end?
#
#   usage: research/e136_abba.sh [TOKENS] [LABEL] [ORDER...]
#
#   TOKENS  decode tokens per leg (default 512). The seed is 512 tokens, so a
#           shorter leg never walks the key length past the boundary the
#           shipped warm protects and never sees the round mix a ranked leg
#           sees. 512 is the only headline window.
#   ORDER   leg arms, default `off on on off`, which is one ABBA pair.
#
# ARMS. Both arms are the SAME worker binary under one environment variable.
#   off  MLX_E136_C1_SKETCH=0, the shipped base: affine-2 centroid scan, probe
#        fraction 0.25, affine-2 row scan over 24,584 rows, fused top-32.
#   on   MLX_E136_C1_SKETCH=1, the candidate: rank-256 int8 sketch, probe
#        fraction 0.35, 4,096 survivors, affine-2 rescore of the survivors
#        only, then the same top-32 reduction over the survivors.
#
# Base legs sit at the outer positions and candidate legs at the inner ones,
# so both arms have the same mean position and monotone thermal drift cancels
# to first order. Timing runs with the cool gate OFF, which
# `research/e79_trace_leg.sh` records verbatim as
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false. These
# legs are counterbalanced local evidence, never a gated or ranked score.
#
# WITNESS. `sel_env` in the round trace reads `unset` on a base leg and
# `unset+c1:<draft steps>` on a candidate leg, and `sel_fused` counts the
# shipped row selection. A candidate leg must therefore show a positive C1
# count and `sel_fused=0`, and a base leg the reverse. A leg that fails that
# check timed the wrong mechanism and is discarded, not interpreted.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-r2}"
(($# > 2)) && order=("${@:3}") || order=(off on on off)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e136_abba: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi

# STALENESS GUARD (E136 defect 5, ledger 202(H) again). A clean worktree does
# NOT imply a current worker. `--local-iterate` extracts only
# `metallib_rebuild_required()` from benchmark.sh, never `swift_build_required()`,
# so it will happily time a `.build-worker` binary built before the candidate
# edit. The first rung-2 witness session did exactly that: a worker 1h45m older
# than the C1 commit ran BOTH arms, and the `on` leg reported
# `e136_c1_draft_steps=0 e136_shipped_selection_draft_steps=31`. The trace
# witness caught it, but only after two legs of GPU time. Assert the gate string
# is inside the binary before spending any.
#
# `grep -c`, not `grep -q`. Under `set -o pipefail` a `grep -q` that matches
# closes the pipe, `strings` dies of SIGPIPE with status 141, and pipefail
# reports the pipeline as failed. That inverts the guard: it then rejects
# exactly the fresh worker it is supposed to accept. `grep -c` drains its
# input, so the pipeline status is grep's own.
worker_bin=".build-worker/release/mlxfast-runtime-worker"
gate_copies="$(strings -a "${worker_bin}" | grep -cF -- 'MLX_E136_C1_SKETCH')"
if ((gate_copies == 0)); then
  echo "e136_abba: ${worker_bin} carries no MLX_E136_C1_SKETCH gate." >&2
  echo "e136_abba: run senpai/rebuild-and-assert-worker.sh first." >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
worker_start="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
failures=0
position=0

for arm in "${order[@]}"; do
  position=$((position + 1))
  case "${arm}" in
    on) flag=1; suffix=c ;;
    off) flag=0; suffix=b ;;
    *) echo "e136_abba: unknown arm ${arm}" >&2; exit 2 ;;
  esac
  tag="e136${label}p${position}${suffix}"
  echo "=== ${tag}: arm=${arm} MLX_E136_C1_SKETCH=${flag} tokens=${tokens} ==="

  MLX_E136_C1_SKETCH="${flag}" research/e79_trace_leg.sh "${tag}" "${tokens}"
  status=$?

  trace="research/out/${tag}/trace.txt"
  worker_now="$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
      | awk '{print $1}')"
  c1_drafts="$(
    sed -n 's/.* sel_env=unset+c1:\([0-9]*\) .*/\1/p' "${trace}" 2>/dev/null \
      | tail -1)"
  c1_drafts="${c1_drafts:-0}"
  fused_max="$(
    sed -n 's/.* sel_fused=\([0-9]*\) .*/\1/p' "${trace}" 2>/dev/null | tail -1)"
  fused_max="${fused_max:-0}"
  {
    echo "experiment=e136-c1-sketch-readout-abba"
    echo "e136_arm=${arm}"
    echo "e136_position=${position}"
    echo "MLX_E136_C1_SKETCH=${flag}"
    echo "e136_c1_draft_steps=${c1_drafts}"
    echo "e136_shipped_selection_draft_steps=${fused_max}"
    echo "session_commit=${session_commit}"
    echo "worker_sha256_session_start=${worker_start}"
    echo "worker_sha256_after_leg=${worker_now}"
  } >> "research/out/${tag}/meta.txt"

  if [[ "${worker_now}" != "${worker_start}" ]]; then
    echo "e136_abba: ${tag} ran a worker that moved during the session" >&2
    status=7
  fi
  if [[ "${flag}" == "1" ]]; then
    if ((c1_drafts == 0)) || ((fused_max != 0)); then
      echo "e136_abba: ${tag} asked for C1 but witnessed c1=${c1_drafts}" \
        "fused=${fused_max}" >&2
      status=8
    fi
  else
    if ((c1_drafts != 0)) || ((fused_max == 0)); then
      echo "e136_abba: ${tag} asked for the base but witnessed" \
        "c1=${c1_drafts} fused=${fused_max}" >&2
      status=8
    fi
  fi
  if ((status != 0)); then
    echo "e136_abba: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
  echo "status=${status}" >> "research/out/${tag}/meta.txt"
  # A broken first leg means every later leg is wasted GPU time.
  if ((failures == 1 && position == 1)); then
    echo "e136_abba: first leg failed; aborting the session" >&2
    break
  fi
done

echo "e136_abba: ${failures} failed legs"
exit "${failures}"
