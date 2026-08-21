#!/usr/bin/env bash
# E99 rung 1: prove the round recorder is schedule-neutral, and prove the
# recorded E94 rung-1 traces are still valid on this base.
#
#   usage: research/e99_rung1.sh [TOKENS]
#
# Two claims, one session.
#
# 1. INSTRUMENTATION NEUTRALITY. `MLX_QWEN_MTP_TRACE=1` only formats strings;
#    it reads no schedule state and writes none. The legs below run the same
#    offered cap with the flag on and off, so `effective_mean_draft_len` and
#    `accepted_draft_rate` must be digit-identical across all four cap-5 legs.
#
# 2. TRACE REUSE. The E94 rung-1 cap sweep was recorded at base c4e849a8.
#    This base adds the E95 q-live-rows rider in `Qwen35.swift`, which claims
#    bit-exactness. If it is bit-exact then the round sequence is unchanged, so
#    the cap-5 and cap-7 traces here must match `e94c5a` and `e94c7a` round for
#    round in (d, acc). That check is what licenses the offline analysis to
#    reuse the whole recorded cap sweep instead of spending five more legs.
#
# Legs run traced/untraced/untraced/traced so monotone thermal drift cancels
# to first order across the traced-untraced contrast, even though the claim
# under test is a digit identity rather than a timing difference.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
failures=0

run_leg() {
  local tag="$1" cap="$2"
  shift 2
  echo "=== ${tag}: offered cap=${cap} tokens=${tokens} $* ==="
  MLXFAST_QWEN_MTP_DEPTH="${cap}" \
    research/e79_trace_leg.sh "${tag}" "${tokens}" "$@"
  local status=$?
  {
    echo "e99_cap=${cap}"
    echo "e99_arm=ship"
    echo "experiment=e99-rung1"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e99_rung1: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
}

run_leg e99r1c5t1 5
run_leg e99r1c5u1 5 --no-trace
run_leg e99r1c5u2 5 --no-trace
run_leg e99r1c5t2 5
run_leg e99r1c7t1 7

echo "e99_rung1: ${failures} failed legs"
exit "${failures}"
