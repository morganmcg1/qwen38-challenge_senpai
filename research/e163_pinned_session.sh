#!/usr/bin/env bash
# E163 stage 1: price one QMV width plan against the shipped one at a PINNED
# verify width.
#
#   usage: research/e163_pinned_session.sh DEPTH ARM [TOKENS] [LABEL] [--witness-only]
#
# WHY PIN. The natural schedule spends 76.9 % of its rounds at verify width 8
# and only 6.4 % at width 5. An arm that changes width 5 alone is therefore
# diluted about 15x, which puts a real per-width effect under the 0.039 % noise
# floor. `MLX_E159_FIXED_DRAFT_DEPTH=DEPTH` proposes exactly DEPTH drafts every
# round, so every round verifies at width DEPTH+1 and the per-width cost is
# measured directly.
#
# A PINNED DEPTH IS AN INSTRUMENT, NOT A SCHEDULE. It replaces `draftPolicy`
# outright and so bypasses `costModelDepth` AND its `widthCap`. The shipped
# scheduler can never exceed verify width 8, because `segmentedVerifyDepthCap`
# is 7. This script refuses DEPTH > 7 so no leg measures an operating point the
# candidate cannot reach.
#
# ONE BINARY. Both plans are compiled into one worker and the plan is chosen at
# run time, so every leg times the same bytes. `worker_sha256` is recorded
# before and after each leg by `e79_trace_leg.sh` and asserted equal here.
#
# WHAT IS MEASURED. Absolute `mtp_seconds_per_token` is the headline. The local
# serial-to-MTP ratio is reported beside it: the serial leg decodes at M = 1,
# which reaches `default: break` in the width switch and launches no routed
# QMV at all, so the serial leg cannot move and cannot cancel this effect.
#
# ORDER. The timed legs run
#
#   shipped@d   ARM@d   shipped@d+1   shipped@d+1   ARM@d   shipped@d
#
# so all three leg kinds have mean leg position 3.5 and a monotone thermal
# drift in leg index cancels to first order in every pairwise contrast. Every
# timed leg keeps the real 40 C cool gate. The two `d+1` legs measure the width
# boundary `R(W+1) - R(W)` on the shipped plan in the same thermal session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

depth="${1:?usage: e163_pinned_session.sh DEPTH ARM [TOKENS] [LABEL]}"
arm="${2:?usage: e163_pinned_session.sh DEPTH ARM [TOKENS] [LABEL]}"
tokens="${3:-512}"
label="${4:-w$((depth + 1))}"
witness_only=0
[[ "${5:-}" == "--witness-only" ]] && witness_only=1

if ((depth < 0 || depth > 7)); then
  echo "e163_pinned_session: DEPTH must be 0..7; the shipped scheduler caps" \
       "verify width at 8 through segmentedVerifyDepthCap=7, and a leg above" \
       "that measures an unreachable operating point" >&2
  exit 2
fi
if [[ "${arm}" == "shipped" ]]; then
  echo "e163_pinned_session: ARM must be the variant plan, not shipped" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e163_pinned_session: the scored surface is dirty; refusing to time" \
       "over uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
worker=".build-worker/release/mlxfast-runtime-worker"
session_worker="$(shasum -a 256 "${worker}" | awk '{print $1}')"

# The built worker must carry both instruments at all. Each selector is a
# string literal over Swift's 15-byte inline limit, so it reaches the string
# table; a worker built before either arm reports zero.
for needle in MLX_E163_IPG_PLAN MLX_E159_FIXED_DRAFT_DEPTH; do
  copies="$(strings -a "${worker}" | grep -c "${needle}")"
  if ((copies < 1)); then
    echo "e163_pinned_session: the built worker carries no ${needle}, so it" \
         "predates this session; rebuild it" >&2
    exit 3
  fi
  echo "witness: worker carries ${needle} x${copies}"
done
echo "witness: worker sha256 ${session_worker}"

export MLX_E159_FIXED_DRAFT_DEPTH="${depth}"
echo "pin: MLX_E159_FIXED_DRAFT_DEPTH=${depth}, so verify width = $((depth + 1))"

# PHASE 1: prove the plan selector reaches the worker and is read there.
#
# Every arm emits the same tokens and the same counters by construction, so no
# output field can witness which plan ran. The selector itself is witnessed
# instead: an unknown value must stop the worker. A leg that ignored the
# variable would decode normally and exit 0, so this witness can fail.
echo "=== witness: an unknown plan name must stop the worker ==="
if MLX_E163_IPG_PLAN=notaplan research/e79_trace_leg.sh "e163${label}wbad" 8 --no-trace
then
  echo "e163_pinned_session: the worker accepted MLX_E163_IPG_PLAN=notaplan," \
       "so the selector never arrived and no arm is proven" >&2
  exit 4
fi
if ! grep -qs 'MLX_E163_IPG_PLAN must be shipped or one of' \
    "research/out/e163${label}wbad/wrapper.err" \
    "research/out/e163${label}wbad/wrapper.out"; then
  echo "e163_pinned_session: the leg failed without the selector's own" \
       "message; look at research/out/e163${label}wbad before trusting this" \
       "session" >&2
  exit 5
fi
echo "witness ok: the selector reaches the worker and is read there"

# PHASE 2: record the realised width histogram under the pin.
#
# Untimed and ungated on purpose: it exists to show what the pin did, not how
# long it took. `d=` on each trace round is the proposed draft count, so the
# verify width of that round is `d + 1`.
echo "=== witness: realised width histogram under the pin ==="
export MLX_E163_IPG_PLAN="${arm}"
research/e79_trace_leg.sh "e163${label}hist" 128
unset MLX_E163_IPG_PLAN
# Only `mtp-trace: round=` lines are rounds. `mtp-anchor:` repeats the same
# `d=` for its timestamp record, so counting every `d=` doubles every round.
awk '/^mtp-trace: round=/ {
  for (i = 1; i <= NF; i++) if ($i ~ /^d=/) { split($i, kv, "="); w[kv[2] + 1]++; n++ }
} END {
  printf "width histogram over %d traced rounds:\n", n
  for (k in w) printf "  verify width %s  rounds %d  share %.4f\n", k, w[k], w[k] / n
}' "research/out/e163${label}hist/trace.txt"

if ((witness_only)); then
  echo "e163_pinned_session: witness phases only; no timed leg ran"
  exit 0
fi

# PHASE 3: the counterbalanced, gated, timed session.
#
#   1 shipped@d    2 arm@d    3 shipped@d+1    4 shipped@d+1    5 arm@d    6 shipped@d
#
# Every leg kind has mean position 3.5, so a monotone drift in leg index
# cancels to first order in the arm contrast AND in the width contrast.
#
# Legs 3 and 4 cost two of the six legs and buy the width boundary
# `R(W+1) - R(W)` on the SHIPPED plan inside one thermal session. No ranked
# receipt can show it, because no ranked prompt sits between 3.65 and 5.38
# rows, and a second session would carry the ~10 % isolated-comparison drift
# of RULE 119.
#
# Timing legs keep the trace on. Every leg carries the same instrumentation,
# so the contrasts are unaffected, and the trace is the only source of the
# seed prefill, the per-round cost and the realised width of each timed leg.
# `R` cannot be computed without it.
plans=(shipped "${arm}" shipped shipped "${arm}" shipped)
depths=("${depth}" "${depth}" "$((depth + 1))" "$((depth + 1))" "${depth}" "${depth}")
if ((depth + 1 > 7)); then
  echo "e163_pinned_session: the width-boundary legs would pin depth" \
       "$((depth + 1)), above segmentedVerifyDepthCap" >&2
  exit 2
fi

failures=0
for position in 1 2 3 4 5 6; do
  plan="${plans[position - 1]}"
  leg_depth="${depths[position - 1]}"
  tag="e163${label}p${position}${plan}d${leg_depth}"
  echo "=== ${tag}: plan=${plan} pinned_depth=${leg_depth}" \
       "verify_width=$((leg_depth + 1)) tokens=${tokens} ==="
  export MLX_E163_IPG_PLAN="${plan}"
  export MLX_E159_FIXED_DRAFT_DEPTH="${leg_depth}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" --cool-gate
  status=$?
  unset MLX_E163_IPG_PLAN
  {
    echo "e163_plan=${plan}"
    echo "e163_position=${position}"
    echo "e163_pinned_depth=${leg_depth}"
    echo "e163_verify_width=$((leg_depth + 1))"
    echo "e163_session_commit=${session_commit}"
    echo "e163_session_worker_sha256=${session_worker}"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e163_pinned_session: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

echo "e163_pinned_session: ${failures} failed legs"
python3 research/e163_pinned_report.py --label "${label}"
report=$?
exit $(((failures > 0) || report != 0))
