#!/usr/bin/env bash
# E99 rungs 6 and 7: the session null, the clamp depth, and the width-
# dependent threshold.
#
#   usage: research/e99_rung67.sh debug   TOKENS
#          research/e99_rung67.sh session TOKENS
#
# WHY THIS IS NOT TWELVE REPLICATES OF THREE THRESHOLDS.
#
# The recorded round sequence is bit-identical across replicates of the same
# arm. Six arm-and-cap pairs were checked and all six match on the full
# (round, depth, accepted) sequence. The ranked-curve figure is a pure
# function of that sequence, so it carries ZERO run-to-run noise and
# replicating a threshold point cannot sharpen it. Only the wall clock is
# stochastic. Replication therefore goes where the noise is, and the freed
# legs go into resolution instead.
#
#   block N, 8 legs   the wall-clock null, and the proof that this build
#                     reproduces the submitted build's shipped sequence.
#                     Order O S S O O S S O, so each arm holds positions
#                     summing to 18 of 1..8 and a linear drift cancels
#                     exactly.
#   block D, 3 legs   rung 7. Clamp depth 1, 2 and 4 at the shipped
#                     threshold. Depth 3 is block N's shipped arm.
#   block W, 5 legs   the width-dependent threshold. Cap 8 peaks at 8.25 to
#                     9.4375 and cap 5 peaks at or above 11.5625, and both
#                     figures are exact, so the optimum moves with width.
#                     These legs place the cap-8 cliff and the cap-5 peak.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:?usage: e99_rung67.sh debug|session TOKENS}"
tokens="${2:?missing TOKENS}"
failures=0

# tag gate threshold depth cap
run_leg() {
  local tag="$1" gate="$2" threshold="$3" depth="$4" cap="$5"
  echo "=== ${tag}: gate=${gate} t=${threshold} d=${depth} cap=${cap}" \
       "tokens=${tokens} ==="
  rm -rf "research/out/${tag}"
  local status
  if [[ "${gate}" == "default" ]]; then
    env -u MLX_QWEN_MTP_MARGIN_GATE -u MLX_QWEN_MTP_MARGIN_GATE_T \
        -u MLX_QWEN_MTP_MARGIN_GATE_D \
      MLXFAST_QWEN_MTP_DEPTH="${cap}" \
      research/e79_trace_leg.sh "${tag}" "${tokens}"
    status=$?
  else
    MLX_QWEN_MTP_MARGIN_GATE="${gate}" \
    MLX_QWEN_MTP_MARGIN_GATE_T="${threshold}" \
    MLX_QWEN_MTP_MARGIN_GATE_D="${depth}" \
    MLXFAST_QWEN_MTP_DEPTH="${cap}" \
      research/e79_trace_leg.sh "${tag}" "${tokens}"
    status=$?
  fi
  local trace="research/out/${tag}/trace.txt"
  {
    echo "experiment=e99-rung67"
    echo "e99_cap=${cap}"
    echo "e99_gate=${gate}"
    echo "e99_gate_threshold=${threshold}"
    echo "e99_gate_depth=${depth}"
    echo "e99_fired_rounds=$(grep -c ' fire=1 ' "${trace}" 2>/dev/null || true)"
    echo "e99_traced_gd=$(
      grep -o ' gd=[0-9]*' "${trace}" 2>/dev/null | sort -u | tr -d ' ' \
        | paste -sd, -)"
    echo "e99_traced_gt=$(
      grep -o ' gt=[0-9.]*' "${trace}" 2>/dev/null | sort -u | tr -d ' ' \
        | paste -sd, -)"
    echo "e99_traced_gate=$(
      grep -o 'gate=[a-z0-9]*' "${trace}" 2>/dev/null | sort -u | paste -sd, -)"
    echo "e99_round_sequence_sha256=$(
      grep -o 'round=[0-9]* d=[0-9]* acc=[0-9]*' "${trace}" 2>/dev/null \
        | shasum -a 256 | awk '{print $1}')"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e99_rung67: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
}

case "${mode}" in
  debug)
    run_leg e99r67dbgE g1 9.4375 2 8
    run_leg e99r67dbgF g1 9.4375 4 8
    ;;
  session)
    # Block N: wall-clock null and cross-build reproduction.
    run_leg e99r6n1o off     9.4375 3 8
    run_leg e99r6n2s default 9.4375 3 8
    run_leg e99r6n3s default 9.4375 3 8
    run_leg e99r6n4o off     9.4375 3 8
    run_leg e99r6n5o off     9.4375 3 8
    run_leg e99r6n6s default 9.4375 3 8
    run_leg e99r6n7s default 9.4375 3 8
    run_leg e99r6n8o off     9.4375 3 8

    # Block D: rung 7, the clamp depth at the shipped threshold.
    run_leg e99r7d1 g1 9.4375 1 8
    run_leg e99r7d2 g1 9.4375 2 8
    run_leg e99r7d4 g1 9.4375 4 8

    # Block W: the width-dependent threshold.
    run_leg e99r6w8a g1  12.5000 3 8
    run_leg e99r6w8b g1  14.0000 3 8
    run_leg e99r6w5o off  9.4375 3 5
    run_leg e99r6w5a g1  12.5000 3 5
    run_leg e99r6w5b g1  14.0000 3 5
    ;;
  *)
    echo "e99_rung67: unknown mode ${mode}" >&2
    exit 2
    ;;
esac

echo "e99_rung67: ${failures} failed legs"
exit "${failures}"
