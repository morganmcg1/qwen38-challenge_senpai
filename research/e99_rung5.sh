#!/usr/bin/env bash
# E99 rung 5: measure the conditional one-bit G clamp against the shipped walk.
#
#   usage: research/e99_rung5.sh abba   TOKENS CAP THRESHOLD
#          research/e99_rung5.sh sweep  TOKENS CAP THRESHOLD [THRESHOLD ...]
#          research/e99_rung5.sh debug  TOKENS CAP THRESHOLD
#
# `abba` runs off / on / on / off in one session, so monotone thermal drift
# cancels to first order across the arm contrast. `sweep` runs one leg per
# threshold, for the measured part of the threshold-sensitivity curve. `debug`
# runs a single short arm-on leg to prove the arm is live before an expensive
# session.
#
# Both arms run the SAME binary and differ only in `MLX_QWEN_MTP_MARGIN_GATE`,
# so no rebuild sits between them. Every leg is traced, which costs both arms
# the same recorder time (E99 rung 1 measured +0.571 % with a cold-start
# confound, ABBA counterbalanced).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:?usage: e99_rung5.sh abba|sweep|debug TOKENS CAP THRESHOLD...}"
tokens="${2:?missing TOKENS}"
cap="${3:?missing CAP}"
shift 3
failures=0

run_leg() {
  local tag="$1" gate="$2" threshold="$3"
  echo "=== ${tag}: gate=${gate} t=${threshold} cap=${cap} tokens=${tokens} ==="
  rm -rf "research/out/${tag}"
  local status
  if [[ "${gate}" == "default" ]]; then
    # The shipped path. This leg exports no gate variable at all, so its trace
    # witnesses what the ranked worker would run.
    env -u MLX_QWEN_MTP_MARGIN_GATE -u MLX_QWEN_MTP_MARGIN_GATE_T \
      MLXFAST_QWEN_MTP_DEPTH="${cap}" \
      research/e79_trace_leg.sh "${tag}" "${tokens}"
    status=$?
  else
    MLX_QWEN_MTP_MARGIN_GATE="${gate}" \
      MLX_QWEN_MTP_MARGIN_GATE_T="${threshold}" \
      MLXFAST_QWEN_MTP_DEPTH="${cap}" \
      research/e79_trace_leg.sh "${tag}" "${tokens}"
    status=$?
  fi
  {
    echo "experiment=e99-rung5"
    echo "e99_cap=${cap}"
    echo "e99_gate=${gate}"
    echo "e99_gate_threshold=${threshold}"
    echo "e99_fired_rounds=$(
      grep -c ' fire=1 ' "research/out/${tag}/trace.txt" 2>/dev/null || true)"
  } >> "research/out/${tag}/meta.txt"
  if ((status != 0)); then
    echo "e99_rung5: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
}

case "${mode}" in
  abba)
    # The B arm exports nothing, so it measures the shipped default path and
    # its threshold is the compiled constant. Pass that same value so the
    # recorded threshold and the traced `gt=` witness agree.
    threshold="${1:?missing THRESHOLD}"
    run_leg "e99r5c${cap}a1" off "${threshold}"
    run_leg "e99r5c${cap}b1" default "${threshold}"
    run_leg "e99r5c${cap}b2" default "${threshold}"
    run_leg "e99r5c${cap}a2" off "${threshold}"
    ;;
  sweep)
    (($#)) || { echo "e99_rung5: sweep needs thresholds" >&2; exit 2; }
    index=0
    for threshold in "$@"; do
      index=$((index + 1))
      run_leg "e99r5c${cap}t${index}" g1 "${threshold}"
    done
    ;;
  debug)
    run_leg "e99r5c${cap}dbg" g1 "${1:?missing THRESHOLD}"
    ;;
  *)
    echo "e99_rung5: unknown mode ${mode}" >&2
    exit 2
    ;;
esac

echo "e99_rung5: ${failures} failed legs"
exit "${failures}"
