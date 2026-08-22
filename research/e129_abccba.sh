#!/usr/bin/env bash
# E129: the three dispatch tables timed against each other inside one session.
#
#   usage: research/e129_abccba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# A = shipped     two verify passes at M=6, M=7 and M=8
# B = onepass67   one pass at M=6 and M=7   (the submitted arm)
# C = onepass678  one pass at M=6, M=7 and M=8
#
# Order inside a replicate is A B C C B A, so every arm has mean position 3.5
# and monotone thermal drift cancels to first order. This is the counterbalance
# `program.md` requires before `MLXFAST_LOCAL_COOL_GATE=0` is a permitted mode;
# entry and exit temperature are recorded per leg and the legs keep
# `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`.
#
# NO REBUILD BETWEEN LEGS. All four tables are compiled into one worker and the
# arm is chosen at run time by `MLX_E120_QMV_TABLE`. Every leg therefore times
# the same bytes, and the whole prep-matching apparatus the E121 design needed
# is not needed here: `worker_sha256` is asserted equal across the session
# instead of being reproduced.
#
# WHAT IS MEASURED. `mtp_seconds_per_token`, absolute. The local ratio is not
# usable for this change: the arm sits in the QMV verify kernel, which both
# local legs run, so a real improvement partly cancels in the ratio. The ratio
# is still recorded, and a reader can see the cancellation happen.
#
# FIRST numbers the first replicate, so a later session extends this estimate
# instead of restarting it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
tokens="${2:-512}"
label="${3:-s1}"
first="${4:-1}"

readonly PLAN_SHIPPED="e120_width_plan/3:3:4,4:4:4,5:5:4,6:3:4,7:4:4,8:4:4,9:3:4"
readonly PLAN_ONEPASS67="e120_width_plan/3:3:4,4:4:4,5:5:4,6:6:4,7:7:4,8:4:4,9:3:4"
readonly PLAN_ONEPASS678="e120_width_plan/3:3:4,4:4:4,5:5:4,6:6:4,7:7:4,8:8:4,9:3:4"

plan_for() {
  case "$1" in
    shipped)    echo "${PLAN_SHIPPED}" ;;
    onepass67)  echo "${PLAN_ONEPASS67}" ;;
    onepass678) echo "${PLAN_ONEPASS678}" ;;
    *) echo "e129_abccba: unknown arm $1" >&2; return 2 ;;
  esac
}

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e129_abccba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

# PHASE 1: prove the selector reaches the worker.
#
# `Table(rawValue:)` falls back to the compiled default on any unrecognized
# string, which is the right runtime behaviour and the wrong test behaviour: a
# selector that never arrives is indistinguishable from one that arrives and is
# honoured, because both run the default. So each witness leg asserts the plan
# the pipeline log recorded, and the two non-default arms are the ones that can
# fail. A stripped environment shows up as `onepass67` under a `shipped`
# request.
witness_tokens="${E129_WITNESS_TOKENS:-16}"
for arm in shipped onepass678 onepass67; do
  tag="e129${label}w${arm}"
  want="$(plan_for "${arm}")" || exit 2
  echo "=== witness ${tag}: arm=${arm} tokens=${witness_tokens} ==="
  out="research/out/${tag}"
  export MLX_E120_QMV_TABLE="${arm}"
  export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${out}/pipelines.json"
  mkdir -p "${out}"
  research/e79_trace_leg.sh "${tag}" "${witness_tokens}" --no-trace
  status=$?
  unset MLX_E120_QMV_PIPELINE_LOG
  got="$(python3 -c "
import json,sys
try:
    print(json.load(open(sys.argv[1]))['plan'])
except Exception as exc:
    print('unreadable: %s' % exc)
" "${out}/pipelines.json" 2>/dev/null)"
  echo "e129_arm=${arm}" >> "${out}/meta.txt"
  echo "e129_plan_want=${want}" >> "${out}/meta.txt"
  echo "e129_plan_got=${got}" >> "${out}/meta.txt"
  if [[ "${got}" != "${want}" ]]; then
    echo "e129_abccba: ${arm} selected ${got:-<nothing>}, wanted ${want}" >&2
    echo "e129_abccba: the selector does not reach the worker; not timing" >&2
    exit 3
  fi
  echo "witness ${arm}: plan ok (leg exit ${status})"
done
unset MLX_E120_QMV_TABLE

# PHASE 2: the counterbalanced timed session.
failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in shipped onepass67 onepass678 onepass678 onepass67 shipped; do
    position=$((position + 1))
    tag="e129${label}k${rep}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
    export MLX_E120_QMV_TABLE="${arm}"
    research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace
    status=$?
    {
      echo "e129_arm=${arm}"
      echo "e129_replicate=${rep}"
      echo "e129_position=${position}"
      echo "e129_session_commit=${session_commit}"
      echo "e129_session_worker_sha256=${session_worker}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e129_abccba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done
unset MLX_E120_QMV_TABLE

echo "e129_abccba: ${failures} failed legs"
python3 research/e129_abccba_report.py --label "${label}"
exit $((failures > 0))
