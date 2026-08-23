#!/usr/bin/env bash
# E135: the restored E87 probe select timed against the incumbent chain.
#
#   usage: research/e135_e87_select_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# S = select      one `qwen_mtp_e87_probe_select` dispatch.
# I = incumbent   `MLX.argPartition` plus the probe-sort compaction, which is
#                 what our base ships and what the promoted frontier replaced.
#
# Order inside a replicate is S I I S, so both arms have mean position 2.5 and
# a monotone drift in leg index cancels to first order in the contrast. This is
# the counterbalance `program.md` requires before `MLXFAST_LOCAL_COOL_GATE=0` is
# a permitted mode; entry and exit temperature are recorded per leg and the legs
# keep `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`.
#
# NO REBUILD BETWEEN LEGS. Both arms are compiled into one worker and chosen at
# run time by `MLX_E87_SELECT`, so every leg times the same bytes and
# `worker_sha256` is asserted equal across the session.
#
# WHAT IS MEASURED. `mtp_seconds_per_token`, absolute, is the headline. The
# local ratio is a valid second instrument for THIS change only: the selection
# runs inside `clusterCandidateIDs`, which the serial leg never enters, so the
# serial leg is untouched and cannot cancel the effect.
#
# THE FREE FALSIFIER. The kernel is verified exact against
# `sorted(argPartition(score))` at every probe rung, so the two arms must emit
# the same tokens. `effective_mean_draft_len` must therefore be identical
# between arms. Any drift there means the port changed behaviour and the timing
# is void.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
tokens="${2:-512}"
label="${3:-s1}"
first="${4:-1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e135_e87_select_abba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "session commit ${session_commit}"
echo "session worker ${session_worker}"

# PHASE 1: prove the selector reaches the worker AND changes the dispatch.
#
# `qwen35E87SelectEnabled` is `MLX_E87_SELECT != "0"`, so an unset variable and
# a variable that never arrived both run SELECT, and only the literal "0"
# selects the incumbent. The witness therefore reads the arm the dispatch
# recorded for itself, not the parsed environment.
#
# Rule 101. Each leg is checked twice: once against its own arm, which must
# pass, and once against the other arm, which MUST fail.
witness_tokens="${E135_WITNESS_TOKENS:-16}"
for arm in select incumbent; do
  tag="e135${label}w${arm}"
  echo "=== witness ${tag}: arm=${arm} tokens=${witness_tokens} ==="
  out="research/out/${tag}"
  mkdir -p "${out}"
  if [[ "${arm}" == "incumbent" ]]; then export MLX_E87_SELECT=0; else
    export MLX_E87_SELECT=1; fi
  export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${out}/pipelines.json"
  research/e79_trace_leg.sh "${tag}" "${witness_tokens}" --no-trace
  status=$?
  unset MLX_E120_QMV_PIPELINE_LOG MLX_E87_SELECT
  {
    echo "e135_e87_arm=${arm}"
    echo "e135_leg_exit=${status}"
  } >> "${out}/meta.txt"

  python3 research/e135_e87_select_check.py "${out}/pipelines.json" \
    --want "${arm}" | tee "${out}/arm-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e135_e87_select_abba: ${arm} leg did not dispatch the ${arm} arm;" \
         "not timing" >&2
    exit 3
  fi

  other=$([[ "${arm}" == "select" ]] && echo incumbent || echo select)
  echo "--- Rule 101 control: the same check must FAIL for ${other} ---"
  python3 research/e135_e87_select_check.py "${out}/pipelines.json" \
    --want "${other}" | tee "${out}/arm-check-control.txt"
  if [[ "${PIPESTATUS[0]}" == "0" ]]; then
    echo "e135_e87_select_abba: the ${arm} leg also passed the ${other}" \
         "check, so the witness cannot fail and proves nothing" >&2
    exit 4
  fi
  echo "control ok: the witness fails when it should"
done

# PHASE 2: the counterbalanced timed session.
failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in select incumbent incumbent select; do
    position=$((position + 1))
    tag="e135${label}k${rep}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
    if [[ "${arm}" == "incumbent" ]]; then export MLX_E87_SELECT=0; else
      export MLX_E87_SELECT=1; fi
    research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace
    status=$?
    unset MLX_E87_SELECT
    {
      echo "e135_e87_arm=${arm}"
      echo "e135_replicate=${rep}"
      echo "e135_position=${position}"
      echo "e135_leg_index=$(( (rep - first) * 4 + position ))"
      echo "e135_session_commit=${session_commit}"
      echo "e135_session_worker_sha256=${session_worker}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e135_e87_select_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e135_e87_select_abba: ${failures} failed legs"
python3 research/e135_report.py --label "${label}" --session e87
exit $((failures > 0))
